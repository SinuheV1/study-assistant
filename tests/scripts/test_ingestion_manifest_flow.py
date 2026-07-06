from __future__ import annotations

import hashlib
import importlib.util
import sys
import uuid
from pathlib import Path

import chromadb

from src.ingestion.manifest import (
    compute_file_hash,
    create_empty_manifest,
    update_manifest_record,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def import_script_module(script_name: str):
    module_path = PROJECT_ROOT / "scripts" / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(script_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


pipeline = import_script_module("run_ingestion_pipeline")


def _collection():
    client = chromadb.EphemeralClient()
    return client.create_collection(f"manifest_flow_{uuid.uuid4().hex}")


def _document_id(file_path: Path) -> str:
    digest = hashlib.sha256(file_path.as_posix().encode("utf-8")).hexdigest()[:12]
    return f"doc_{digest}"


def _chunk_id(document_id: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"chunk_{document_id}_{digest}"


def _patch_fast_ingestion(monkeypatch):
    monkeypatch.setattr(pipeline, "save_extracted_text", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "save_json", lambda *args, **kwargs: None)

    def fake_ingest_file_type(file_path):
        file_path = Path(file_path)
        text = file_path.read_text(encoding="utf-8")
        metadata = {
            "document_id": _document_id(file_path),
            "file_name": file_path.name,
            "file_path": file_path.as_posix(),
            "file_size": file_path.stat().st_size,
            "source_type": "personal_notes",
            "raw_source_type": "notes",
            "course": "TEST101",
            "title": file_path.stem,
        }
        return {"raw_text": text, "cleaned_text": text, "metadata": metadata}, file_path.stem

    def fake_chunk_document(cleaned_text, document_metadata, target_size, overlap_size):
        if cleaned_text == "NO_CHUNKS":
            return []

        document_id = document_metadata["document_id"]
        return [
            {
                "chunk_id": _chunk_id(document_id, cleaned_text),
                "chunk_text": cleaned_text,
                "metadata": {
                    "document_id": document_id,
                    "file_name": document_metadata["file_name"],
                    "file_path": document_metadata["file_path"],
                    "source_type": document_metadata["source_type"],
                    "course": document_metadata["course"],
                    "title": document_metadata["title"],
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "section": "Unknown",
                },
            }
        ]

    def fake_embed_chunks(chunk_records, model_name):
        if chunk_records[0]["chunk_text"] == "NO_EMBED":
            return []

        return [
            {
                **record,
                "embedding": [1.0, 0.0, 0.0, 0.0],
            }
            for record in chunk_records
        ]

    monkeypatch.setattr(pipeline, "ingest_file_type", fake_ingest_file_type)
    monkeypatch.setattr(pipeline, "chunk_document", fake_chunk_document)
    monkeypatch.setattr(pipeline, "embed_chunks", fake_embed_chunks)


def _process(file_path: Path, collection, manifest: dict, skip_unchanged: bool = True):
    return pipeline.process_one_file(
        file_path=file_path,
        collection=collection,
        manifest=manifest,
        skip_unchanged=skip_unchanged,
    )


def test_first_ingestion_records_success_in_manifest(tmp_path, monkeypatch):
    _patch_fast_ingestion(monkeypatch)
    file_path = tmp_path / "note.txt"
    file_path.write_text("first version", encoding="utf-8")
    manifest = create_empty_manifest()
    collection = _collection()

    result = _process(file_path, collection, manifest)

    record = manifest["documents"][file_path.as_posix()]
    assert result["status"] == "success"
    assert record["status"] == "success"
    assert record["file_hash"] == compute_file_hash(file_path)
    assert record["chunks_created"] == 1
    assert collection.count() == 1


def test_second_unchanged_run_skips_successful_file(tmp_path, monkeypatch):
    _patch_fast_ingestion(monkeypatch)
    file_path = tmp_path / "note.txt"
    file_path.write_text("same version", encoding="utf-8")
    manifest = create_empty_manifest()
    collection = _collection()

    first = _process(file_path, collection, manifest)
    first_ingested_at = manifest["documents"][file_path.as_posix()]["ingested_at"]
    second = _process(file_path, collection, manifest)

    assert first["status"] == "success"
    assert second["status"] == "skipped"
    assert manifest["documents"][file_path.as_posix()]["ingested_at"] == first_ingested_at
    assert collection.count() == 1


def test_failed_files_are_retried_even_when_hash_is_unchanged(tmp_path, monkeypatch):
    _patch_fast_ingestion(monkeypatch)
    file_path = tmp_path / "note.txt"
    file_path.write_text("retry me", encoding="utf-8")
    file_hash = compute_file_hash(file_path)
    manifest = create_empty_manifest()
    update_manifest_record(
        manifest,
        file_path,
        {
            "document_id": _document_id(file_path),
            "file_name": file_path.name,
            "file_path": file_path.as_posix(),
        },
        file_hash,
        chunks_created=0,
        status="failed",
    )

    result = _process(file_path, _collection(), manifest)

    assert result["status"] == "success"
    assert manifest["documents"][file_path.as_posix()]["status"] == "success"


def test_changed_file_reingests_and_replaces_old_chunks(tmp_path, monkeypatch):
    _patch_fast_ingestion(monkeypatch)
    file_path = tmp_path / "note.txt"
    file_path.write_text("old content", encoding="utf-8")
    manifest = create_empty_manifest()
    collection = _collection()
    document_id = _document_id(file_path)

    first = _process(file_path, collection, manifest)
    old_ids = collection.get(where={"document_id": document_id})["ids"]

    file_path.write_text("new content", encoding="utf-8")
    second = _process(file_path, collection, manifest)
    new_ids = collection.get(where={"document_id": document_id})["ids"]

    assert first["document_id"] == document_id
    assert second["status"] == "success"
    assert len(old_ids) == 1
    assert len(new_ids) == 1
    assert old_ids[0] not in new_ids
    assert new_ids == [_chunk_id(document_id, "new content")]


def test_force_reingest_bypasses_skip_logic(tmp_path, monkeypatch):
    _patch_fast_ingestion(monkeypatch)
    file_path = tmp_path / "note.txt"
    file_path.write_text("force me", encoding="utf-8")
    manifest = create_empty_manifest()
    collection = _collection()

    first = _process(file_path, collection, manifest)
    second = _process(file_path, collection, manifest, skip_unchanged=False)

    assert first["status"] == "success"
    assert second["status"] == "success"
    assert collection.count() == 1


def test_reset_collection_cli_bypasses_skip_logic(tmp_path, monkeypatch):
    file_path = tmp_path / "note.txt"
    file_path.write_text("reset me", encoding="utf-8")
    captured = {}

    class FakeCollection:
        def count(self):
            return 0

    def fake_load_vectordb_collection(reset=False):
        captured["reset"] = reset
        return FakeCollection()

    def fake_process_one_file(file_path, collection, manifest, skip_unchanged=True):
        captured["skip_unchanged"] = skip_unchanged
        return {
            "file_path": str(file_path),
            "status": "success",
            "chunks_created": 0,
            "embeddings_created": 0,
        }

    monkeypatch.setattr(
        sys, "argv", ["run_ingestion_pipeline.py", "--file", str(file_path), "--reset-collection"]
    )
    monkeypatch.setattr(pipeline, "load_manifest", lambda path: create_empty_manifest())
    monkeypatch.setattr(pipeline, "save_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "load_vectordb_collection", fake_load_vectordb_collection)
    monkeypatch.setattr(pipeline, "process_one_file", fake_process_one_file)

    pipeline.main()

    assert captured["reset"] is True
    assert captured["skip_unchanged"] is False


def test_reset_manifest_cli_starts_from_empty_manifest(tmp_path, monkeypatch):
    file_path = tmp_path / "note.txt"
    file_path.write_text("reset manifest", encoding="utf-8")
    captured = {}

    class FakeCollection:
        def count(self):
            return 0

    def fake_process_one_file(file_path, collection, manifest, skip_unchanged=True):
        captured["manifest"] = manifest
        return {
            "file_path": str(file_path),
            "status": "success",
            "chunks_created": 0,
            "embeddings_created": 0,
        }

    def fail_load_manifest(path):
        raise AssertionError("reset-manifest should not load an existing manifest")

    monkeypatch.setattr(
        sys, "argv", ["run_ingestion_pipeline.py", "--file", str(file_path), "--reset-manifest"]
    )
    monkeypatch.setattr(pipeline, "load_manifest", fail_load_manifest)
    monkeypatch.setattr(pipeline, "save_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "load_vectordb_collection", lambda reset=False: FakeCollection())
    monkeypatch.setattr(pipeline, "process_one_file", fake_process_one_file)

    pipeline.main()

    assert captured["manifest"]["version"] == 1
    assert captured["manifest"]["documents"] == {}


def test_old_vectors_are_not_deleted_when_embedding_fails(tmp_path, monkeypatch):
    _patch_fast_ingestion(monkeypatch)
    file_path = tmp_path / "note.txt"
    file_path.write_text("working content", encoding="utf-8")
    manifest = create_empty_manifest()
    collection = _collection()
    document_id = _document_id(file_path)

    first = _process(file_path, collection, manifest)
    old_ids = collection.get(where={"document_id": document_id})["ids"]

    file_path.write_text("NO_EMBED", encoding="utf-8")
    second = _process(file_path, collection, manifest)
    ids_after_failure = collection.get(where={"document_id": document_id})["ids"]

    assert first["status"] == "success"
    assert second["status"] == "failed"
    assert ids_after_failure == old_ids
    assert manifest["documents"][file_path.as_posix()]["status"] == "failed"


def test_summary_includes_skipped_files(capsys):
    class FakeCollection:
        def count(self):
            return 2

    pipeline.print_batch_summary(
        [
            {
                "file_path": "processed.txt",
                "status": "success",
                "chunks_created": 2,
                "embeddings_created": 2,
            },
            {
                "file_path": "skipped.txt",
                "status": "skipped",
                "chunks_created": 0,
                "embeddings_created": 0,
            },
        ],
        FakeCollection(),
    )

    output = capsys.readouterr().out
    assert "Skipped: 1" in output
    assert "Skipped unchanged files:" in output
    assert "skipped.txt" in output


def test_manifest_is_saved_after_each_file_and_at_end(tmp_path, monkeypatch):
    _patch_fast_ingestion(monkeypatch)
    file_path = tmp_path / "note.txt"
    file_path.write_text("save me", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"

    monkeypatch.setattr(sys, "argv", ["run_ingestion_pipeline.py", "--file", str(file_path)])
    monkeypatch.setattr(pipeline, "manifest_path", manifest_path)
    monkeypatch.setattr(pipeline, "load_vectordb_collection", lambda reset=False: _collection())

    pipeline.main()

    assert manifest_path.exists()
    saved = manifest_path.read_text(encoding="utf-8")
    assert file_path.as_posix() in saved


def test_manifest_path_from_config_resolves_to_project_root():
    assert pipeline.manifest_path.is_absolute()
    assert pipeline.manifest_path.name == "ingestion_manifest.json"
