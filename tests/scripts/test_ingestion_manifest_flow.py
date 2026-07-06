from __future__ import annotations

import hashlib
import importlib.util
import sys
import uuid
from pathlib import Path

import chromadb
import pytest

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
    monkeypatch.setattr(pipeline, "get_bm25_index", lambda *args, **kwargs: None)
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
    monkeypatch.setattr(pipeline, "get_bm25_index", lambda *args, **kwargs: None)
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
    monkeypatch.setattr(pipeline, "get_bm25_index", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "load_vectordb_collection", lambda reset=False: _collection())

    pipeline.main()

    assert manifest_path.exists()
    saved = manifest_path.read_text(encoding="utf-8")
    assert file_path.as_posix() in saved


def test_manifest_path_from_config_resolves_to_project_root():
    assert pipeline.manifest_path.is_absolute()
    assert pipeline.manifest_path.name == "ingestion_manifest.json"


def test_reset_artifacts_flag_appears_in_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_ingestion_pipeline.py", "--help"])

    with pytest.raises(SystemExit):
        pipeline.parse_args()

    output = capsys.readouterr().out
    assert "--reset-artifacts" in output


def test_reset_generated_artifacts_clears_only_configured_artifact_paths(tmp_path, monkeypatch):
    chunk_dir = tmp_path / "processed" / "chunks"
    embeddings_dir = tmp_path / "processed" / "embeddings"
    extracted_text_dir = tmp_path / "processed" / "extracted_texts"
    bm25_index_dir = tmp_path / "processed" / "bm25_index"
    manifest_path = tmp_path / "ingestion_manifest.json"
    raw_dir = tmp_path / "raw"

    for directory in [chunk_dir, embeddings_dir, extracted_text_dir, bm25_index_dir, raw_dir]:
        directory.mkdir(parents=True)
        (directory / "artifact.txt").write_text("generated", encoding="utf-8")

    (chunk_dir / "nested").mkdir()
    (chunk_dir / "nested" / "nested.txt").write_text("generated", encoding="utf-8")
    manifest_path.write_text("{}", encoding="utf-8")
    (raw_dir / "source.pdf").write_text("source", encoding="utf-8")

    monkeypatch.setattr(pipeline, "chunks_dir", chunk_dir)
    monkeypatch.setattr(pipeline, "embeddings_dir", embeddings_dir)
    monkeypatch.setattr(pipeline, "extracted_text_dir", extracted_text_dir)
    monkeypatch.setattr(pipeline, "bm25_index_dir", bm25_index_dir)
    monkeypatch.setattr(pipeline, "manifest_path", manifest_path)

    assert pipeline.reset_generated_artifacts()

    assert chunk_dir.exists()
    assert list(chunk_dir.iterdir()) == []
    assert embeddings_dir.exists()
    assert list(embeddings_dir.iterdir()) == []
    assert extracted_text_dir.exists()
    assert list(extracted_text_dir.iterdir()) == []
    assert not bm25_index_dir.exists()
    assert not manifest_path.exists()
    assert raw_dir.exists()
    assert (raw_dir / "source.pdf").exists()


def test_reset_artifacts_refuses_path_equal_to_data_raw(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    raw_dir = project_root / "data" / "raw"
    chunk_dir = project_root / "data" / "processed" / "chunks"
    embeddings_dir = project_root / "data" / "processed" / "embeddings"
    extracted_text_dir = project_root / "data" / "processed" / "extracted_texts"
    bm25_index_dir = project_root / "data" / "processed" / "bm25_index"
    manifest_path = project_root / "data" / "ingestion_manifest.json"

    for directory in [raw_dir, chunk_dir, embeddings_dir, extracted_text_dir, bm25_index_dir]:
        directory.mkdir(parents=True)
        (directory / "artifact.txt").write_text("keep me", encoding="utf-8")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")
    (raw_dir / "source.pdf").write_text("source", encoding="utf-8")

    monkeypatch.setattr(pipeline, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(pipeline, "chunks_dir", raw_dir)
    monkeypatch.setattr(pipeline, "embeddings_dir", embeddings_dir)
    monkeypatch.setattr(pipeline, "extracted_text_dir", extracted_text_dir)
    monkeypatch.setattr(pipeline, "bm25_index_dir", bm25_index_dir)
    monkeypatch.setattr(pipeline, "manifest_path", manifest_path)

    assert not pipeline.reset_generated_artifacts()
    assert (raw_dir / "source.pdf").exists()
    assert (embeddings_dir / "artifact.txt").exists()
    assert (extracted_text_dir / "artifact.txt").exists()
    assert bm25_index_dir.exists()
    assert manifest_path.exists()


def test_reset_artifacts_refuses_path_inside_data_raw(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    raw_dir = project_root / "data" / "raw"
    unsafe_inside_raw = raw_dir / "nested"
    chunk_dir = project_root / "data" / "processed" / "chunks"
    embeddings_dir = project_root / "data" / "processed" / "embeddings"
    extracted_text_dir = project_root / "data" / "processed" / "extracted_texts"
    bm25_index_dir = project_root / "data" / "processed" / "bm25_index"
    manifest_path = project_root / "data" / "ingestion_manifest.json"

    for directory in [
        raw_dir,
        unsafe_inside_raw,
        chunk_dir,
        embeddings_dir,
        extracted_text_dir,
        bm25_index_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "artifact.txt").write_text("keep me", encoding="utf-8")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")
    (unsafe_inside_raw / "source.md").write_text("source", encoding="utf-8")

    monkeypatch.setattr(pipeline, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(pipeline, "chunks_dir", chunk_dir)
    monkeypatch.setattr(pipeline, "embeddings_dir", unsafe_inside_raw)
    monkeypatch.setattr(pipeline, "extracted_text_dir", extracted_text_dir)
    monkeypatch.setattr(pipeline, "bm25_index_dir", bm25_index_dir)
    monkeypatch.setattr(pipeline, "manifest_path", manifest_path)

    assert not pipeline.reset_generated_artifacts()
    assert (unsafe_inside_raw / "source.md").exists()
    assert (chunk_dir / "artifact.txt").exists()
    assert (extracted_text_dir / "artifact.txt").exists()
    assert bm25_index_dir.exists()
    assert manifest_path.exists()


def test_reset_artifacts_aborts_without_partial_delete_when_one_target_is_unsafe(
    tmp_path, monkeypatch
):
    project_root = tmp_path / "project"
    raw_dir = project_root / "data" / "raw"
    chunk_dir = project_root / "data" / "processed" / "chunks"
    embeddings_dir = project_root / "data" / "processed" / "embeddings"
    extracted_text_dir = project_root / "data" / "processed" / "extracted_texts"
    bm25_index_dir = project_root / "data" / "processed" / "bm25_index"
    manifest_path = project_root / "data" / "ingestion_manifest.json"

    for directory in [raw_dir, chunk_dir, embeddings_dir, extracted_text_dir, bm25_index_dir]:
        directory.mkdir(parents=True)
        (directory / "artifact.txt").write_text("keep me", encoding="utf-8")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")
    (raw_dir / "source.txt").write_text("source", encoding="utf-8")

    monkeypatch.setattr(pipeline, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(pipeline, "chunks_dir", chunk_dir)
    monkeypatch.setattr(pipeline, "embeddings_dir", embeddings_dir)
    monkeypatch.setattr(pipeline, "extracted_text_dir", extracted_text_dir)
    monkeypatch.setattr(pipeline, "bm25_index_dir", raw_dir)
    monkeypatch.setattr(pipeline, "manifest_path", manifest_path)

    assert not pipeline.reset_generated_artifacts()
    assert (chunk_dir / "artifact.txt").exists()
    assert (embeddings_dir / "artifact.txt").exists()
    assert (extracted_text_dir / "artifact.txt").exists()
    assert (bm25_index_dir / "artifact.txt").exists()
    assert manifest_path.exists()
    assert (raw_dir / "source.txt").exists()


@pytest.mark.parametrize(
    "critical_path",
    [".git", "src", "configs", "scripts", "docs", "README.md"],
)
def test_reset_artifacts_refuses_repo_critical_paths(tmp_path, critical_path):
    project_root = tmp_path / "project"
    raw_dir = project_root / "data" / "raw"
    target = project_root / critical_path

    if target.suffix:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("keep me", encoding="utf-8")
    else:
        target.mkdir(parents=True)
        (target / "keep.txt").write_text("keep me", encoding="utf-8")

    raw_dir.mkdir(parents=True)

    assert not pipeline._is_safe_deletion_target(
        target,
        project_root=project_root,
        raw_dir=raw_dir,
    )


def test_reset_artifacts_cli_aborts_before_delete_or_ingest_for_repo_critical_path(
    tmp_path, monkeypatch
):
    project_root = tmp_path / "project"
    file_path = tmp_path / "note.txt"
    file_path.write_text("do not ingest", encoding="utf-8")
    raw_dir = project_root / "data" / "raw"
    git_dir = project_root / ".git"
    embeddings_dir = project_root / "data" / "processed" / "embeddings"
    extracted_text_dir = project_root / "data" / "processed" / "extracted_texts"
    bm25_index_dir = project_root / "data" / "processed" / "bm25_index"
    manifest_path = project_root / "data" / "ingestion_manifest.json"

    for directory in [raw_dir, git_dir, embeddings_dir, extracted_text_dir, bm25_index_dir]:
        directory.mkdir(parents=True)
        (directory / "artifact.txt").write_text("keep me", encoding="utf-8")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")

    def fail_load_vectordb_collection(reset=False):
        raise AssertionError("reset-artifacts should abort before Chroma reset")

    def fail_process_one_file(*args, **kwargs):
        raise AssertionError("reset-artifacts should abort before ingestion")

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_ingestion_pipeline.py", "--file", str(file_path), "--reset-artifacts"],
    )
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(pipeline, "chunks_dir", git_dir)
    monkeypatch.setattr(pipeline, "embeddings_dir", embeddings_dir)
    monkeypatch.setattr(pipeline, "extracted_text_dir", extracted_text_dir)
    monkeypatch.setattr(pipeline, "bm25_index_dir", bm25_index_dir)
    monkeypatch.setattr(pipeline, "manifest_path", manifest_path)
    monkeypatch.setattr(pipeline, "load_vectordb_collection", fail_load_vectordb_collection)
    monkeypatch.setattr(pipeline, "process_one_file", fail_process_one_file)

    pipeline.main()

    assert (git_dir / "artifact.txt").exists()
    assert (embeddings_dir / "artifact.txt").exists()
    assert (extracted_text_dir / "artifact.txt").exists()
    assert (bm25_index_dir / "artifact.txt").exists()
    assert manifest_path.exists()
    assert raw_dir.exists()


def test_reset_artifacts_cli_resets_collection_manifest_and_skip_logic(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    file_path = tmp_path / "note.txt"
    file_path.write_text("reset artifacts", encoding="utf-8")
    chunk_dir = project_root / "data" / "processed" / "chunks"
    embeddings_dir = project_root / "data" / "processed" / "embeddings"
    extracted_text_dir = project_root / "data" / "processed" / "extracted_texts"
    bm25_index_dir = project_root / "data" / "processed" / "bm25_index"
    manifest_path = project_root / "data" / "ingestion_manifest.json"
    raw_dir = project_root / "data" / "raw"
    captured = {}

    for directory in [chunk_dir, embeddings_dir, extracted_text_dir, bm25_index_dir, raw_dir]:
        directory.mkdir(parents=True)
        (directory / "artifact.txt").write_text("generated", encoding="utf-8")

    manifest_path.write_text('{"documents": {"old": {"status": "success"}}}', encoding="utf-8")
    (raw_dir / "source.md").write_text("source", encoding="utf-8")

    class FakeCollection:
        def count(self):
            return 0

    def fake_load_vectordb_collection(reset=False):
        captured["reset"] = reset
        return FakeCollection()

    def fake_process_one_file(file_path, collection, manifest, skip_unchanged=True):
        captured["manifest"] = manifest
        captured["skip_unchanged"] = skip_unchanged
        return {
            "file_path": str(file_path),
            "status": "success",
            "chunks_created": 0,
            "embeddings_created": 0,
        }

    def fail_load_manifest(path):
        raise AssertionError("reset-artifacts should not load the existing manifest")

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_ingestion_pipeline.py", "--file", str(file_path), "--reset-artifacts"],
    )
    monkeypatch.setattr(pipeline, "chunks_dir", chunk_dir)
    monkeypatch.setattr(pipeline, "embeddings_dir", embeddings_dir)
    monkeypatch.setattr(pipeline, "extracted_text_dir", extracted_text_dir)
    monkeypatch.setattr(pipeline, "bm25_index_dir", bm25_index_dir)
    monkeypatch.setattr(pipeline, "manifest_path", manifest_path)
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(pipeline, "load_manifest", fail_load_manifest)
    monkeypatch.setattr(pipeline, "save_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "get_bm25_index", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "load_vectordb_collection", fake_load_vectordb_collection)
    monkeypatch.setattr(pipeline, "process_one_file", fake_process_one_file)

    pipeline.main()

    assert captured["reset"] is True
    assert captured["skip_unchanged"] is False
    assert captured["manifest"]["documents"] == {}
    assert list(chunk_dir.iterdir()) == []
    assert list(embeddings_dir.iterdir()) == []
    assert list(extracted_text_dir.iterdir()) == []
    assert not bm25_index_dir.exists()
    assert not manifest_path.exists()
    assert raw_dir.exists()
    assert (raw_dir / "source.md").exists()
