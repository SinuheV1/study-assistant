from __future__ import annotations

from src.ingestion import manifest


def test_load_manifest_missing_or_malformed_returns_empty(tmp_path):
    missing = tmp_path / "missing.json"
    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")

    assert manifest.load_manifest(missing)["documents"] == {}
    assert manifest.load_manifest(malformed)["documents"] == {}


def test_save_and_load_manifest_roundtrip(tmp_path):
    path = tmp_path / "manifest.json"
    data = manifest.create_empty_manifest()
    data["documents"]["file.txt"] = {"status": "success"}

    manifest.save_manifest(data, path)

    loaded = manifest.load_manifest(path)
    assert loaded["version"] == 1
    assert loaded["documents"]["file.txt"]["status"] == "success"


def test_compute_file_hash_changes_with_content_not_mtime(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("one", encoding="utf-8")
    first = manifest.compute_file_hash(file_path)

    file_path.touch()
    assert manifest.compute_file_hash(file_path) == first

    file_path.write_text("two", encoding="utf-8")
    assert manifest.compute_file_hash(file_path) != first
    assert manifest.compute_file_hash(tmp_path / "missing.txt") is None
    assert manifest.compute_file_hash(tmp_path) is None


def test_is_file_unchanged_matrix(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("one", encoding="utf-8")
    file_hash = manifest.compute_file_hash(file_path)
    data = manifest.create_empty_manifest()

    assert not manifest.is_file_unchanged(data, file_path, file_hash)

    manifest.update_manifest_record(
        data,
        file_path,
        {"document_id": "doc", "file_name": "file.txt", "file_path": file_path.as_posix()},
        file_hash,
        chunks_created=1,
        status="failed",
    )
    assert not manifest.is_file_unchanged(data, file_path, file_hash)

    manifest.update_manifest_record(
        data,
        file_path,
        {"document_id": "doc", "file_name": "file.txt", "file_path": file_path.as_posix()},
        file_hash,
        chunks_created=1,
        status="success",
    )
    assert manifest.is_file_unchanged(data, file_path, file_hash)
    assert not manifest.is_file_unchanged(data, file_path, "different")
    assert not manifest.is_file_unchanged(data, file_path, None)


def test_update_manifest_record_shape(tmp_path):
    file_path = tmp_path / "file.txt"
    data = manifest.create_empty_manifest()
    updated = manifest.update_manifest_record(
        data,
        file_path,
        {
            "document_id": "doc",
            "file_name": "file.txt",
            "file_path": file_path.as_posix(),
            "file_size": 10,
            "source_type": "personal_notes",
            "raw_source_type": "notes",
            "course": "ANLP",
            "title": "file",
        },
        "abc",
        chunks_created=3,
        status="success",
    )

    record = updated["documents"][file_path.as_posix()]
    assert record["document_id"] == "doc"
    assert record["file_hash"] == "abc"
    assert record["chunks_created"] == 3
    assert record["status"] == "success"
