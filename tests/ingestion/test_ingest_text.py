from __future__ import annotations

from pathlib import Path

from src.ingestion import ingest_text


def test_detect_text_file_type_accepts_text_and_markdown():
    assert ingest_text.detect_text_file_type("notes.txt") == ".txt"
    assert ingest_text.detect_text_file_type("notes.md") == ".md"
    assert ingest_text.detect_text_file_type("notes.pdf") is None


def test_infer_course_and_source_from_path():
    path = Path("workspace/raw/lecture_pdfs/ANLP/lecture.txt")

    assert ingest_text.infer_source_from_path(path.as_posix()) == "lecture_pdfs"
    assert ingest_text.infer_course_from_path(path.as_posix()) == "ANLP"


def test_infer_path_without_raw_returns_none():
    path = Path("workspace/lecture_pdfs/ANLP/lecture.txt")

    assert ingest_text.infer_source_from_path(path.as_posix()) is None
    assert ingest_text.infer_course_from_path(path.as_posix()) is None


def test_infer_direct_file_under_source_has_no_course():
    path = Path("workspace/raw/lecture_pdfs/lecture.txt")

    assert ingest_text.infer_source_from_path(path.as_posix()) == "lecture_pdfs"
    assert ingest_text.infer_course_from_path(path.as_posix()) is None


def test_build_document_metadata_is_stable_for_path(tmp_path):
    file_path = tmp_path / "workspace" / "raw" / "notes" / "ANLP" / "note.md"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("hello", encoding="utf-8")

    first = ingest_text.build_document_metadata(file_path)
    second = ingest_text.build_document_metadata(file_path)

    assert first["document_id"] == second["document_id"]
    assert first["file_path"] == file_path.as_posix()
    assert first["source_type"] == "personal_notes"
    assert first["raw_source_type"] == "notes"
    assert first["course"] == "ANLP"
    assert first["file_type"] == ".md"
    assert first["file_size"] == 5


def test_ingest_text_document_returns_record(tmp_path):
    file_path = tmp_path / "workspace" / "raw" / "notes" / "ANLP" / "note.txt"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("Hello\nwrapped line", encoding="utf-8")

    record = ingest_text.ingest_text_document(file_path)

    assert record["file_type"] == ".txt"
    assert record["raw_text"] == "Hello\nwrapped line"
    assert record["metadata"]["course"] == "ANLP"
