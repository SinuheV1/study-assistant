from __future__ import annotations

from src.chunking import chunker
from src.ingestion import ingest_text


def _document_metadata() -> dict:
    return {
        "document_id": "doc_test",
        "file_name": "notes.txt",
        "file_path": "notes.txt",
        "source_type": "lecture_pdfs",
        "course": "ANLP",
        "title": "Notes",
    }


def test_is_heading_current_heuristics():
    assert chunker.is_heading("## Overview")
    assert chunker.is_heading("Short Heading")
    assert not chunker.is_heading("<!-- image -->")
    assert not chunker.is_heading("1. numbered item")
    assert not chunker.is_heading("- bullet item")
    assert not chunker.is_heading("This sentence ends.")
    assert not chunker.is_heading("Whatarethedesired")
    assert not chunker.is_heading("x" * 76)


def test_split_into_blocks_strips_and_drops_blank_sections():
    text = " First block \n\n\n Second block\nstill second \n\n"

    assert chunker.split_into_blocks(text) == ["First block", "Second block\nstill second"]


def test_build_chunks_attaches_heading_and_assigns_sections():
    chunks = chunker.build_chunks_from_blocks(
        ["## Topic", "This is topic content.", "## Next", "More content."],
        target_size=200,
        overlap_size=0,
    )

    assert len(chunks) == 1
    assert "## Topic\n\nThis is topic content." in chunks[0]["text"]
    assert chunks[0]["section"] == "Next"
    assert chunks[0]["sections"] == ["Topic", "Next"]


def test_build_chunks_carries_overlap_between_chunks():
    chunks = chunker.build_chunks_from_blocks(
        ["Alpha beta gamma delta.", "Second block that forces a split."],
        target_size=35,
        overlap_size=12,
    )

    assert len(chunks) == 2
    assert chunks[1]["text"].startswith("delta.")


def test_split_oversized_block_uses_sentence_overlap():
    parts = chunker.split_oversized_block(
        "First sentence. Second sentence is here. Third sentence arrives.",
        target_size=38,
        overlap_size=20,
    )

    assert len(parts) >= 2
    assert parts[1].startswith("First sentence.") or parts[1].startswith("Second sentence")


def test_create_chunk_metadata_is_stable_and_complete():
    metadata = _document_metadata()

    first = chunker.create_chunk_metadata(metadata, "chunk text", 0, 2, "Intro", ["Intro"])
    second = chunker.create_chunk_metadata(metadata, "chunk text", 0, 2, "Intro", ["Intro"])

    assert first == second
    assert first["chunk_id"].startswith("chunk_")
    assert first["document_id"] == "doc_test"
    assert first["section"] == "Intro"
    assert first["total_chunks"] == 2


def test_chunk_document_routes_lecture_to_generic_chunking():
    records = chunker.chunk_document(
        "## Intro\n\nA short paragraph.",
        _document_metadata(),
        target_size=200,
        overlap_size=0,
    )

    assert len(records) == 1
    assert records[0]["metadata"]["source_type"] == "lecture_pdfs"
    assert records[0]["metadata"]["section"] == "Intro"


def test_normalize_source_type_known_aliases_match():
    assert chunker.normalize_source_type("lecture") == "lecture_pdfs"
    assert ingest_text.normalize_source_type("lecture_pdf") == "lecture_pdfs"
    assert chunker.normalize_source_type("markdown_notes") == "personal_notes"
    assert ingest_text.normalize_source_type("markdown_notes") == "personal_notes"


def test_normalize_source_type_unknown_divergence_is_current_behavior():
    # Current behavior diverges: backlog H5 can decide whether to consolidate these.
    assert chunker.normalize_source_type("slides") == "generic_text"
    assert ingest_text.normalize_source_type("slides") == "slides"
