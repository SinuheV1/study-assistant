from __future__ import annotations

from src.chunking import textbook_chunker


def test_is_textbook_heading_accepts_expected_patterns():
    assert textbook_chunker.is_textbook_heading("## 2.1 What Is Statistical Learning?")
    assert textbook_chunker.is_textbook_heading("Chapter 2 Statistical Learning")
    assert textbook_chunker.is_textbook_heading("2.1.4 Supervised Versus Unsupervised Learning")


def test_is_textbook_heading_rejects_non_headings():
    assert not textbook_chunker.is_textbook_heading("expected test MSE")
    assert not textbook_chunker.is_textbook_heading("x" * 141)


def test_extract_page_blocks_supports_page_markers():
    text = "<PAGE 1>\nOne\n--- Page 2 ---\nTwo\n<!-- page: 3 -->\nThree"

    assert textbook_chunker.extract_page_blocks(text, {}) == [
        {"page": 1, "text": "One"},
        {"page": 2, "text": "Two"},
        {"page": 3, "text": "Three"},
    ]


def test_extract_page_blocks_supports_form_feed_and_pages_metadata():
    assert textbook_chunker.extract_page_blocks("First\fSecond", {}) == [
        {"page": 1, "text": "First"},
        {"page": 2, "text": "Second"},
    ]
    assert textbook_chunker.extract_page_blocks(
        "ignored",
        {"pages": [{"page": 10, "text": "Ten"}, "Eleven"]},
    ) == [
        {"page": 10, "text": "Ten"},
        {"page": 2, "text": "Eleven"},
    ]


def test_extract_page_blocks_no_marker_fallback():
    assert textbook_chunker.extract_page_blocks("Plain text", {}) == [
        {"page": None, "text": "Plain text"}
    ]


def test_heading_normalization_and_chapter_detection():
    assert (
        textbook_chunker.normalize_textbook_heading("##  Chapter 2  Learning ")
        == "Chapter 2 Learning"
    )
    assert textbook_chunker.heading_is_chapter("Chapter 2 Learning")
    assert textbook_chunker.heading_is_chapter("Ch. 3 Models")
    assert not textbook_chunker.heading_is_chapter("2.1 Statistical Learning")


def test_get_sentence_overlap_units_budget_and_order():
    units = [{"text": "alpha"}, {"text": "bravo"}, {"text": "charlie"}]

    assert textbook_chunker.get_sentence_overlap_units(units, 0) == []
    assert textbook_chunker.get_sentence_overlap_units(units, 12) == [
        {"text": "bravo"},
        {"text": "charlie"},
    ]


def test_finalize_textbook_chunk_excludes_unknown_pages():
    chunk = textbook_chunker.finalize_textbook_chunk(
        [
            {"text": "First.", "page": 2, "chapter": "Chapter 1", "section": "Intro"},
            {"text": "Second.", "page": None, "chapter": "Chapter 1", "section": "Intro"},
            {"text": "Third.", "page": 4, "chapter": "Chapter 1", "section": "Methods"},
        ]
    )

    assert chunk["text"] == "First. Second. Third."
    assert chunk["page_start"] == 2
    assert chunk["page_end"] == 4
    assert chunk["chapter"] == "Chapter 1"
    assert chunk["section"] == "Methods"
    assert chunk["sections"] == ["Intro", "Methods"]


def test_build_chunks_from_units_respects_size_and_overlap():
    units = [
        {"text": "A" * 20, "page": 1, "chapter": "Ch", "section": "One"},
        {"text": "B" * 20, "page": 1, "chapter": "Ch", "section": "One"},
        {"text": "C" * 20, "page": 2, "chapter": "Ch", "section": "Two"},
    ]

    chunks = textbook_chunker.build_chunks_from_units(
        units,
        target_size=45,
        overlap_size=25,
        min_chunk_size=30,
    )

    assert len(chunks) == 2
    assert chunks[0]["page_start"] == 1
    assert chunks[1]["page_end"] == 2
    assert chunks[1]["text"].startswith("B" * 20)


def test_split_block_into_sentences_uses_stubbed_tokenizer():
    assert textbook_chunker.split_block_into_sentences("First. Second.") == [
        "First.",
        "Second.",
    ]
