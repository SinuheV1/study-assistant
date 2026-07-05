from __future__ import annotations

from src.generation import generator


def test_format_page_range_current_cases():
    assert generator.format_page_range(None, None) == "unknown"
    assert generator.format_page_range("", "") == "unknown"
    assert generator.format_page_range(3, 3) == "3"
    assert generator.format_page_range(3, 5) == "3-5"


def test_strip_llm_sources_removes_source_blocks():
    assert generator.strip_llm_sources("Answer\nSources:\n- bad") == "Answer"
    assert generator.strip_llm_sources("Answer\nSource:\n- bad") == "Answer"
    assert generator.strip_llm_sources("") == ""


def test_format_answer_with_sources_deduplicates_top_three_sources():
    results = [
        {"metadata": {"file_name": "a.pdf", "section": "Intro", "page_start": 1, "page_end": 1}},
        {"metadata": {"file_name": "a.pdf", "section": "Intro", "page_start": 1, "page_end": 1}},
        {"metadata": {"file_name": "b.pdf", "section": "Methods", "page_start": 2, "page_end": 4}},
        {"metadata": {"file_name": "c.pdf", "section": "Later", "page_start": 5, "page_end": 5}},
    ]

    formatted = generator.format_answer_with_sources("Answer\nSources:\nignored", results)

    assert formatted == "Answer\n\nSources:\n- a.pdf, Intro, pages 1\n- b.pdf, Methods, pages 2-4"


def test_build_context_block_contains_metadata_and_empty_is_blank():
    assert generator.build_context_block([]) == ""

    context = generator.build_context_block(
        [
            {
                "rank": 2,
                "chunk_id": "chunk_1",
                "chunk_text": "Study text",
                "metadata": {
                    "file_name": "lecture.pdf",
                    "title": "Lecture",
                    "course": "ANLP",
                    "source_type": "lecture_pdfs",
                    "chapter": "Chapter 1",
                    "section": "Intro",
                    "page_start": 1,
                    "page_end": 2,
                },
            }
        ]
    )

    assert "[Context Block 1]" in context
    assert "Rank: 2" in context
    assert "Pages: 1-2" in context
    assert "Citation: lecture.pdf, Intro, pages 1-2" in context
    assert "Study text" in context


def test_build_user_message_contains_context_and_question():
    message = generator.build_user_message("What?", "Context block")

    assert "Context:\nContext block" in message
    assert "Question:\nWhat?" in message
