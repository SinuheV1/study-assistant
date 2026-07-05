from __future__ import annotations

from src.services import rag_service


def test_infer_week_from_metadata_or_path():
    assert rag_service._infer_week({"week": "week_7"}) == "week_7"
    assert rag_service._infer_week({"file_path": "course/Week-05/notes.md"}) == "week_05"
    assert rag_service._infer_week({"file_path": "course/week5/notes.md"}) == "week_05"
    assert rag_service._infer_week({}) is None


def test_matches_filters_course_case_insensitive_and_week():
    metadata = {"course": "ANLP", "file_path": "course/week_03/notes.md"}

    assert rag_service._matches_filters(metadata, course="anlp")
    assert rag_service._matches_filters(metadata, week="week_03")
    assert not rag_service._matches_filters(metadata, course="stats")
    assert not rag_service._matches_filters(metadata, week="week_04")


def test_score_for_result_precedence_and_invalid_values():
    assert rag_service._score_for_result({"rerank_score": "0.9", "hybrid_score": 0.8}) == 0.9
    assert rag_service._score_for_result({"hybrid_score": 0.8, "similarity": 0.7}) == 0.8
    assert rag_service._score_for_result({"similarity": 0.7, "bm25_score": 3}) == 0.7
    assert rag_service._score_for_result({"bm25_score": 3}) == 3.0
    assert rag_service._score_for_result({"rerank_score": "bad", "similarity": 0.7}) is None


def test_format_search_result_page_logic_and_fallback_source():
    result = rag_service._format_search_result(
        {
            "rank": 1,
            "chunk_text": "Text",
            "similarity": 0.5,
            "metadata": {
                "chunk_id": "chunk_meta",
                "title": "Title",
                "file_path": "course/week_02/file.md",
                "source_type": "notes",
                "course": "ANLP",
                "section": "Intro",
                "page_start": 2,
                "page_end": 4,
            },
        }
    )

    assert result["chunk_id"] == "chunk_meta"
    assert result["source"] == "Title"
    assert result["week"] == "week_02"
    assert result["page"] == "2-4"
    assert result["score"] == 0.5


def test_format_search_result_none_and_single_page():
    no_page = rag_service._format_search_result({"metadata": {}})
    single_page = rag_service._format_search_result(
        {"metadata": {"file_name": "file.pdf", "page_start": 3, "page_end": ""}}
    )

    assert no_page["page"] is None
    assert single_page["page"] == 3
