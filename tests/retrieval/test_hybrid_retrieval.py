from __future__ import annotations

from src.retrieval import hybrid_retrieval


def test_normalize_scores_empty_missing_single_and_equal_scores():
    assert hybrid_retrieval.normalize_scores([], "score", "score_norm") == []
    assert hybrid_retrieval.normalize_scores([{"chunk_id": "a"}], "score", "score_norm") == [
        {"chunk_id": "a", "score_norm": 0.0}
    ]
    assert hybrid_retrieval.normalize_scores(
        [{"score": 2.0}, {"score": 2.0}],
        "score",
        "score_norm",
    ) == [{"score": 2.0, "score_norm": 1.0}, {"score": 2.0, "score_norm": 1.0}]


def test_normalize_scores_min_max_spread():
    results = hybrid_retrieval.normalize_scores(
        [{"score": 2.0}, {"score": 4.0}, {"score": None}],
        "score",
        "score_norm",
    )

    assert [result["score_norm"] for result in results] == [0.0, 1.0, 0.0]


def test_merge_retrieval_results_combines_overlapping_chunk_ids():
    dense = [{"chunk_id": "a", "rank": 1, "similarity": 0.8, "dense_score_norm": 1.0}]
    bm25 = [
        {"chunk_id": "a", "rank": 2, "bm25_score": 3.0, "bm25_score_norm": 0.5},
        {"chunk_id": "b", "rank": 1, "bm25_score": 5.0, "bm25_score_norm": 1.0},
    ]

    merged = {
        result["chunk_id"]: result
        for result in hybrid_retrieval.merge_retrieval_results(dense, bm25)
    }

    assert merged["a"]["dense_rank"] == 1
    assert merged["a"]["bm25_rank"] == 2
    assert merged["b"]["dense_rank"] is None
    assert merged["b"]["bm25_score_norm"] == 1.0


def test_compute_hybrid_score_sorts_and_applies_section_penalty():
    results = [
        {
            "chunk_id": "exercise",
            "dense_score_norm": 1.0,
            "bm25_score_norm": 0.0,
            "metadata": {"section": "Exercises"},
        },
        {
            "chunk_id": "normal",
            "dense_score_norm": 0.5,
            "bm25_score_norm": 0.0,
            "metadata": {"section": "Chapter 1"},
        },
    ]

    scored = hybrid_retrieval.compute_hybrid_score(results, alpha=1.0, query="define learning")

    assert scored[0]["chunk_id"] == "normal"
    assert scored[1]["section_penalty"] == 0.25
    assert scored[1]["hybrid_score"] == 0.25


def test_compute_hybrid_score_alpha_zero_uses_bm25_component():
    scored = hybrid_retrieval.compute_hybrid_score(
        [
            {"chunk_id": "dense", "dense_score_norm": 1.0, "bm25_score_norm": 0.0},
            {"chunk_id": "bm25", "dense_score_norm": 0.0, "bm25_score_norm": 1.0},
        ],
        alpha=0.0,
        query="normal question",
    )

    assert scored[0]["chunk_id"] == "bm25"


def test_practice_query_and_exercise_section_detection():
    assert hybrid_retrieval.is_practice_query("Give me practice problems")
    assert not hybrid_retrieval.is_practice_query("What is regression?")
    assert hybrid_retrieval.is_exercise_section({"metadata": {"section": "Chapter 2 Exercises"}})
    assert not hybrid_retrieval.is_exercise_section(
        {"metadata": {"section": "2.1.5 Regression Versus Classification Problems"}}
    )


def test_filter_exercise_sections_current_fallback_behavior():
    results = [
        {"chunk_id": "exercise", "metadata": {"section": "Exercises"}},
        {"chunk_id": "normal", "metadata": {"section": "Overview"}},
    ]

    assert hybrid_retrieval.filter_exercise_sections(results, "What is learning?") == [results[1]]
    assert hybrid_retrieval.filter_exercise_sections(results, "practice questions") == results
    assert hybrid_retrieval.filter_exercise_sections(
        [results[0]],
        "What is learning?",
        min_results=1,
    ) == [results[0]]
