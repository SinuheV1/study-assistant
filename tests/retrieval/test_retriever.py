from __future__ import annotations

from src.retrieval import retriever


def test_format_retrieval_results_assigns_rank_and_similarity():
    raw = {
        "ids": [["chunk_a", "chunk_b"]],
        "documents": [["Alpha", "Beta"]],
        "metadatas": [[{"title": "A"}, {"title": "B"}]],
        "distances": [[0.0, 1.0]],
    }

    results = retriever.format_retrieval_results(raw)

    assert results == [
        {
            "rank": 1,
            "chunk_id": "chunk_a",
            "chunk_text": "Alpha",
            "metadata": {"title": "A"},
            "distance": 0.0,
            "similarity": 1.0,
        },
        {
            "rank": 2,
            "chunk_id": "chunk_b",
            "chunk_text": "Beta",
            "metadata": {"title": "B"},
            "distance": 1.0,
            "similarity": 0.5,
        },
    ]
