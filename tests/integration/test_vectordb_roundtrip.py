from __future__ import annotations

import chromadb

from src.vector_store import vectordb


def test_ephemeral_vector_store_roundtrip_and_delete():
    client = chromadb.EphemeralClient()
    collection = vectordb.get_or_create_collection(client, "test_collection")
    records = [
        {
            "chunk_id": "chunk_a",
            "chunk_text": "alpha",
            "metadata": {"document_id": "doc_a", "course": None},
            "embedding": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "chunk_id": "chunk_b",
            "chunk_text": "beta",
            "metadata": {"document_id": "doc_b", "course": "ANLP"},
            "embedding": [0.0, 1.0, 0.0, 0.0],
        },
        {
            "chunk_id": "missing_embedding",
            "chunk_text": "bad",
            "metadata": {"document_id": "doc_bad"},
            "embedding": None,
        },
    ]

    valid = vectordb.validate_embedded_chunk_records(records)

    assert len(valid) == 2
    assert valid[0]["metadata"]["course"] == ""

    vectordb.add_records_to_collection(collection, records)
    assert vectordb.get_collection_count(collection) == 2

    results = vectordb.query_collection(collection, [1.0, 0.0, 0.0, 0.0], top_k=1)
    assert results["ids"][0] == ["chunk_a"]

    vectordb.delete_by_document_id(collection, "doc_a")
    assert vectordb.get_collection_count(collection) == 1
