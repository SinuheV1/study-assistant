from __future__ import annotations

import json

from src.retrieval import bm25_retriever, hybrid_retrieval


def _chunk_records() -> list[dict]:
    return [
        {
            "chunk_id": "chunk_least_squares",
            "chunk_text": "Least squares fits a linear regression model.",
            "metadata": {"document_id": "doc_stats", "section": "Regression"},
        },
        {
            "chunk_id": "chunk_knn",
            "chunk_text": "K nearest neighbors predicts using nearby observations.",
            "metadata": {"document_id": "doc_stats", "section": "KNN"},
        },
        {
            "chunk_id": "chunk_bias_variance",
            "chunk_text": "Bias and variance describe prediction error tradeoffs.",
            "metadata": {"document_id": "doc_stats", "section": "Bias Variance"},
        },
    ]


def _write_chunks(chunk_dir, records=None, file_name="chunks.json"):
    chunk_dir.mkdir(parents=True, exist_ok=True)
    path = chunk_dir / file_name
    path.write_text(json.dumps(records or _chunk_records()), encoding="utf-8")
    return path


def _ids_and_scores(results):
    return [(result["chunk_id"], result["bm25_score"]) for result in results]


def test_build_save_load_roundtrip_matches_fresh_bm25(tmp_path):
    chunk_dir = tmp_path / "chunks"
    index_dir = tmp_path / "bm25_index"
    records = _chunk_records()
    _write_chunks(chunk_dir, records)
    fingerprint = bm25_retriever.compute_bm25_fingerprint(chunk_dir)

    saved = bm25_retriever.build_and_save_bm25_index(records, index_dir, fingerprint)
    loaded = bm25_retriever.load_bm25_index(index_dir)

    fresh_results = bm25_retriever.bm25_retrieve("least squares regression", records, top_k=3)
    saved_results = bm25_retriever.bm25_retrieve_from_index(
        "least squares regression",
        saved,
        top_k=3,
    )
    loaded_results = bm25_retriever.bm25_retrieve_from_index(
        "least squares regression",
        loaded,
        top_k=3,
    )

    assert _ids_and_scores(saved_results) == _ids_and_scores(fresh_results)
    assert _ids_and_scores(loaded_results) == _ids_and_scores(fresh_results)


def test_get_bm25_index_loads_unchanged_cache_without_rebuilding(tmp_path, monkeypatch):
    chunk_dir = tmp_path / "chunks"
    index_dir = tmp_path / "bm25_index"
    _write_chunks(chunk_dir)

    first = bm25_retriever.get_bm25_index(chunk_dir, index_dir)

    def fail_rebuild(*args, **kwargs):
        raise AssertionError("unchanged cache should load without rebuilding")

    monkeypatch.setattr(bm25_retriever, "build_and_save_bm25_index", fail_rebuild)
    second = bm25_retriever.get_bm25_index(chunk_dir, index_dir)

    assert second.fingerprint == first.fingerprint
    assert [record["chunk_id"] for record in second.records] == [
        record["chunk_id"] for record in first.records
    ]


def test_chunk_artifact_change_invalidates_cache_and_rebuilds(tmp_path, monkeypatch):
    chunk_dir = tmp_path / "chunks"
    index_dir = tmp_path / "bm25_index"
    chunk_file = _write_chunks(chunk_dir)
    bm25_retriever.get_bm25_index(chunk_dir, index_dir)
    calls = {"count": 0}
    original_build = bm25_retriever.build_and_save_bm25_index

    def spy_build(*args, **kwargs):
        calls["count"] += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(bm25_retriever, "build_and_save_bm25_index", spy_build)
    changed_records = _chunk_records() + [
        {
            "chunk_id": "chunk_new",
            "chunk_text": "A new chunk changes the chunk artifact fingerprint.",
            "metadata": {"document_id": "doc_new"},
        }
    ]
    chunk_file.write_text(json.dumps(changed_records), encoding="utf-8")

    rebuilt = bm25_retriever.get_bm25_index(chunk_dir, index_dir)

    assert calls["count"] == 1
    assert [record["chunk_id"] for record in rebuilt.records][-1] == "chunk_new"


def test_bm25s_version_change_invalidates_cache(tmp_path, monkeypatch):
    chunk_dir = tmp_path / "chunks"
    index_dir = tmp_path / "bm25_index"
    _write_chunks(chunk_dir)
    bm25_retriever.get_bm25_index(chunk_dir, index_dir)
    calls = {"count": 0}
    original_build = bm25_retriever.build_and_save_bm25_index

    def spy_build(*args, **kwargs):
        calls["count"] += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(bm25_retriever.bm25s, "__version__", "999.0-test")
    monkeypatch.setattr(bm25_retriever, "build_and_save_bm25_index", spy_build)

    bm25_retriever.get_bm25_index(chunk_dir, index_dir)

    assert calls["count"] == 1


def test_corrupt_metadata_rebuilds_without_crashing(tmp_path, monkeypatch):
    chunk_dir = tmp_path / "chunks"
    index_dir = tmp_path / "bm25_index"
    _write_chunks(chunk_dir)
    bm25_retriever.get_bm25_index(chunk_dir, index_dir)
    (index_dir / bm25_retriever.BM25_METADATA_FILE).write_text("{", encoding="utf-8")
    calls = {"count": 0}
    original_build = bm25_retriever.build_and_save_bm25_index

    def spy_build(*args, **kwargs):
        calls["count"] += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(bm25_retriever, "build_and_save_bm25_index", spy_build)

    rebuilt = bm25_retriever.get_bm25_index(chunk_dir, index_dir)

    assert calls["count"] == 1
    assert len(rebuilt.records) == 3


def test_corrupt_index_rebuilds_without_crashing(tmp_path, monkeypatch):
    chunk_dir = tmp_path / "chunks"
    index_dir = tmp_path / "bm25_index"
    _write_chunks(chunk_dir)
    bm25_retriever.get_bm25_index(chunk_dir, index_dir)
    (index_dir / "data.csc.index.npy").write_text("not a numpy file", encoding="utf-8")
    calls = {"count": 0}
    original_build = bm25_retriever.build_and_save_bm25_index

    def spy_build(*args, **kwargs):
        calls["count"] += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(bm25_retriever, "build_and_save_bm25_index", spy_build)

    rebuilt = bm25_retriever.get_bm25_index(chunk_dir, index_dir)

    assert calls["count"] == 1
    assert len(rebuilt.records) == 3


def test_empty_chunk_dir_returns_empty_bundle_without_artifact(tmp_path):
    chunk_dir = tmp_path / "chunks"
    index_dir = tmp_path / "bm25_index"
    chunk_dir.mkdir()

    bundle = bm25_retriever.get_bm25_index(chunk_dir, index_dir)

    assert bundle.records == []
    assert bm25_retriever.bm25_retrieve_from_index("anything", bundle, top_k=3) == []
    assert not index_dir.exists()


def test_hybrid_retrieve_with_persisted_bm25_matches_legacy_path(tmp_path, monkeypatch):
    chunk_dir = tmp_path / "chunks"
    index_dir = tmp_path / "bm25_index"
    records = _chunk_records()
    _write_chunks(chunk_dir, records)
    bm25_index = bm25_retriever.get_bm25_index(chunk_dir, index_dir)
    dense_results = [
        {
            "rank": 1,
            "chunk_id": "chunk_knn",
            "chunk_text": "K nearest neighbors predicts using nearby observations.",
            "metadata": {"document_id": "doc_stats", "section": "KNN"},
            "similarity": 0.9,
        }
    ]

    monkeypatch.setattr(
        hybrid_retrieval,
        "retrieve_relevant_chunks",
        lambda **kwargs: dense_results,
    )

    legacy = hybrid_retrieval.hybrid_retrieve(
        query="least squares regression",
        collection=object(),
        chunk_records=records,
        embedding_model="fake",
        dense_k=1,
        bm25_k=3,
        top_k=3,
        alpha=0.6,
    )
    persisted = hybrid_retrieval.hybrid_retrieve(
        query="least squares regression",
        collection=object(),
        chunk_records=bm25_index.records,
        embedding_model="fake",
        dense_k=1,
        bm25_k=3,
        top_k=3,
        alpha=0.6,
        bm25_index=bm25_index,
    )

    assert persisted == legacy
