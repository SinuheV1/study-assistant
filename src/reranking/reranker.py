from sentence_transformers import CrossEncoder

from src.utils.logging import setup_logger

log = setup_logger(__name__)
_model_cache: dict = {}


def load_reranker_model(model_name: str):
    if model_name not in _model_cache:
        _model_cache[model_name] = CrossEncoder(model_name)
        log.info(f"CrossEncoder model successfully loaded: {model_name}")

    return _model_cache[model_name]


def score_query_chunk_pairs(
    query: str, retrieved_results: list[dict], reranker_model, batch_size: int = 16
):
    pairs = []

    for result in retrieved_results:
        chunk_text = result.get("chunk_text", "")

        if not chunk_text or not chunk_text.strip():
            log.warning(
                f"Skipping empty chunk during reranking: {result.get('chunk_id', 'unknown')}"
            )
            pairs.append([query, ""])
            continue

        pairs.append([query, chunk_text])

    scores = reranker_model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

    return scores


def rerank_results(
    query: str, retrieved_results: list[dict], model_name: str, top_k: int, batch_size: int = 16
) -> list[dict]:
    """
    Takes the chunks returned by dense/vector retrieval, scores each chunk again
    with a cross-encoder reranker, sorts by reranker score, then returns the best chunks.
    """

    if not query or not query.strip():
        log.warning("Query is empty. Returning original retrieved results.")
        return retrieved_results[:top_k]

    if not retrieved_results:
        log.info("Retrieved results empty. Returning empty list.")
        return []

    model = load_reranker_model(model_name=model_name)

    scores = score_query_chunk_pairs(
        query=query,
        retrieved_results=retrieved_results,
        reranker_model=model,
        batch_size=batch_size,
    )

    reranked_results = []

    for result, score in zip(retrieved_results, scores):
        result = result.copy()

        # Store original dense/vector or hybrid rank info
        result["dense_rank"] = result.get("rank")
        result["dense_similarity"] = result.get("similarity")

        # Store reranker score
        result["rerank_score"] = float(score)

        reranked_results.append(result)

    reranked_results = sorted(
        reranked_results, key=lambda x: x.get("rerank_score", float("-inf")), reverse=True
    )

    for index, result in enumerate(reranked_results, start=1):
        result["rank"] = index

    log.info(f"Reranked {len(retrieved_results)} chunks and returned top {top_k}.")

    return reranked_results[:top_k]
