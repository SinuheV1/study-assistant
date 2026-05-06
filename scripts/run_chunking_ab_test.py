# scripts/run_chunking_ab_test.py

import json
from pathlib import Path

from src.utils.logging import setup_logger
from src.utils.io import save_json
from src.ingestion.ingest_docling_document import ingest_docling_document
from src.chunking.chunker import chunk_document
from src.embedding.embedder import embed_chunks
from src.vector_store.vectordb import (
    initialize_vector_db,
    reset_collection,
    add_records_to_collection,
    get_collection_count)
from src.retrieval.retriever import retrieve_relevant_chunks
from src.generation.generator import generate_answer


log = setup_logger(__name__)

# -----------------------------
# Shared config
# -----------------------------
file_path = "data/raw/lecture_pdfs/Lecture_01.pdf"
eval_queries_path = "evaluation/queries.json"

collection_name = "study_assistant_chunks"
embedding_model = "all-MiniLM-L6-v2"
llm_model = "llama3.2:3b"
top_k = 3

run_generation = True

results_dir = Path("evaluation/results/chunking_ab_test")
artifacts_dir = Path("data/processed/ab_tests")
vector_store_base_dir = Path("data/processed/vector_store_ab")


chunk_configs = {
    "A_700_50": {
        "target_size": 700,
        "overlap_size": 50},
    "B_900_75": {
        "target_size": 900,
        "overlap_size": 75}}


def load_eval_queries(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "queries" in data:
        return data["queries"]

    if isinstance(data, list):
        return data

    raise ValueError("queries.json must be either a list or contain a top-level 'queries' key.")


def keyword_score(text, keywords):
    if not keywords:
        return 0.0

    text = (text or "").lower()
    hits = sum(1 for kw in keywords if kw.lower() in text)

    return hits / len(keywords)


def evaluate_config(collection, queries, config_name):
    per_query_results = []
    retrieval_scores = []
    generation_scores = []

    for q in queries:
        query = q["query"]
        expected_keywords = q.get("expected_keywords", [])

        retrieved_results = retrieve_relevant_chunks(
            query=query,
            collection=collection,
            model_name=embedding_model,
            top_k=top_k)

        combined_chunks = " ".join(
            result.get("chunk_text", "") for result in retrieved_results)

        retrieval_score = keyword_score(combined_chunks, expected_keywords)
        retrieval_scores.append(retrieval_score)

        answer = ""

        if run_generation:
            answer = generate_answer(
                query=query,
                retrieved_results=retrieved_results,
                model_name=llm_model)

        generation_score = keyword_score(answer or "", expected_keywords)
        generation_scores.append(generation_score)

        per_query_results.append(
            {
                "config_name": config_name,
                "query": query,
                "expected_keywords": expected_keywords,
                "retrieval_score": retrieval_score,
                "generation_score": generation_score,
                "retrieved_chunks": [{
                        "rank": result.get("rank"),
                        "chunk_id": result.get("chunk_id"),
                        "distance": result.get("distance"),
                        "similarity": result.get("similarity"),
                        "preview": result.get("chunk_text", "")[:300],
                        "metadata": result.get("metadata", {})}
                    for result in retrieved_results],
                "answer": answer})

        print("\n" + "=" * 80)
        print(f"Config: {config_name}")
        print(f"Query: {query}")
        print(f"Retrieval Score: {retrieval_score:.2f}")
        print(f"Generation Score: {generation_score:.2f}")

    avg_retrieval_score = sum(retrieval_scores) / len(retrieval_scores)
    avg_generation_score = sum(generation_scores) / len(generation_scores)

    return {
        "config_name": config_name,
        "avg_retrieval_score": avg_retrieval_score,
        "avg_generation_score": avg_generation_score,
        "per_query_results": per_query_results}


def run_chunking_ab_test():
    log.info("Starting chunking A/B test.")

    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    vector_store_base_dir.mkdir(parents=True, exist_ok=True)

    queries = load_eval_queries(eval_queries_path)

    document = ingest_docling_document(file_path)
    if document is None:
        log.warning("Document ingestion failed. Stopping A/B test.")
        return None

    metadata = document.get("metadata", {})
    document_id = metadata.get("document_id", "unknown_doc")

    summary_results = []

    for config_name, config in chunk_configs.items():
        print("\n" + "#" * 100)
        print(f"Running config: {config_name}")
        print("#" * 100)

        target_size = config["target_size"]
        overlap_size = config["overlap_size"]

        # -----------------------------
        # Chunk
        # -----------------------------
        chunks = chunk_document(
            cleaned_text=document["cleaned_text"],
            document_metadata=metadata,
            target_size=target_size,
            overlap_size=overlap_size)

        if not chunks:
            log.warning(f"No chunks generated for {config_name}. Skipping.")
            continue

        config_artifact_dir = artifacts_dir / config_name
        chunks_path = config_artifact_dir / "chunks" / f"{document_id}_chunks.json"

        save_json(chunks, chunks_path)

        # -----------------------------
        # Embed
        # -----------------------------
        embedded_chunks = embed_chunks(
            chunk_records=chunks,
            model_name=embedding_model)

        if not embedded_chunks:
            log.warning(f"No embeddings generated for {config_name}. Skipping.")
            continue

        embeddings_path = (
            config_artifact_dir / "embeddings" / f"{document_id}_embeddings.json")

        save_json(embedded_chunks, embeddings_path)

        # -----------------------------
        # Build separate vector index
        # -----------------------------
        persist_dir = vector_store_base_dir / config_name

        client = initialize_vector_db(str(persist_dir))
        collection = reset_collection(client, collection_name)

        add_records_to_collection(collection, embedded_chunks)
        vector_count = get_collection_count(collection)

        # -----------------------------
        # Evaluate
        # -----------------------------
        eval_result = evaluate_config(
            collection=collection,
            queries=queries,
            config_name=config_name)

        config_result = {
            "config_name": config_name,
            "target_size": target_size,
            "overlap_size": overlap_size,
            "chunk_count": len(chunks),
            "embedding_count": len(embedded_chunks),
            "vector_db_count": vector_count,
            "avg_retrieval_score": eval_result["avg_retrieval_score"],
            "avg_generation_score": eval_result["avg_generation_score"],
            "per_query_results": eval_result["per_query_results"]}

        save_json(
            config_result,
            results_dir / f"{config_name}_results.json")

        summary_results.append(
            {
                "config_name": config_name,
                "target_size": target_size,
                "overlap_size": overlap_size,
                "chunk_count": len(chunks),
                "embedding_count": len(embedded_chunks),
                "vector_db_count": vector_count,
                "avg_retrieval_score": eval_result["avg_retrieval_score"],
                "avg_generation_score": eval_result["avg_generation_score"]})

    # -----------------------------
    # Compare configs
    # -----------------------------
    if not summary_results:
        log.warning("No valid A/B test results generated.")
        return None

    winner_by_retrieval = max(
        summary_results,
        key=lambda x: x["avg_retrieval_score"])

    winner_by_generation = max(
        summary_results,
        key=lambda x: x["avg_generation_score"])

    summary = {
        "experiment_name": "chunking_ab_test",
        "file_path": file_path,
        "embedding_model": embedding_model,
        "llm_model": llm_model,
        "top_k": top_k,
        "run_generation": run_generation,
        "results": summary_results,
        "winner_by_retrieval": winner_by_retrieval,
        "winner_by_generation": winner_by_generation}

    save_json(summary, results_dir / "ab_summary.json")

    print("\n" + "=" * 100)
    print("A/B TEST SUMMARY")
    print("=" * 100)

    for result in summary_results:
        print(f"\nConfig: {result['config_name']}")
        print(f"Target size: {result['target_size']}")
        print(f"Overlap size: {result['overlap_size']}")
        print(f"Chunks: {result['chunk_count']}")
        print(f"Avg Retrieval Score: {result['avg_retrieval_score']:.2f}")
        print(f"Avg Generation Score: {result['avg_generation_score']:.2f}")

    print("\nWinner by retrieval:")
    print(winner_by_retrieval["config_name"])

    print("\nWinner by generation:")
    print(winner_by_generation["config_name"])

    log.info("Chunking A/B test completed.")

    return summary


if __name__ == "__main__":
    run_chunking_ab_test()