import json

from src.generation.generator import generate_answer
from src.reranking.reranker import rerank_results
from src.retrieval.bm25_retriever import load_chunk_records
from src.retrieval.hybrid_retrieval import hybrid_retrieve
from src.retrieval.retriever import retrieve_relevant_chunks
from src.utils.config import load_config
from src.vector_store.vectordb import get_or_create_collection, initialize_vector_db

config = load_config()

persist_directory = config["paths"]["persist_dir"]
collection_name = config["vector_store"]["collection_name"]
chunk_directory = config["paths"]["chunk_dir"]
evaluation_queries_path = config["paths"]["evaluation_queries"]

embed_model = config["models"]["embedding"]
llm_model = config["models"]["llm"]
reranker_model = config["models"]["reranker"]

top_k = config["evaluation"]["top_k"]
candidate_k = config["evaluation"]["candidate_k"]

dense_k = config["retrieval"]["dense_k"]
bm25_k = config["retrieval"]["bm25_k"]
hybrid_alpha = config["retrieval"]["hybrid_alpha"]

debug = False


def get_query_group(query_id: str) -> str:
    if query_id.startswith("lexical_"):
        return "lexical"
    return "semantic"


def keyword_score(text, keywords):
    if not keywords:
        return 0.0

    text = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text)

    return hits / len(keywords)


def score_results(results, keywords):
    combined_chunks = " ".join(r.get("chunk_text", "") for r in results)
    return keyword_score(combined_chunks, keywords)


def safe_average(scores):
    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def init_score_store():
    return {
        "dense_retrieval": [],
        "dense_generation": [],
        "reranked_retrieval": [],
        "reranked_generation": [],
        "hybrid_retrieval": [],
        "hybrid_generation": [],
        "hybrid_reranked_retrieval": [],
        "hybrid_reranked_generation": [],
        "dense_reranker_changed_top": 0,
        "hybrid_changed_top": 0,
        "hybrid_reranked_changed_top": 0,
        "query_count": 0,
    }


def print_debug_sources(label, results):
    print(f"\n=== DEBUG {label} SOURCES ===")

    for r in results:
        metadata = r.get("metadata", {})

        print(f"Rank: {r.get('rank')}")
        print(f"Dense Rank: {r.get('dense_rank')}")
        print(f"Dense Similarity: {r.get('dense_similarity')}")
        print(f"BM25 Rank: {r.get('bm25_rank')}")
        print(f"BM25 Score: {r.get('bm25_score')}")
        print(f"Hybrid Score: {r.get('hybrid_score')}")
        print(f"Rerank Score: {r.get('rerank_score')}")
        print(f"Document ID: {metadata.get('document_id')}")
        print(f"File: {metadata.get('file_name')}")
        print(f"Course: {metadata.get('course')}")
        print(f"Section: {metadata.get('section')}")
        print(f"Source Type: {metadata.get('source_type')}")
        print(f"Preview: {r.get('chunk_text', '')[:120]}")
        print("----")


def update_store(
    store,
    dense_r_score,
    dense_g_score,
    reranked_r_score,
    reranked_g_score,
    hybrid_r_score,
    hybrid_g_score,
    hybrid_reranked_r_score,
    hybrid_reranked_g_score,
    dense_top_changed,
    hybrid_top_changed,
    hybrid_reranked_top_changed,
):
    store["dense_retrieval"].append(dense_r_score)
    store["dense_generation"].append(dense_g_score)

    store["reranked_retrieval"].append(reranked_r_score)
    store["reranked_generation"].append(reranked_g_score)

    store["hybrid_retrieval"].append(hybrid_r_score)
    store["hybrid_generation"].append(hybrid_g_score)

    store["hybrid_reranked_retrieval"].append(hybrid_reranked_r_score)
    store["hybrid_reranked_generation"].append(hybrid_reranked_g_score)

    if dense_top_changed:
        store["dense_reranker_changed_top"] += 1

    if hybrid_top_changed:
        store["hybrid_changed_top"] += 1

    if hybrid_reranked_top_changed:
        store["hybrid_reranked_changed_top"] += 1

    store["query_count"] += 1


def print_summary(label, store):
    dense_avg_retrieval = safe_average(store["dense_retrieval"])
    dense_avg_generation = safe_average(store["dense_generation"])

    reranked_avg_retrieval = safe_average(store["reranked_retrieval"])
    reranked_avg_generation = safe_average(store["reranked_generation"])

    hybrid_avg_retrieval = safe_average(store["hybrid_retrieval"])
    hybrid_avg_generation = safe_average(store["hybrid_generation"])

    hybrid_reranked_avg_retrieval = safe_average(store["hybrid_reranked_retrieval"])
    hybrid_reranked_avg_generation = safe_average(store["hybrid_reranked_generation"])

    query_count = store["query_count"]

    print(f"\n=== {label.upper()} RESULTS ===")

    print("\nPipeline                    Retrieval   Generation")
    print("-------------------------------------------------")
    print(
        f"Dense Baseline              {dense_avg_retrieval:.2f}        {dense_avg_generation:.2f}"
    )
    print(
        f"Dense + Reranker            {reranked_avg_retrieval:.2f}        {reranked_avg_generation:.2f}"
    )
    print(
        f"Hybrid                      {hybrid_avg_retrieval:.2f}        {hybrid_avg_generation:.2f}"
    )
    print(
        f"Hybrid + Reranker           {hybrid_reranked_avg_retrieval:.2f}        {hybrid_reranked_avg_generation:.2f}"
    )

    print("\nDeltas vs Dense Baseline")
    print("-------------------------------------------------")
    print(
        f"Dense + Reranker Retrieval Delta:      {reranked_avg_retrieval - dense_avg_retrieval:+.2f}"
    )
    print(
        f"Dense + Reranker Generation Delta:     {reranked_avg_generation - dense_avg_generation:+.2f}"
    )
    print(
        f"Hybrid Retrieval Delta:                {hybrid_avg_retrieval - dense_avg_retrieval:+.2f}"
    )
    print(
        f"Hybrid Generation Delta:               {hybrid_avg_generation - dense_avg_generation:+.2f}"
    )
    print(
        f"Hybrid + Reranker Retrieval Delta:     {hybrid_reranked_avg_retrieval - dense_avg_retrieval:+.2f}"
    )
    print(
        f"Hybrid + Reranker Generation Delta:    {hybrid_reranked_avg_generation - dense_avg_generation:+.2f}"
    )

    print("\nTop Result Changes vs Dense")
    print("-------------------------------------------------")
    print(f"Dense + Reranker Changed Top:   {store['dense_reranker_changed_top']}/{query_count}")
    print(f"Hybrid Changed Top:             {store['hybrid_changed_top']}/{query_count}")
    print(f"Hybrid + Reranker Changed Top:  {store['hybrid_reranked_changed_top']}/{query_count}")


def run_evaluation():
    with open(evaluation_queries_path) as f:
        data = json.load(f)

    queries = data["queries"]

    client = initialize_vector_db(str(persist_directory))
    collection = get_or_create_collection(client, collection_name)
    chunk_records = load_chunk_records(str(chunk_directory))

    all_scores = init_score_store()
    semantic_scores = init_score_store()
    lexical_scores = init_score_store()

    for q in queries:
        query_id = q["id"]
        query = q["query"]
        keywords = q["expected_keywords"]
        query_group = get_query_group(query_id)

        dense_results = retrieve_relevant_chunks(
            query=query,
            collection=collection,
            model_name=embed_model,
            top_k=top_k,
        )

        dense_r_score = score_results(dense_results, keywords)
        dense_answer = generate_answer(query, dense_results, llm_model)
        dense_g_score = keyword_score(dense_answer or "", keywords)

        candidate_results = retrieve_relevant_chunks(
            query=query,
            collection=collection,
            model_name=embed_model,
            top_k=candidate_k,
        )

        reranked_results = rerank_results(
            query=query,
            retrieved_results=candidate_results,
            model_name=reranker_model,
            top_k=top_k,
        )

        reranked_r_score = score_results(reranked_results, keywords)
        reranked_answer = generate_answer(query, reranked_results, llm_model)
        reranked_g_score = keyword_score(reranked_answer or "", keywords)

        hybrid_results = hybrid_retrieve(
            query=query,
            collection=collection,
            chunk_records=chunk_records,
            embedding_model=embed_model,
            dense_k=dense_k,
            bm25_k=bm25_k,
            top_k=top_k,
            alpha=hybrid_alpha,
        )

        hybrid_r_score = score_results(hybrid_results, keywords)
        hybrid_answer = generate_answer(query, hybrid_results, llm_model)
        hybrid_g_score = keyword_score(hybrid_answer or "", keywords)

        hybrid_candidates = hybrid_retrieve(
            query=query,
            collection=collection,
            chunk_records=chunk_records,
            embedding_model=embed_model,
            dense_k=dense_k,
            bm25_k=bm25_k,
            top_k=candidate_k,
            alpha=hybrid_alpha,
        )

        hybrid_reranked_results = rerank_results(
            query=query,
            retrieved_results=hybrid_candidates,
            model_name=reranker_model,
            top_k=top_k,
        )

        hybrid_reranked_r_score = score_results(hybrid_reranked_results, keywords)
        hybrid_reranked_answer = generate_answer(query, hybrid_reranked_results, llm_model)
        hybrid_reranked_g_score = keyword_score(hybrid_reranked_answer or "", keywords)

        dense_top_chunk = dense_results[0].get("chunk_id") if dense_results else None
        reranked_top_chunk = reranked_results[0].get("chunk_id") if reranked_results else None
        hybrid_top_chunk = hybrid_results[0].get("chunk_id") if hybrid_results else None
        hybrid_reranked_top_chunk = (
            hybrid_reranked_results[0].get("chunk_id") if hybrid_reranked_results else None
        )

        dense_top_changed = dense_top_chunk != reranked_top_chunk
        hybrid_top_changed = dense_top_chunk != hybrid_top_chunk
        hybrid_reranked_top_changed = dense_top_chunk != hybrid_reranked_top_chunk

        update_store(
            all_scores,
            dense_r_score,
            dense_g_score,
            reranked_r_score,
            reranked_g_score,
            hybrid_r_score,
            hybrid_g_score,
            hybrid_reranked_r_score,
            hybrid_reranked_g_score,
            dense_top_changed,
            hybrid_top_changed,
            hybrid_reranked_top_changed,
        )

        if query_group == "lexical":
            group_store = lexical_scores
        else:
            group_store = semantic_scores

        update_store(
            group_store,
            dense_r_score,
            dense_g_score,
            reranked_r_score,
            reranked_g_score,
            hybrid_r_score,
            hybrid_g_score,
            hybrid_reranked_r_score,
            hybrid_reranked_g_score,
            dense_top_changed,
            hybrid_top_changed,
            hybrid_reranked_top_changed,
        )

        print(f"\nQuery ID: {query_id}")
        print(f"Group: {query_group}")
        print(f"Query: {query}")

        print(f"Dense Retrieval Score: {dense_r_score:.2f}")
        print(f"Dense Generation Score: {dense_g_score:.2f}")

        print(f"Reranked Retrieval Score: {reranked_r_score:.2f}")
        print(f"Reranked Generation Score: {reranked_g_score:.2f}")

        print(f"Hybrid Retrieval Score: {hybrid_r_score:.2f}")
        print(f"Hybrid Generation Score: {hybrid_g_score:.2f}")

        print(f"Hybrid + Reranker Retrieval Score: {hybrid_reranked_r_score:.2f}")
        print(f"Hybrid + Reranker Generation Score: {hybrid_reranked_g_score:.2f}")

        print(f"Dense + Reranker Top Changed: {dense_top_changed}")
        print(f"Hybrid Top Changed: {hybrid_top_changed}")
        print(f"Hybrid + Reranker Top Changed: {hybrid_reranked_top_changed}")

        if debug:
            print_debug_sources("DENSE", dense_results)
            print_debug_sources("DENSE + RERANKED", reranked_results)
            print_debug_sources("HYBRID", hybrid_results)
            print_debug_sources("HYBRID + RERANKED", hybrid_reranked_results)

    print("\n=== FINAL RESULTS ===")
    print_summary("All Queries", all_scores)
    print_summary("Semantic Queries", semantic_scores)
    print_summary("Lexical / Hybrid Queries", lexical_scores)


if __name__ == "__main__":
    run_evaluation()
