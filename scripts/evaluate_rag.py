import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

from src.generation.generator import generate_answer
from src.reranking.reranker import rerank_results
from src.retrieval.bm25_retriever import get_bm25_index, load_chunk_records
from src.retrieval.hybrid_retrieval import hybrid_retrieve
from src.retrieval.retriever import retrieve_relevant_chunks
from src.utils.config import PROJECT_ROOT, load_config
from src.vector_store.vectordb import get_or_create_collection, initialize_vector_db

config = load_config()

persist_directory = config["paths"]["persist_dir"]
collection_name = config["vector_store"]["collection_name"]
chunk_directory = config["paths"]["chunk_dir"]
bm25_index_directory = config["paths"]["bm25_index_dir"]
evaluation_queries_path = config["paths"]["evaluation_queries"]
eval_results_directory = config["paths"]["eval_results_dir"]

embed_model = config["models"]["embedding"]
llm_model = config["models"]["llm"]
reranker_model = config["models"]["reranker"]

top_k = config["evaluation"]["top_k"]
candidate_k = config["evaluation"]["candidate_k"]

dense_k = config["retrieval"]["dense_k"]
bm25_k = config["retrieval"]["bm25_k"]
hybrid_alpha = config["retrieval"]["hybrid_alpha"]

debug = False

PIPELINE_LABELS = {
    "dense": "Dense Baseline",
    "reranked": "Dense + Reranker",
    "hybrid": "Hybrid",
    "hybrid_reranked": "Hybrid + Reranker",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run RAG evaluation.")
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument(
        "--save-results",
        dest="save_results",
        action="store_true",
        help="Save a timestamped evaluation artifact. Enabled by default.",
    )
    save_group.add_argument(
        "--no-save-results",
        dest="save_results",
        action="store_false",
        help="Do not save an evaluation artifact.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory where evaluation artifacts are written.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional short name appended to the artifact filename.",
    )
    parser.set_defaults(save_results=True)
    return parser.parse_args()


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


def safe_slug(value: str | None) -> str | None:
    if not value:
        return None

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    slug = slug.strip("-")
    return slug or None


def _run_git_command(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    return result.stdout.strip() or None


def _get_git_dirty_status() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    return bool(result.stdout.strip())


def get_git_metadata() -> dict:
    branch = _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
    commit = _run_git_command(["rev-parse", "HEAD"])

    return {
        "branch": branch or "unknown",
        "commit": commit or "unknown",
        "dirty": _get_git_dirty_status(),
    }


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


def summarize_score_store(store: dict) -> dict:
    dense_avg_retrieval = safe_average(store["dense_retrieval"])
    dense_avg_generation = safe_average(store["dense_generation"])

    reranked_avg_retrieval = safe_average(store["reranked_retrieval"])
    reranked_avg_generation = safe_average(store["reranked_generation"])

    hybrid_avg_retrieval = safe_average(store["hybrid_retrieval"])
    hybrid_avg_generation = safe_average(store["hybrid_generation"])

    hybrid_reranked_avg_retrieval = safe_average(store["hybrid_reranked_retrieval"])
    hybrid_reranked_avg_generation = safe_average(store["hybrid_reranked_generation"])

    return {
        "query_count": store["query_count"],
        "pipelines": {
            PIPELINE_LABELS["dense"]: {
                "retrieval": dense_avg_retrieval,
                "generation": dense_avg_generation,
            },
            PIPELINE_LABELS["reranked"]: {
                "retrieval": reranked_avg_retrieval,
                "generation": reranked_avg_generation,
            },
            PIPELINE_LABELS["hybrid"]: {
                "retrieval": hybrid_avg_retrieval,
                "generation": hybrid_avg_generation,
            },
            PIPELINE_LABELS["hybrid_reranked"]: {
                "retrieval": hybrid_reranked_avg_retrieval,
                "generation": hybrid_reranked_avg_generation,
            },
        },
        "deltas_vs_dense": {
            PIPELINE_LABELS["reranked"]: {
                "retrieval": reranked_avg_retrieval - dense_avg_retrieval,
                "generation": reranked_avg_generation - dense_avg_generation,
            },
            PIPELINE_LABELS["hybrid"]: {
                "retrieval": hybrid_avg_retrieval - dense_avg_retrieval,
                "generation": hybrid_avg_generation - dense_avg_generation,
            },
            PIPELINE_LABELS["hybrid_reranked"]: {
                "retrieval": hybrid_reranked_avg_retrieval - dense_avg_retrieval,
                "generation": hybrid_reranked_avg_generation - dense_avg_generation,
            },
        },
        "top_result_changes_vs_dense": {
            PIPELINE_LABELS["reranked"]: store["dense_reranker_changed_top"],
            PIPELINE_LABELS["hybrid"]: store["hybrid_changed_top"],
            PIPELINE_LABELS["hybrid_reranked"]: store["hybrid_reranked_changed_top"],
        },
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
    summary = summarize_score_store(store)
    dense = summary["pipelines"][PIPELINE_LABELS["dense"]]
    reranked = summary["pipelines"][PIPELINE_LABELS["reranked"]]
    hybrid = summary["pipelines"][PIPELINE_LABELS["hybrid"]]
    hybrid_reranked = summary["pipelines"][PIPELINE_LABELS["hybrid_reranked"]]
    deltas = summary["deltas_vs_dense"]
    top_changes = summary["top_result_changes_vs_dense"]
    query_count = store["query_count"]

    print(f"\n=== {label.upper()} RESULTS ===")

    print("\nPipeline                    Retrieval   Generation")
    print("-------------------------------------------------")
    print(f"Dense Baseline              {dense['retrieval']:.2f}        {dense['generation']:.2f}")
    print(
        f"Dense + Reranker            {reranked['retrieval']:.2f}        {reranked['generation']:.2f}"
    )
    print(
        f"Hybrid                      {hybrid['retrieval']:.2f}        {hybrid['generation']:.2f}"
    )
    print(
        f"Hybrid + Reranker           {hybrid_reranked['retrieval']:.2f}        {hybrid_reranked['generation']:.2f}"
    )

    print("\nDeltas vs Dense Baseline")
    print("-------------------------------------------------")
    print(
        f"Dense + Reranker Retrieval Delta:      {deltas[PIPELINE_LABELS['reranked']]['retrieval']:+.2f}"
    )
    print(
        f"Dense + Reranker Generation Delta:     {deltas[PIPELINE_LABELS['reranked']]['generation']:+.2f}"
    )
    print(
        f"Hybrid Retrieval Delta:                {deltas[PIPELINE_LABELS['hybrid']]['retrieval']:+.2f}"
    )
    print(
        f"Hybrid Generation Delta:               {deltas[PIPELINE_LABELS['hybrid']]['generation']:+.2f}"
    )
    print(
        f"Hybrid + Reranker Retrieval Delta:     {deltas[PIPELINE_LABELS['hybrid_reranked']]['retrieval']:+.2f}"
    )
    print(
        f"Hybrid + Reranker Generation Delta:    {deltas[PIPELINE_LABELS['hybrid_reranked']]['generation']:+.2f}"
    )

    print("\nTop Result Changes vs Dense")
    print("-------------------------------------------------")
    print(
        f"Dense + Reranker Changed Top:   {top_changes[PIPELINE_LABELS['reranked']]}/{query_count}"
    )
    print(f"Hybrid Changed Top:             {top_changes[PIPELINE_LABELS['hybrid']]}/{query_count}")
    print(
        f"Hybrid + Reranker Changed Top:  {top_changes[PIPELINE_LABELS['hybrid_reranked']]}/{query_count}"
    )


def extract_top_sources(results: list[dict], limit: int = 3) -> list[dict]:
    sources = []

    for result in results[:limit]:
        metadata = result.get("metadata", {}) or {}
        sources.append(
            {
                "chunk_id": result.get("chunk_id"),
                "file_name": metadata.get("file_name"),
                "page_start": metadata.get("page_start"),
                "page_end": metadata.get("page_end"),
                "section": metadata.get("section"),
                "source_type": metadata.get("source_type"),
            }
        )

    return sources


def build_query_artifact_records(
    query_id: str,
    query: str,
    group: str,
    pipeline_results: dict[str, dict],
) -> list[dict]:
    records = []

    for pipeline_name, values in pipeline_results.items():
        results = values["results"]
        records.append(
            {
                "query_id": query_id,
                "query": query,
                "group": group,
                "pipeline": pipeline_name,
                "retrieval_score": values["retrieval_score"],
                "generation_score": values["generation_score"],
                "top_chunk_ids": [
                    result.get("chunk_id") for result in results[:3] if result.get("chunk_id")
                ],
                "top_sources": extract_top_sources(results),
            }
        )

    return records


def build_eval_artifact(
    *,
    all_scores: dict,
    semantic_scores: dict,
    lexical_scores: dict,
    query_records: list[dict],
    results_dir: Path,
    run_name: str | None = None,
    created_at: datetime | None = None,
    git_metadata: dict | None = None,
) -> dict:
    created_at = created_at or datetime.now()
    run_slug = safe_slug(run_name)
    timestamp = created_at.strftime("%Y-%m-%d_%H%M%S")
    run_id = f"{timestamp}_{run_slug}" if run_slug else timestamp

    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": created_at.isoformat(timespec="seconds"),
        "git": git_metadata if git_metadata is not None else get_git_metadata(),
        "config": {
            "embedding_model": embed_model,
            "generation_model": llm_model,
            "reranker_model": reranker_model,
            "retrieval": {
                "top_k": top_k,
                "candidate_k": candidate_k,
                "dense_k": dense_k,
                "bm25_k": bm25_k,
                "hybrid_alpha": hybrid_alpha,
            },
            "paths": {
                "persist_dir": str(persist_directory),
                "collection_name": collection_name,
                "evaluation_queries": str(evaluation_queries_path),
                "eval_results_dir": str(results_dir),
            },
        },
        "summary": {
            "all": summarize_score_store(all_scores),
            "groups": {
                "semantic": summarize_score_store(semantic_scores),
                "lexical": summarize_score_store(lexical_scores),
            },
        },
        "pipelines": list(PIPELINE_LABELS.values()),
        "queries": query_records,
    }


def save_eval_artifact(artifact: dict, results_dir: str | Path) -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = artifact["run_id"]
    output_path = results_dir / f"{run_id}_eval_run.json"

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)

    return output_path


def run_evaluation(
    *,
    save_results: bool = True,
    results_dir: str | Path | None = None,
    run_name: str | None = None,
):
    with open(evaluation_queries_path) as f:
        data = json.load(f)

    queries = data["queries"]

    client = initialize_vector_db(str(persist_directory))
    collection = get_or_create_collection(client, collection_name)
    bm25_index = get_bm25_index(
        chunk_dir=chunk_directory,
        index_dir=bm25_index_directory,
    )
    chunk_records = bm25_index.records or load_chunk_records(str(chunk_directory))

    all_scores = init_score_store()
    semantic_scores = init_score_store()
    lexical_scores = init_score_store()
    query_records = []

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
            bm25_index=bm25_index,
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
            bm25_index=bm25_index,
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

        query_records.extend(
            build_query_artifact_records(
                query_id=query_id,
                query=query,
                group=query_group,
                pipeline_results={
                    PIPELINE_LABELS["dense"]: {
                        "results": dense_results,
                        "retrieval_score": dense_r_score,
                        "generation_score": dense_g_score,
                    },
                    PIPELINE_LABELS["reranked"]: {
                        "results": reranked_results,
                        "retrieval_score": reranked_r_score,
                        "generation_score": reranked_g_score,
                    },
                    PIPELINE_LABELS["hybrid"]: {
                        "results": hybrid_results,
                        "retrieval_score": hybrid_r_score,
                        "generation_score": hybrid_g_score,
                    },
                    PIPELINE_LABELS["hybrid_reranked"]: {
                        "results": hybrid_reranked_results,
                        "retrieval_score": hybrid_reranked_r_score,
                        "generation_score": hybrid_reranked_g_score,
                    },
                },
            )
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

    output_dir = Path(results_dir) if results_dir is not None else eval_results_directory
    artifact = build_eval_artifact(
        all_scores=all_scores,
        semantic_scores=semantic_scores,
        lexical_scores=lexical_scores,
        query_records=query_records,
        results_dir=output_dir,
        run_name=run_name,
    )

    if save_results:
        output_path = save_eval_artifact(artifact, output_dir)
        print(f"\nSaved evaluation artifact: {output_path}")

    return artifact


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        save_results=args.save_results,
        results_dir=args.results_dir,
        run_name=args.run_name,
    )
