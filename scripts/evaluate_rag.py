import argparse
import json
import re
import subprocess
import time
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
    "dense_reranker": "Dense + Reranker",
    "hybrid": "Hybrid",
    "hybrid_reranker": "Hybrid + Reranker",
}
PIPELINE_ORDER = ["dense", "dense_reranker", "hybrid", "hybrid_reranker"]
PIPELINE_ALIASES = {
    "dense": "dense",
    "dense_reranker": "dense_reranker",
    "reranked": "dense_reranker",
    "hybrid": "hybrid",
    "hybrid_reranker": "hybrid_reranker",
    "hybrid_reranked": "hybrid_reranker",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run RAG evaluation.")
    parser.add_argument(
        "--mode",
        choices=("full", "retrieval"),
        default="full",
        help="Evaluation mode: full runs generation, retrieval skips generation.",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Limit the number of queries evaluated after group filtering.",
    )
    parser.add_argument(
        "--group",
        choices=("semantic", "lexical"),
        default=None,
        help="Evaluate only queries in this group.",
    )
    parser.add_argument(
        "--pipelines",
        type=parse_pipeline_keys,
        default=None,
        help="Comma-separated pipelines: dense,dense_reranker,hybrid,hybrid_reranker.",
    )
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


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("--limit must be greater than zero")
    return parsed


def parse_pipeline_keys(value: str) -> list[str]:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    if not requested:
        raise argparse.ArgumentTypeError("--pipelines must include at least one pipeline")

    parsed = []
    invalid = []
    for key in requested:
        canonical_key = PIPELINE_ALIASES.get(key)
        if canonical_key is None:
            invalid.append(key)
        elif canonical_key not in parsed:
            parsed.append(canonical_key)

    if invalid:
        valid = ", ".join(PIPELINE_ORDER)
        raise argparse.ArgumentTypeError(
            f"Invalid pipeline key(s): {', '.join(invalid)}. Valid keys: {valid}"
        )

    return [key for key in PIPELINE_ORDER if key in parsed]


def normalize_pipeline_keys(pipelines: str | list[str] | None) -> list[str]:
    if pipelines is None:
        return list(PIPELINE_ORDER)
    if isinstance(pipelines, str):
        return parse_pipeline_keys(pipelines)

    parsed = parse_pipeline_keys(",".join(pipelines))
    return parsed


def filter_eval_queries(queries: list[dict], group: str | None, limit: int | None) -> list[dict]:
    if group is not None:
        queries = [query for query in queries if get_query_group(query["id"]) == group]
    if limit is not None:
        queries = queries[:limit]

    return queries


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


def optional_average(scores: list[float]) -> float | None:
    if not scores:
        return None

    return safe_average(scores)


def format_score(value: float | None) -> str:
    if value is None:
        return "N/A"

    return f"{value:.2f}"


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
        "scores": {
            key: {
                "retrieval": [],
                "generation": [],
            }
            for key in PIPELINE_ORDER
        },
        "top_result_changes": {
            "dense_reranker": 0,
            "hybrid": 0,
            "hybrid_reranker": 0,
        },
        "query_count": 0,
    }


def summarize_score_store(store: dict, pipeline_keys: list[str] | None = None) -> dict:
    pipeline_keys = pipeline_keys or list(PIPELINE_ORDER)
    pipeline_summaries = {}

    for key in pipeline_keys:
        scores = store["scores"][key]
        pipeline_summaries[PIPELINE_LABELS[key]] = {
            "retrieval": optional_average(scores["retrieval"]),
            "generation": optional_average(scores["generation"]),
        }

    dense_summary = pipeline_summaries.get(PIPELINE_LABELS["dense"])
    dense_retrieval = dense_summary["retrieval"] if dense_summary else None
    dense_generation = dense_summary["generation"] if dense_summary else None

    deltas = {}
    if dense_retrieval is not None:
        for key in pipeline_keys:
            if key == "dense":
                continue

            summary = pipeline_summaries[PIPELINE_LABELS[key]]
            retrieval = summary["retrieval"]
            generation = summary["generation"]
            deltas[PIPELINE_LABELS[key]] = {
                "retrieval": retrieval - dense_retrieval if retrieval is not None else None,
                "generation": (
                    generation - dense_generation
                    if generation is not None and dense_generation is not None
                    else None
                ),
            }

    return {
        "query_count": store["query_count"],
        "pipelines": pipeline_summaries,
        "deltas_vs_dense": deltas,
        "top_result_changes_vs_dense": {
            PIPELINE_LABELS[key]: store["top_result_changes"][key]
            for key in pipeline_keys
            if key in store["top_result_changes"]
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
    pipeline_results: dict[str, dict],
    top_changes: dict[str, bool],
):
    for pipeline_key, values in pipeline_results.items():
        store["scores"][pipeline_key]["retrieval"].append(values["retrieval_score"])
        if values["generation_score"] is not None:
            store["scores"][pipeline_key]["generation"].append(values["generation_score"])

    for pipeline_key, changed in top_changes.items():
        if changed and pipeline_key in store["top_result_changes"]:
            store["top_result_changes"][pipeline_key] += 1

    store["query_count"] += 1


def print_summary(label, store, pipeline_keys: list[str]):
    summary = summarize_score_store(store, pipeline_keys)
    deltas = summary["deltas_vs_dense"]
    top_changes = summary["top_result_changes_vs_dense"]
    query_count = store["query_count"]

    print(f"\n=== {label.upper()} RESULTS ===")

    print("\nPipeline                    Retrieval   Generation")
    print("-------------------------------------------------")
    for pipeline_key in pipeline_keys:
        pipeline_label = PIPELINE_LABELS[pipeline_key]
        scores = summary["pipelines"][pipeline_label]
        print(
            f"{pipeline_label:<28} {format_score(scores['retrieval']):<10} "
            f"{format_score(scores['generation'])}"
        )

    if deltas:
        print("\nDeltas vs Dense Baseline")
        print("-------------------------------------------------")
        for pipeline_label, values in deltas.items():
            retrieval_delta = values["retrieval"]
            generation_delta = values["generation"]
            retrieval_text = f"{retrieval_delta:+.2f}" if retrieval_delta is not None else "N/A"
            generation_text = f"{generation_delta:+.2f}" if generation_delta is not None else "N/A"
            print(f"{pipeline_label} Retrieval Delta:      {retrieval_text}")
            print(f"{pipeline_label} Generation Delta:     {generation_text}")

    if top_changes:
        print("\nTop Result Changes vs Dense")
        print("-------------------------------------------------")
        for pipeline_label, changed_count in top_changes.items():
            print(f"{pipeline_label} Changed Top:  {changed_count}/{query_count}")


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

    for pipeline_key, values in pipeline_results.items():
        results = values["results"]
        records.append(
            {
                "query_id": query_id,
                "query": query,
                "group": group,
                "pipeline": PIPELINE_LABELS[pipeline_key],
                "retrieval_score": values["retrieval_score"],
                "generation_score": values["generation_score"],
                "retrieval_seconds": values["retrieval_seconds"],
                "generation_seconds": values["generation_seconds"],
                "total_seconds": values["total_seconds"],
                "top_chunk_ids": [
                    result.get("chunk_id") for result in results[:3] if result.get("chunk_id")
                ],
                "top_sources": extract_top_sources(results),
            }
        )

    return records


def initialize_timing_store(pipeline_keys: list[str]) -> dict:
    return {
        PIPELINE_LABELS[key]: {
            "query_count": 0,
            "retrieval_seconds": 0.0,
            "generation_seconds": 0.0,
            "total_seconds": 0.0,
        }
        for key in pipeline_keys
    }


def record_timing(timing_store: dict, pipeline_key: str, values: dict) -> None:
    pipeline_timing = timing_store[PIPELINE_LABELS[pipeline_key]]
    pipeline_timing["query_count"] += 1
    pipeline_timing["retrieval_seconds"] += values["retrieval_seconds"]
    pipeline_timing["generation_seconds"] += values["generation_seconds"]
    pipeline_timing["total_seconds"] += values["total_seconds"]


def evaluate_pipeline(
    *,
    pipeline_key: str,
    mode: str,
    query: str,
    keywords: list[str],
    collection,
    chunk_records: list[dict],
    bm25_index,
) -> dict:
    retrieval_started = time.perf_counter()

    if pipeline_key == "dense":
        results = retrieve_relevant_chunks(
            query=query,
            collection=collection,
            model_name=embed_model,
            top_k=top_k,
        )
    elif pipeline_key == "dense_reranker":
        candidate_results = retrieve_relevant_chunks(
            query=query,
            collection=collection,
            model_name=embed_model,
            top_k=candidate_k,
        )
        results = rerank_results(
            query=query,
            retrieved_results=candidate_results,
            model_name=reranker_model,
            top_k=top_k,
        )
    elif pipeline_key == "hybrid":
        results = hybrid_retrieve(
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
    elif pipeline_key == "hybrid_reranker":
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
        results = rerank_results(
            query=query,
            retrieved_results=hybrid_candidates,
            model_name=reranker_model,
            top_k=top_k,
        )
    else:
        raise ValueError(f"Unsupported pipeline: {pipeline_key}")

    retrieval_seconds = time.perf_counter() - retrieval_started
    retrieval_score = score_results(results, keywords)
    generation_seconds = 0.0
    generation_score = None

    if mode == "full":
        generation_started = time.perf_counter()
        answer = generate_answer(query, results, llm_model)
        generation_seconds = time.perf_counter() - generation_started
        generation_score = keyword_score(answer or "", keywords)

    return {
        "results": results,
        "retrieval_score": retrieval_score,
        "generation_score": generation_score,
        "retrieval_seconds": retrieval_seconds,
        "generation_seconds": generation_seconds,
        "total_seconds": retrieval_seconds + generation_seconds,
    }


def get_top_chunk_id(values: dict | None) -> str | None:
    if not values:
        return None

    results = values["results"]
    return results[0].get("chunk_id") if results else None


def get_top_changes(pipeline_results: dict[str, dict]) -> dict[str, bool]:
    dense_top_chunk = get_top_chunk_id(pipeline_results.get("dense"))
    if dense_top_chunk is None:
        return {}

    return {
        pipeline_key: dense_top_chunk != get_top_chunk_id(pipeline_results.get(pipeline_key))
        for pipeline_key in ("dense_reranker", "hybrid", "hybrid_reranker")
        if pipeline_key in pipeline_results
    }


def print_query_results(
    *,
    query_id: str,
    query_group: str,
    query: str,
    pipeline_results: dict[str, dict],
    top_changes: dict[str, bool],
) -> None:
    print(f"\nQuery ID: {query_id}")
    print(f"Group: {query_group}")
    print(f"Query: {query}")

    for pipeline_key, values in pipeline_results.items():
        pipeline_label = PIPELINE_LABELS[pipeline_key]
        print(f"{pipeline_label} Retrieval Score: {values['retrieval_score']:.2f}")
        print(f"{pipeline_label} Generation Score: {format_score(values['generation_score'])}")

    for pipeline_key, changed in top_changes.items():
        print(f"{PIPELINE_LABELS[pipeline_key]} Top Changed: {changed}")


def print_run_metadata(
    *,
    mode: str,
    query_count: int,
    pipeline_keys: list[str],
    total_seconds: float,
) -> None:
    print("\n=== EVALUATION RUN ===")
    print(f"Eval mode: {mode}")
    print(f"Queries evaluated: {query_count}")
    print(f"Pipelines: {', '.join(PIPELINE_LABELS[key] for key in pipeline_keys)}")
    print(f"Generation: {'enabled' if mode == 'full' else 'skipped'}")
    print(f"Total time: {total_seconds:.2f} seconds")


def build_eval_artifact(
    *,
    all_scores: dict,
    semantic_scores: dict,
    lexical_scores: dict,
    query_records: list[dict],
    results_dir: Path,
    pipeline_keys: list[str],
    run_options: dict,
    timing: dict,
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
        "run_options": run_options,
        "timing": timing,
        "summary": {
            "all": summarize_score_store(all_scores, pipeline_keys),
            "groups": {
                "semantic": summarize_score_store(semantic_scores, pipeline_keys),
                "lexical": summarize_score_store(lexical_scores, pipeline_keys),
            },
        },
        "pipelines": [PIPELINE_LABELS[key] for key in pipeline_keys],
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
    mode: str = "full",
    limit: int | None = None,
    group: str | None = None,
    pipelines: str | list[str] | None = None,
):
    if mode not in {"full", "retrieval"}:
        raise ValueError("mode must be 'full' or 'retrieval'")
    if group not in {None, "semantic", "lexical"}:
        raise ValueError("group must be 'semantic', 'lexical', or None")
    if limit is not None and limit < 1:
        raise ValueError("limit must be greater than zero")

    pipeline_keys = normalize_pipeline_keys(pipelines)

    with open(evaluation_queries_path) as f:
        data = json.load(f)

    queries = filter_eval_queries(data["queries"], group, limit)
    run_started_at = datetime.now()
    run_started_perf = time.perf_counter()

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
    pipeline_timing = initialize_timing_store(pipeline_keys)

    for q in queries:
        query_id = q["id"]
        query = q["query"]
        keywords = q["expected_keywords"]
        query_group = get_query_group(query_id)
        pipeline_results = {}

        for pipeline_key in pipeline_keys:
            pipeline_results[pipeline_key] = evaluate_pipeline(
                pipeline_key=pipeline_key,
                mode=mode,
                query=query,
                keywords=keywords,
                collection=collection,
                chunk_records=chunk_records,
                bm25_index=bm25_index,
            )
            record_timing(pipeline_timing, pipeline_key, pipeline_results[pipeline_key])

        top_changes = get_top_changes(pipeline_results)
        update_store(all_scores, pipeline_results, top_changes)

        if query_group == "lexical":
            group_store = lexical_scores
        else:
            group_store = semantic_scores

        update_store(group_store, pipeline_results, top_changes)

        query_records.extend(
            build_query_artifact_records(
                query_id=query_id,
                query=query,
                group=query_group,
                pipeline_results=pipeline_results,
            )
        )

        print_query_results(
            query_id=query_id,
            query_group=query_group,
            query=query,
            pipeline_results=pipeline_results,
            top_changes=top_changes,
        )

        if debug:
            for pipeline_key, values in pipeline_results.items():
                print_debug_sources(PIPELINE_LABELS[pipeline_key].upper(), values["results"])

    run_finished_at = datetime.now()
    total_seconds = time.perf_counter() - run_started_perf
    timing = {
        "started_at": run_started_at.isoformat(timespec="seconds"),
        "finished_at": run_finished_at.isoformat(timespec="seconds"),
        "total_seconds": total_seconds,
        "pipelines": pipeline_timing,
    }
    run_options = {
        "mode": mode,
        "limit": limit,
        "group": group,
        "pipelines": pipeline_keys,
    }

    print_run_metadata(
        mode=mode,
        query_count=len(queries),
        pipeline_keys=pipeline_keys,
        total_seconds=total_seconds,
    )
    print("\n=== FINAL RESULTS ===")
    print_summary("All Queries", all_scores, pipeline_keys)
    print_summary("Semantic Queries", semantic_scores, pipeline_keys)
    print_summary("Lexical / Hybrid Queries", lexical_scores, pipeline_keys)

    output_dir = Path(results_dir) if results_dir is not None else eval_results_directory
    artifact = build_eval_artifact(
        all_scores=all_scores,
        semantic_scores=semantic_scores,
        lexical_scores=lexical_scores,
        query_records=query_records,
        results_dir=output_dir,
        pipeline_keys=pipeline_keys,
        run_options=run_options,
        timing=timing,
        run_name=run_name,
        created_at=run_started_at,
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
        mode=args.mode,
        limit=args.limit,
        group=args.group,
        pipelines=args.pipelines,
    )
