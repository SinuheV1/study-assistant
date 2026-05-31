from src.utils.logging import setup_logger
from src.retrieval.retriever import retrieve_relevant_chunks
from src.retrieval.bm25_retriever import bm25_retrieve

log = setup_logger(__name__)

def normalize_scores(results:list[dict],score_key:str,normalized_key:str)->list[dict]:
    if not results:
        return []
    
    results = [result.copy() for result in results]
    scores = [
        result.get(score_key)
        for result in results
        if result.get(score_key) is not None]

    if not scores:
        for result in results:
            result[normalized_key] = 0.0
        return results

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        for result in results:
            result[normalized_key] = 1.0
        return results

    for result in results:
        raw_score = result.get(score_key)

        if raw_score is None:
            result[normalized_key] = 0.0
        else:
            result[normalized_key] = (raw_score - min_score) / (max_score - min_score)

    return results
    
def merge_retrieval_results(dense_results:list[dict],bm25_results:list[dict])->list[dict]:
    merged={}
    
    for result in dense_results:
        chunk_id=result.get('chunk_id')
        if not chunk_id:
            continue
        merged[chunk_id]=result.copy()
        merged[chunk_id]['dense_rank']=result.get('rank')
        merged[chunk_id]['dense_similarity']=result.get('similarity')
        merged[chunk_id]['dense_score_norm']=result.get('dense_score_norm',0.0)
        
    for result in bm25_results:
        chunk_id=result.get('chunk_id')
        if not chunk_id:
            continue
        if chunk_id not in merged:
            merged[chunk_id] = result.copy()

        merged[chunk_id]["bm25_rank"] = result.get("rank")
        merged[chunk_id]["bm25_score"] = result.get("bm25_score")
        merged[chunk_id]["bm25_score_norm"] = result.get("bm25_score_norm", 0.0)
        
    for result in merged.values():
        result.setdefault("dense_rank", None)
        result.setdefault("dense_similarity", None)
        result.setdefault("dense_score_norm", 0.0)

        result.setdefault("bm25_rank", None)
        result.setdefault("bm25_score", None)
        result.setdefault("bm25_score_norm", 0.0)

    return list(merged.values())

def is_practice_query(query: str) -> bool:
    query = query.lower()

    practice_terms = [
        "exercise",
        "practice",
        "problem",
        "homework",
        "quiz",
        "conceptual",
        "applied",
        "question",
        "questions",
    ]

    return any(term in query for term in practice_terms)

def normalize_section_name(section: str) -> str:
    return " ".join(str(section).strip().lower().split())


def is_exercise_section(result: dict) -> bool:
    metadata = result.get("metadata", {})
    section = normalize_section_name(metadata.get("section", ""))

    exact_exercise_sections = {
        "conceptual",
        "applied",
        "exercises",
        "exercise",
        "problems",
    }

    if section in exact_exercise_sections:
        return True

    # Allow section names like:
    # "Exercises"
    # "Chapter 2 Exercises"
    # but avoid matching:
    # "2.1.5 Regression Versus Classification Problems"
    if section.startswith("exercises"):
        return True

    if section.endswith("exercises"):
        return True

    return False

def filter_exercise_sections(
    results: list[dict],
    query: str,
    min_results: int = 1,
) -> list[dict]:
    """
    Remove exercise/practice sections for normal study questions.

    If the query asks for practice/exercises, keep them.
    If filtering leaves at least min_results, use the filtered set.
    Otherwise, fall back to original results.
    """
    if is_practice_query(query):
        return results

    filtered_results = [
        result
        for result in results
        if not is_exercise_section(result)
    ]

    if len(filtered_results) >= min_results:
        return filtered_results

    return results

def get_section_penalty(result: dict, query: str) -> float:
    metadata = result.get("metadata", {})
    section = str(metadata.get("section", "")).lower()

    # If the user is asking for exercises/practice, do not penalize exercise chunks.
    if is_practice_query(query):
        return 1.0

    exercise_sections = [
        "conceptual",
        "applied",
        "exercises",
        "exercise",
    ]

    lab_sections = [
        "lab",
    ]

    if any(term in section for term in exercise_sections):
        return 0.25

    if any(term in section for term in lab_sections):
        return 0.85

    return 1.0

def compute_hybrid_score(merged_results: list[dict], alpha: float, query: str) -> list[dict]:
    for result in merged_results:
        dense_component = result.get("dense_score_norm", 0.0)
        bm25_component = result.get("bm25_score_norm", 0.0)

        raw_hybrid_score = alpha * dense_component + (1 - alpha) * bm25_component

        section_penalty = get_section_penalty(result, query)

        final_hybrid_score = raw_hybrid_score * section_penalty

        result["hybrid_score_raw"] = raw_hybrid_score
        result["section_penalty"] = section_penalty
        result["hybrid_score"] = final_hybrid_score

    scored_results = sorted(
        merged_results,
        key=lambda x: x.get("hybrid_score", 0.0),
        reverse=True,
    )

    for index, result in enumerate(scored_results, start=1):
        result["rank"] = index

    return scored_results


def hybrid_retrieve(query:str,collection,
                    chunk_records:list[dict],embedding_model:str,
                    dense_k:int,bm25_k:int,top_k:int,alpha:float) -> list[dict]:
    
    dense_results = retrieve_relevant_chunks(
        query=query,
        collection=collection,
        model_name=embedding_model,
        top_k=dense_k)
    
    dense_results = normalize_scores(
        results=dense_results,
        score_key='similarity',
        normalized_key='dense_score_norm')
    
    bm25_results=bm25_retrieve(
        query=query,
        chunk_records=chunk_records,
        top_k=bm25_k)

    bm25_results = normalize_scores(
        results=bm25_results,
        score_key='bm25_score',
        normalized_key='bm25_score_norm')
    
    merged_results = merge_retrieval_results(
        dense_results=dense_results,
        bm25_results=bm25_results)

    scored_results = compute_hybrid_score(
        merged_results=merged_results,
        alpha=alpha,
        query=query,
    )

    scored_results = filter_exercise_sections(
        results=scored_results,
        query=query,
        min_results=1,
    )
    log.info(
        f'Hybrid retrieval returned {min(top_k, len(scored_results))} results '
        f'from {len(merged_results)} merged candidates.')
    
    return scored_results[:top_k]
    

