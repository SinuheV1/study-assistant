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


def compute_hybrid_score(merged_results:list[dict],alpha:float)->list:
    for result in merged_results:
        #original chromadb rank
        dense_component=result.get('dense_score_norm',0.0)
        #original bm25 rank
        bm25_component=result.get('bm25_score_norm',0.0)
        hybrid_score=alpha*dense_component + (1-alpha) *bm25_component
        result['hybrid_score']=hybrid_score
    scored_results = sorted(merged_results, key=lambda x: x.get('hybrid_score',0.0),reverse=True)
    for index, result in enumerate(scored_results, start=1):
        #hybrid rank
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
        alpha=alpha)
    
    log.info(
        f'Hybrid retrieval returned {min(top_k, len(scored_results))} results '
        f'from {len(merged_results)} merged candidates.')
    
    return scored_results[:top_k]    

