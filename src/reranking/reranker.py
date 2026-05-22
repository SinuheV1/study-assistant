from src.utils.logging import setup_logger
from sentence_transformers import CrossEncoder

log = setup_logger(__name__)


def load_reranker_model(model_name: str):
    model=CrossEncoder(model_name)
    if model:
        log.info(f'CrossEncoder Model Successfully loaded: {model_name}')
    return model

def score_query_chunk_pairs(query:str, retrieved_results:str, reranker_model:str) -> str:
    #evaluate query+chunk together
    pairs=[]
    for result in retrieved_results:
        chunk_text=result['chunk_text']
        pair=[query,chunk_text]
        pairs.append(pair)
    scores=reranker_model.predict(pairs)
    return scores

def rerank_results(query:str,retrieved_results:list[dict],model_name:str,top_k:int):
    '''
    takes the chunks returned by dense/vector retrieval, scores each chunk again with a reranker,
    sorts by the reranker score, then returns the best chunks
    '''
    if not retrieved_results:
        log.info(f'Retrieved results empty. Returning empty list.')
        return []
    model=load_reranker_model(model_name=model_name)
    scores=score_query_chunk_pairs(query=query,retrieved_results=retrieved_results,reranker_model=model)
    reranked_results=[]
    #loop through dense/vector retrieval result and its score together
    for result,score in zip(retrieved_results,scores):
        result=result.copy()
        #store original dense/vector rank and similarity
        result['dense_rank']=result.get('rank')
        result['dense_similarity']=result.get('similarity')
        #store reranker score
        result['rerank_score']=float(score)
        reranked_results.append(result)
    #sort chunk from highest reranker score to lowest
    reranked_results=sorted(reranked_results,key=lambda x: x.get('rerank_score',float('-inf')),reverse=True)
    #loop through sorted reranker results and start counting at 1
    for index,result in enumerate(reranked_results,start=1):
        #overwrite rank with final reranked rank
        result['rank']=index
    #return only top reranked results
    return reranked_results[:top_k]