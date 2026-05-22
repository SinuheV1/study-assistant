import json
from src.vector_store.vectordb import initialize_vector_db, get_or_create_collection
from src.retrieval.retriever import retrieve_relevant_chunks
from src.generation.generator import generate_answer
from src.reranking.reranker import rerank_results

persist_directory='data/processed/vector_store'
collection_name='study_assistant_chunks'
embed_model='all-MiniLM-L6-v2'
llm_model='llama3.2:3b'
top_k=3
candidate_k=8
reranker_model='cross-encoder/ms-marco-MiniLM-L-6-v2'


def keyword_score(text,keywords):
    if not keywords:
        return 0
    text=text.lower()
    hits=sum(1 for kw in keywords if kw.lower() in text)
    return hits/len(keywords)

def score_results(results, keywords):
    combined_chunks = ' '.join(r['chunk_text'] for r in results)
    return keyword_score(combined_chunks, keywords)

def print_debug_sources(label, results):
    print(f'\n=== DEBUG {label} SOURCES ===')

    for r in results:
        metadata = r.get('metadata', {})

        print(f"Rank: {r.get('rank')}")
        print(f"Dense Rank: {r.get('dense_rank')}")
        print(f"Dense Similarity: {r.get('dense_similarity')}")
        print(f"Rerank Score: {r.get('rerank_score')}")
        print(f"Document ID: {metadata.get('document_id')}")
        print(f"File: {metadata.get('file_name')}")
        print(f"Course: {metadata.get('course')}")
        print(f"Section: {metadata.get('section')}")
        print(f"Source Type: {metadata.get('source_type')}")
        print(f"Preview: {r.get('chunk_text', '')[:120]}")
        print('----')


def run_evaluation():
    with open('evaluation/queries.json') as f:
        data=json.load(f)
    queries=data['queries']
        
    client=initialize_vector_db(persist_directory)
    collection=get_or_create_collection(client,collection_name)
    
    dense_retrieval_scores = []
    dense_generation_scores = []

    reranked_retrieval_scores = []
    reranked_generation_scores = []

    changed_top_result_count = 0
    
    for q in queries:
        query = q['query']
        keywords = q['expected_keywords']

        dense_results = retrieve_relevant_chunks(
            query=query,
            collection=collection,
            model_name=embed_model,
            top_k=top_k)

        dense_r_score = score_results(dense_results, keywords)
        dense_answer = generate_answer(query, dense_results, llm_model)
        dense_g_score = keyword_score(dense_answer or '', keywords)

        dense_retrieval_scores.append(dense_r_score)
        dense_generation_scores.append(dense_g_score)

        candidate_results = retrieve_relevant_chunks(
            query=query,
            collection=collection,
            model_name=embed_model,
            top_k=candidate_k)

        reranked_results = rerank_results(
            query=query,
            retrieved_results=candidate_results,
            model_name=reranker_model,
            top_k=top_k)

        reranked_r_score = score_results(reranked_results, keywords)
        reranked_answer = generate_answer(query, reranked_results, llm_model)
        reranked_g_score = keyword_score(reranked_answer or '', keywords)

        reranked_retrieval_scores.append(reranked_r_score)
        reranked_generation_scores.append(reranked_g_score)

        dense_top_chunk = dense_results[0].get('chunk_id') if dense_results else None
        reranked_top_chunk = reranked_results[0].get('chunk_id') if reranked_results else None

        if dense_top_chunk != reranked_top_chunk:
            changed_top_result_count += 1

        print(f'\nQuery: {query}')
        print(f'Dense Retrieval Score: {dense_r_score:.2f}')
        print(f'Dense Generation Score: {dense_g_score:.2f}')
        print(f'Reranked Retrieval Score: {reranked_r_score:.2f}')
        print(f'Reranked Generation Score: {reranked_g_score:.2f}')
        print(f'Top Result Changed: {dense_top_chunk != reranked_top_chunk}')

        print_debug_sources('DENSE', dense_results)
        print_debug_sources('RERANKED', reranked_results)

    dense_avg_retrieval = sum(dense_retrieval_scores) / len(dense_retrieval_scores)
    dense_avg_generation = sum(dense_generation_scores) / len(dense_generation_scores)

    reranked_avg_retrieval = sum(reranked_retrieval_scores) / len(reranked_retrieval_scores)
    reranked_avg_generation = sum(reranked_generation_scores) / len(reranked_generation_scores)

    print('\n=== FINAL RESULTS ===')

    print('\n=== DENSE BASELINE ===')
    print(f'Avg Retrieval Score: {dense_avg_retrieval:.2f}')
    print(f'Avg Generation Score: {dense_avg_generation:.2f}')

    print('\n=== DENSE + RERANKER ===')
    print(f'Avg Retrieval Score: {reranked_avg_retrieval:.2f}')
    print(f'Avg Generation Score: {reranked_avg_generation:.2f}')

    print('\n=== DELTAS ===')
    print(f'Retrieval Delta: {reranked_avg_retrieval - dense_avg_retrieval:+.2f}')
    print(f'Generation Delta: {reranked_avg_generation - dense_avg_generation:+.2f}')
    print(f'Top Result Changed: {changed_top_result_count}/{len(queries)} queries')


if __name__ == "__main__":
    run_evaluation()