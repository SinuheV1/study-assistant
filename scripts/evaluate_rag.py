import json
from src.vector_store.vectordb import initialize_vector_db, get_or_create_collection
from src.retrieval.retriever import retrieve_relevant_chunks
from src.generation.generator import generate_answer

persist_directory='data/processed/vector_store'
collection_name='study_assistant_chunks'
embed_model='all-MiniLM-L6-v2'
llm_model='llama3.2:3b'
top_k=3

def keyword_score(text,keywords):
    text=text.lower()
    hits=sum(1 for kw in keywords if kw.lower() in text)
    return hits/len(keywords)

def run_evaluation():
    with open('evaluation/queries.json') as f:
        data=json.load(f)
        queries=data['queries']
        
        client=initialize_vector_db(persist_directory)
        collection=get_or_create_collection(client,collection_name)
        
        retrieval_scores=[]
        generation_scores=[]
        
        for q in queries:
            query=q['query']
            keywords=q['expected_keywords']
            
            results=retrieve_relevant_chunks(query=query,collection=collection,model_name=embed_model,top_k=top_k)
            print("\n=== DEBUG RETRIEVED SOURCES ===")
            for r in results:
                metadata = r.get("metadata", {})
                print(f"Document ID: {metadata.get('document_id')}")
                print(f"File: {metadata.get('file_name')}")
                print(f"Source Type: {metadata.get('source_type')}")
                print(f"Preview: {r.get('chunk_text')[:100]}")
                print("----")
            combined_chunks=''.join(r['chunk_text'] for r in results)
            r_score=keyword_score(combined_chunks,keywords)
            retrieval_scores.append(r_score)
            
            answer=generate_answer(query,results,llm_model)
            g_score=keyword_score(answer or '',keywords)
            generation_scores.append(g_score)
            
            print(f'\nQuery: {query}')
            print(f'Retrieval Score: {r_score:.2f}')
            print(f'Generation Score: {g_score:.2f}')
        
            
        print('\n=== FINAL RESULTS ===')
        print(f'Avg Retrieval Score: {sum(retrieval_scores)/len(retrieval_scores):.2f}')
        print(f'Avg Generation Score: {sum(generation_scores)/len(generation_scores):.2f}')


if __name__ == '__main__':
    run_evaluation()