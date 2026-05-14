import argparse
from src.utils.logging import setup_logger
from src.vector_store.vectordb import initialize_vector_db, get_or_create_collection
from src.retrieval.retriever import retrieve_relevant_chunks
from src.generation.generator import generate_answer


log = setup_logger(__name__)


persist_directory = "data/processed/vector_store"
collection_name = "study_assistant_chunks"
embedding_model = "all-MiniLM-L6-v2"
llm_model = "llama3.2:3b"
top_k = 4


def parse_args():
    parser=argparse.ArgumentParser('Run Query Pipeline.')
    parser.add_argument(
        '--query',
        '-q',
        required=True,
        help='Query to run. ')
    parser.add_argument(
        '--top-k',
        type=int,
        default=top_k,
        help='Number of chunks to retrieve. ')
    parser.add_argument(
        '--embedding-model',
        default=embedding_model,
        help='Specify embedding model to use.')
    parser.add_argument(
        '--model',
        '-m',
        default=llm_model,
        help='Specify Ollama model to use for generation.')
    parser.add_argument(
        '--persist-dir',
        default=persist_directory,
        help='Specify path to change persist-dir. ')
    parser.add_argument(
        '--collection',
        default=collection_name,
        help='Specify collection to use. Useful for A/B testing or experimenting. ')
    parser.add_argument(
        '--show-sources',
        action='store_true',
        help='Flag to show retrieved sources. Useful for debugging. ')
    parser.add_argument(
        '--show-context',
        action='store_true',
        help='Flag to print full retrieved chunks for debugging.')
    parser.add_argument(
        '--no-generate',
        action='store_true',
        help='Flag to turn off generation. Useful for retrieval only debugging.')
    parser.add_argument(
        '--preview-chars',
        type=int,
        default=300,
        help='Controls how much chunk text is printed. ')
    return parser.parse_args()

def load_vector_collection(persist_dir,collection_name):
    client = initialize_vector_db(persist_dir)
    collection = get_or_create_collection(client, collection_name)
    return collection

def run_retrieval(query,collection,embedding_model,top_k):
    retrieved_results=retrieve_relevant_chunks(
        query=query,
        collection=collection,
        model_name=embedding_model,
        top_k=top_k)
    return retrieved_results

def validate_retrieval_results(results):
    if not results:
        log.info('No relevant chunks found. ')
        return False
    return True

def print_sources(results,preview_chars):
    for result in results:
        metadata=result.get('metadata',{})
        rank=result.get('rank')
        chunk_id=result.get('chunk_id')
        similarity=result.get('similarity')
        chunk_text=result.get('chunk_text','')
        title=metadata.get('title')
        file_name=metadata.get('file_name')
        source_type=metadata.get('source_type')
        
        print('='*80)
        print(f'Rank: {rank}')
        print(f'Title: {title}')
        print(f'File: {file_name}')
        print(f'Source Type: {source_type}')
        print(f'Chunk ID: {chunk_id}')
        print(f'Similarity: {similarity}')
        
        print('\nPreview:')
        print(chunk_text[:preview_chars])
        print()
        
def print_context(results):
    for result in results:
        metadata=result.get('metadata',{})
        rank=result.get('rank')
        chunk_id=result.get('chunk_id')
        similarity=result.get('similarity')
        chunk_text=result.get('chunk_text', '')
        title=metadata.get('title')
        file_name=metadata.get('file_name')
        source_type=metadata.get('source_type')
        
        print('='*80)
        print(f'Chunk Rank: {rank}')
        print(f'Title: {title}')
        print(f'File Name: {file_name}')
        print(f'Source Type: {source_type}')
        print(f'Chunk ID: {chunk_id}')
        print(f'Similarity: {similarity}')
        
        print('\nFull Context:\n')
        print(chunk_text)
        print()


def run_query_pipeline(args):
    collection=load_vector_collection(
        args.persist_dir,
        args.collection)
    results=run_retrieval(
        query=args.query,
        collection=collection,
        embedding_model=args.embedding_model,
        top_k=args.top_k)
    if validate_retrieval_results(results) is False:
        return None
    if args.show_context:
        print_context(results)
    elif args.show_sources:
        print_sources(results,args.preview_chars)          
    if args.no_generate:
        return results
    answer=generate_answer(
        query=args.query,
        retrieved_results=results,
        model_name=args.model)
    print('\n=== ANSWER ====')
    print(answer)

    return answer

if __name__ == "__main__":
    args = parse_args()
    run_query_pipeline(args)