from src.retrieval.retriever import print_retrieval_results, retrieve_relevant_chunks
from src.vector_store.vectordb import get_or_create_collection, initialize_vector_db

persist_directory = "data/processed/vector_store"
collection_name = "study_assistant_chunks"
model_name = "all-MiniLM-L6-v2"
top_k = 2

query = "What are the assumptions of linear regression?"

client = initialize_vector_db(persist_directory)
collection = get_or_create_collection(client, collection_name)

results = retrieve_relevant_chunks(
    query=query, collection=collection, model_name=model_name, top_k=top_k
)

print_retrieval_results(results)
