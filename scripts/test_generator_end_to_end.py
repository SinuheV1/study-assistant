from src.vector_store.vectordb import initialize_vector_db, get_or_create_collection
from src.retrieval.retriever import retrieve_relevant_chunks, print_retrieval_results
from src.generation.generator import generate_answer, format_answer_with_sources

persist_directory = "data/processed/vector_store"
collection_name = "study_assistant_chunks"
embedding_model = "all-MiniLM-L6-v2"
llm_model = "llama3.1:8b"
top_k = 2

query = "What are the assumptions of linear regression?"

client = initialize_vector_db(persist_directory)
collection = get_or_create_collection(client, collection_name)

retrieved_results = retrieve_relevant_chunks(
    query=query,
    collection=collection,
    model_name=embedding_model,
    top_k=top_k,
)

print("\n=== RETRIEVED CHUNKS ===")
print_retrieval_results(retrieved_results)

answer = generate_answer(
    query=query,
    retrieved_results=retrieved_results,
    model_name=llm_model,
)

formatted_answer = format_answer_with_sources(
    answer=answer,
    retrieved_results=retrieved_results,
)

print("\n=== GENERATED ANSWER ===")
print(formatted_answer)
