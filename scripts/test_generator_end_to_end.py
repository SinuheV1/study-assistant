from src.generation.generator import format_answer_with_sources, generate_answer
from src.retrieval.retriever import print_retrieval_results, retrieve_relevant_chunks
from src.utils.config import load_config
from src.vector_store.vectordb import get_or_create_collection, initialize_vector_db

config = load_config()
persist_directory = config["paths"]["persist_dir"]
collection_name = config["vector_store"]["collection_name"]
embedding_model = config["models"]["embedding"]
llm_model = config["models"]["llm"]
top_k = 2

query = "What are the assumptions of linear regression?"

client = initialize_vector_db(persist_directory)
collection = get_or_create_collection(client, collection_name)

retrieved_results = retrieve_relevant_chunks(
    query=query, collection=collection, model_name=embedding_model, top_k=top_k
)

print("\n=== RETRIEVED CHUNKS ===")
print_retrieval_results(retrieved_results)

answer = generate_answer(query=query, retrieved_results=retrieved_results, model_name=llm_model)

formatted_answer = format_answer_with_sources(answer=answer, retrieved_results=retrieved_results)

print("\n=== GENERATED ANSWER ===")
print(formatted_answer)
