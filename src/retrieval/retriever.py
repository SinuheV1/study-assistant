from src.utils.logging import setup_logger
from src.embedding.embedder import load_embedding_model
from src.vector_store.vectordb import query_collection
import ollama


log = setup_logger(__name__)

def embed_query(query, model_name):
    if not query or not query.strip():
        log.warning("Query is empty")
        return None

    model = load_embedding_model(model_name)

    response = ollama.embed(
        model=model,
        input=query)

    query_embedding = response["embeddings"][0]

    if query_embedding is None:
        log.warning("Empty query. No query embedding generated. Ending retrieval")
        return None

    return query_embedding

def search_vector_store(collection, query_embedding, top_k):
    results = query_collection(collection, query_embedding, top_k)
    return results

def format_retrieval_results(raw_results):
    formatted_results = []

    for ids, docs, metadatas, distances in zip(
        raw_results["ids"],
        raw_results["documents"],
        raw_results["metadatas"],
        raw_results["distances"]
    ):
        for rank, (chunk_id, doc, meta, dist) in enumerate(
            zip(ids, docs, metadatas, distances),
            start=1
        ):
            similarity = 1 / (1 + dist)

            result_record = {
                "rank": rank,
                "chunk_id": chunk_id,
                "chunk_text": doc,
                "metadata": meta,
                "distance": dist,
                "similarity": similarity
            }

            formatted_results.append(result_record)

    return formatted_results

def retrieve_relevant_chunks(query, collection, model_name, top_k):
    query_embedding = embed_query(query, model_name)

    if query_embedding is None:
        return []

    raw_results = search_vector_store(collection, query_embedding, top_k)
    formatted_results = format_retrieval_results(raw_results)

    log.info(f"Number of Chunks retrieved: {len(formatted_results)}")

    return formatted_results

def print_retrieval_results(results):
    for result in results:
        metadata = result.get("metadata", {})
        chunk_text = result.get("chunk_text", "")

        print(f"Rank: {result.get('rank')}")
        print(f"Distance: {result.get('distance')}")
        print(f"Source Type: {metadata.get('source_type')}")
        print(f"Title: {metadata.get('title')}")
        print(f"Chunk Text Preview: {chunk_text[:30]}")
        print(f"Similarity: {result.get('similarity')}")