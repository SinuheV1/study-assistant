from src.chunking.chunker import chunk_document
from src.embedding.embedder import embed_chunks
from src.ingestion.ingest_text import ingest_text_document
from src.utils.config import PROJECT_ROOT, load_config
from src.vector_store.vectordb import (
    add_records_to_collection,
    get_collection_count,
    initialize_vector_db,
    reset_collection,
)

file_path = "data/raw/lecture_transcripts/example_ingestion_test_messy.txt"

config = load_config()
persist_directory = PROJECT_ROOT / "data" / "processed" / "vector_store_smoke"
collection_name = config["vector_store"]["collection_name"]
embedding_model = config["models"]["embedding"]

target_size = config["chunking"]["target_size"]
overlap_size = config["chunking"]["overlap_size"]


# step 1 ingest and clean
ingestion_result = ingest_text_document(file_path)
cleaned_text = ingestion_result["cleaned_text"]
metadata = ingestion_result["metadata"]

print("\nINGESTION COMPLETE")
print(metadata)


# step 2 chunk
chunk_records = chunk_document(
    cleaned_text=cleaned_text,
    document_metadata=metadata,
    target_size=target_size,
    overlap_size=overlap_size,
)

print(f"\nCHUNKS CREATED: {len(chunk_records)}")
print(chunk_records[0]["chunk_text"])


# step 3 embed
embedded_records = embed_chunks(chunk_records=chunk_records, model_name=embedding_model)

print(f"\nEMBEDDED RECORDS: {len(embedded_records)}")
print(f"Embedding length: {len(embedded_records[0]['embedding'])}")


# step 4 vector db
client = initialize_vector_db(persist_directory)

collection = reset_collection(client=client, collection_name=collection_name)

add_records_to_collection(collection=collection, embedded_chunk_records=embedded_records)

count = get_collection_count(collection)

print(f"\nVECTOR DB COUNT: {count}")
print("TEST PASSED" if count == len(embedded_records) else "TEST FAILED")
