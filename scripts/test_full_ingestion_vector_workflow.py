from src.ingestion.ingest_text import main as ingest_text
from src.chunking.chunker import chunk_document
from src.embedding.embedder import embed_chunks
from src.vector_store.vectordb import (
    initialize_vector_db,
    reset_collection,
    add_records_to_collection,
    get_collection_count)

file_path = "data/raw/lecture_transcripts/example_ingestion_test_messy.txt"

persist_directory = "data/processed/vector_store"
collection_name = "study_assistant_chunks"
embedding_model = "all-MiniLM-L6-v2"

target_size = 400
overlap_size = 100


#step 1 ingest and clean
ingestion_result = ingest_text(file_path)
cleaned_text = ingestion_result["cleaned_text"]
metadata = ingestion_result["metadata"]

print("\nINGESTION COMPLETE")
print(metadata)


#step 2 chunk
chunk_records = chunk_document(cleaned_text=cleaned_text,document_metadata=metadata,
    target_size=target_size,
    overlap_size=overlap_size)

print(f"\nCHUNKS CREATED: {len(chunk_records)}")
print(chunk_records[0]["chunk_text"])


#step 3 embed
embedded_records = embed_chunks(chunk_records=chunk_records,model_name=embedding_model)

print(f"\nEMBEDDED RECORDS: {len(embedded_records)}")
print(f"Embedding length: {len(embedded_records[0]['embedding'])}")


#step 4 vector db
client = initialize_vector_db(persist_directory)

collection = reset_collection(client=client,collection_name=collection_name)

add_records_to_collection(collection=collection,embedded_chunk_records=embedded_records)

count = get_collection_count(collection)

print(f"\nVECTOR DB COUNT: {count}")
print("TEST PASSED" if count == len(embedded_records) else "TEST FAILED")
