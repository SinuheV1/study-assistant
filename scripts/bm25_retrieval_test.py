from src.retrieval.bm25_retriever import bm25_retrieve, load_chunk_records

chunks = load_chunk_records("data/processed/chunks")
results = bm25_retrieve("least squares", chunks, top_k=5)

for r in results:
    print(r["rank"], r["bm25_score"], r["metadata"].get("section"))
    print(r["chunk_text"][:200])
