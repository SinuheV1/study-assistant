import json
from pathlib import Path

import bm25s

from src.utils.logging import setup_logger

log = setup_logger(__name__)


def load_chunk_records(chunk_dir: str) -> list[str]:
    chunk_dir = Path(chunk_dir)
    if not chunk_dir.exists():
        log.warning(f"Path does not exist. {chunk_dir}")
        return None

    # add all chunk files to list
    chunk_records = []
    for file_path in chunk_dir.glob("*.json"):
        if not file_path.is_file():
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        chunk_records.extend(records)

    log.info(f"Loaded {len(chunk_records)} chunk records from {chunk_dir}")
    return chunk_records


def build_bm25_index(chunk_records: list) -> None:
    corpus = []
    # add text from each chunk to corpus
    for chunk in chunk_records:
        text = chunk.get("chunk_text", "")
        corpus.append(text)

    # initialize bm25 object and tokenize and index
    tokenized_corpus = bm25s.tokenize(corpus)
    retriever = bm25s.BM25()
    retriever.index(tokenized_corpus)

    return retriever


def bm25_retrieve(query: str, chunk_records: list[str], top_k: int) -> list[str]:
    if not chunk_records:
        log.warning("No chunk records provided to BM25 retriever.")
        return []
    retriever = build_bm25_index(chunk_records)
    tokenized_query = bm25s.tokenize(query)
    results, scores = retriever.retrieve(tokenized_query, k=top_k)
    bm25_results = []
    for rank, (chunk_index, score) in enumerate(zip(results[0], scores[0]), start=1):
        chunk_record = chunk_records[int(chunk_index)]
        bm25_results.append(
            {
                "rank": rank,
                "chunk_id": chunk_record.get("chunk_id"),
                "chunk_text": chunk_record.get("chunk_text", ""),
                "metadata": chunk_record.get("metadata", {}),
                "bm25_score": float(score),
            }
        )
    return bm25_results
