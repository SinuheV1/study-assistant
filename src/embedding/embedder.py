from src.utils.logging import setup_logger
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os



log = setup_logger(__name__)
_model_cache: dict = {}

def load_embedding_model(model_name: str):
    if model_name not in _model_cache:
            _model_cache[model_name] = SentenceTransformer(model_name)
            log.info(f"Embedding model loaded: {model_name}")
        
    return _model_cache[model_name]


def validate_chunk_records(chunk_records: list[dict]) -> list[dict]:
    
    valid_chunk_records=[]
    for chunk_record in chunk_records:
        chunk_text = chunk_record.get("chunk_text")
        if not chunk_text or not chunk_text.strip():
            log.warning(f"Skipping chunk '{chunk_record.get('chunk_id', 'unknown')}': "
                        "missing or empty chunk_text.")
            continue
        valid_chunk_records.append(chunk_record)
        
    return valid_chunk_records

def extract_texts_for_embeddings(chunk_records: list[dict]) -> list[str]:

    return [record["chunk_text"] for record in chunk_records]

def generate_embeddings(model, texts: list[str], batch_size: int = 64):

    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return embeddings


def attach_embeddings_to_chunks(chunk_records: list[dict], embeddings: np.ndarray, model_name: str) -> list[dict]:
    
    embedded_chunk_records=[]
    for index,chunk_record in enumerate(chunk_records):
        embedded_record = {
            "chunk_id": chunk_record["chunk_id"],
            "chunk_text": chunk_record["chunk_text"],
            "metadata": chunk_record["metadata"],
            "embedding": embeddings[index].tolist(),
            "embedding_model": model_name}
        
        embedded_chunk_records.append(embedded_record)
    return embedded_chunk_records
        

def embed_chunks(chunk_records: list[dict], model_name: str, batch_size: int = 64) -> list[dict]:

    valid_chunk_records=validate_chunk_records(chunk_records)
    dropped = len(chunk_records) - len(valid_chunk_records)
    if dropped > 0:
        log.warning(f"{dropped} chunk(s) failed validation and were skipped.")
    if not valid_chunk_records:
        log.error("No valid chunk records to embed. Returning empty list.")
        return []

    model=load_embedding_model(model_name)
    texts=extract_texts_for_embeddings(valid_chunk_records)
    embeddings = generate_embeddings(model, texts, batch_size=batch_size)
    
    embedded_chunk_records=attach_embeddings_to_chunks(valid_chunk_records,embeddings,model_name)
    
    log.info(f'Number of chunks embedded: {len(embedded_chunk_records)}')
    
    return embedded_chunk_records
    
    
def save_embedded_chunks(embedded_chunk_records: list[dict], output_path: str) -> None:

    os.makedirs(output_path, exist_ok=True)
    if not embedded_chunk_records:
            log.warning("No embedded chunk records provided. Nothing was saved.")
            return

    document_id = embedded_chunk_records[0]["metadata"]["document_id"]
    file_name = f"{document_id}_embedded_chunks.json"
    full_output_path = os.path.join(output_path, file_name)
    
    with open(full_output_path, "w", encoding="utf-8") as f:
        json.dump(embedded_chunk_records, f, indent=2, ensure_ascii=False)

    log.info(f"Saved {len(embedded_chunk_records)} embedded chunk records to {full_output_path}")