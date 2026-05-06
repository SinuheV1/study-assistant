from pathlib import Path
from src.utils.logging import setup_logger
from src.utils.io import save_extracted_text,save_json
from src.ingestion.ingest_docling_document import ingest_docling_document
from src.chunking.chunker import chunk_document
from src.embedding.embedder import embed_chunks
from src.vector_store.vectordb import (initialize_vector_db,get_or_create_collection,add_records_to_collection,
    get_collection_count)


log = setup_logger(__name__)
#Config Variables
file_path = 'data/raw/lecture_pdfs/Lecture_01.pdf'
parser = 'docling'
target_size = 900
overlap_size = 75
embedding_model = 'all-MiniLM-L6-v2'
persist_dir = 'data/processed/vector_store'
collection_name = 'study_assistant_chunks'
extracted_text_dir = 'data/processed/extracted_texts'
chunks_dir = 'data/processed/chunks'
embeddings_dir = 'data/processed/embeddings'


def run_ingestion_pipeline():
    
    log.info(f'Starting Ingestion Pipeline.')
    document=ingest_docling_document(file_path)
    if document is None:
        log.warning(f'Document returned None.')
        return None
    metadata=document.get('metadata',{})
    document_id=metadata.get('document_id','')
    
    save_extracted_text(document,extracted_text_dir)
    
    chunks=chunk_document(cleaned_text=document['cleaned_text'],document_metadata=metadata,
                        target_size=target_size,overlap_size=overlap_size)
    
    if not chunks:
        log.warning(f'Chunks returned empty.')
        return None
        
    save_json(chunks,Path(chunks_dir) / f'{document_id}_chunks.json')
    
    embedded_chunks=embed_chunks(chunk_records=chunks,model_name=embedding_model)
    
    if not embedded_chunks: 
        log.warning(f'Embedded chunks returned empty. ')
        return None
    save_json(embedded_chunks,Path(embeddings_dir) / f'{document_id}_embeddings.json')
    
    client=initialize_vector_db(persist_dir)
    collection = get_or_create_collection(client,collection_name)
    
    add_records_to_collection(collection,embedded_chunks)
    
    count=get_collection_count(collection)
    
    print(f'==== Pipeline Summary ====')
    print(f'Document Id: {document_id}')
    print(f"Document: {metadata.get('file_name')}")
    print(f'Chunks created: {len(chunks)}')
    print(f'Embeddings created: {len(embedded_chunks)}')
    print(f'Vector DB count: {count}')
    
    log.info(f'Ingestion Pipeline completed. ')
    
    
if __name__ == '__main__':
    run_ingestion_pipeline()