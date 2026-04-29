from src.utils.logging import setup_logger
import os
import chromadb
from chromadb.errors import NotFoundError

log=setup_logger(__name__)


def initialize_vector_db(persist_directory):
    
    os.makedirs(persist_directory, exist_ok=True)
    chroma_client=chromadb.PersistentClient(path=persist_directory)
    
    log.info(f'Client Successfully loaded at directory: {persist_directory}')
    return chroma_client

def get_or_create_collection(chroma_client,collection_name):
    
    #get or create collection if not exist
    collection=chroma_client.get_or_create_collection(name=collection_name)
    
    log.info(f'Collection: {collection_name} Ready.')
    
    return collection

def validate_embedded_chunk_records(embedded_chunk_records):
    
    #initialize empty list to hold valid records
    valid_records=[]
    
    for record in embedded_chunk_records:
        chunk_id=record.get('chunk_id')
        chunk_text=record.get('chunk_text')
        metadata=record.get('metadata')
        embedding=record.get('embedding')
        
        if not chunk_id:
            log.warning(f"Chunk_id missing from {record}")
            continue
        if not chunk_text:
            log.warning(f"Chunk_text missing from {record}")
            continue
        if not metadata or not isinstance(metadata,dict):
            log.warning(f'Metadata missing from {record}')
            continue
        if embedding is None:
            log.warning(f"Embedding is None")
            continue
        elif len(embedding)==0:
                log.warning(f'Embedding has length 0(empty)')
                continue
        for k,v in metadata.items():
            if v is None:
                metadata[k] = ''
                log.info(f'Metadata {k} value found as None. Converted to empty string for insertion. Chunk_id : {chunk_id}')
    
        valid_records.append(record)
    return valid_records

def prepare_records_for_insertion(embedded_chunk_records):
    #initialize empty lists
    ids=[]
    documents=[]
    metadatas=[]
    embeddings=[]
    
    
    for record in embedded_chunk_records:
        ids.append(record['chunk_id'])
        documents.append(record['chunk_text'])
        metadatas.append(record['metadata'])
        embeddings.append(record['embedding'])
        
    assert len(ids) == len(documents) == len(metadatas) == len(embeddings)
    log.info(f'Lists: ids, documents, metadatas, and embeddings - have equal length')
    return ids, documents, metadatas, embeddings

def add_records_to_collection(collection,embedded_chunk_records):
    
    valid_records=validate_embedded_chunk_records(embedded_chunk_records)
    
    if not valid_records:
        log.warning(f'Valid Records does not exist or is empty.')
        return
    
    ids,documents,metadatas,embeddings=prepare_records_for_insertion(valid_records)
    
    collection.add(ids=ids,documents=documents,metadatas=metadatas,embeddings=embeddings)
    
    log.info(f'Number of valid records added to collection: {len(valid_records)}')
    
    
def get_collection_count(collection):
    count=collection.count()
    log.info(f'Number of records in collection: {count}')
    return count

def query_collection(collection,query_embedding,top_k):
    results=collection.query(query_embeddings=[query_embedding],n_results=top_k)
    return results
    
def reset_collection(client,collection_name):
    try:
        client.delete_collection(name=collection_name)
        log.info(f'Deleted collection: {collection_name}')
    except NotFoundError:
        log.info(f'Collection: {collection_name} does not exist')
    new_collection=client.create_collection(name=collection_name)
    log.info(f'New collection created: {collection_name}')
    return new_collection
    
