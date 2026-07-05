import argparse
from typing import Any
from pathlib import Path
from src.utils.logging import setup_logger
from src.utils.io import save_extracted_text,save_json
from src.ingestion.ingest_docling_document import ingest_docling_document
from src.ingestion.ingest_text import ingest_text_document
from src.chunking.chunker import chunk_document
from src.embedding.embedder import embed_chunks
from src.utils.config import load_config
from src.vector_store.vectordb import (initialize_vector_db,get_or_create_collection,add_records_to_collection,
    get_collection_count,reset_collection)


log = setup_logger(__name__)
#Config Variables


config = load_config()
target_size = config["chunking"]["target_size"]
overlap_size = config["chunking"]["overlap_size"]
embedding_model = config["models"]["embedding"]
persist_dir = config["paths"]["persist_dir"]
collection_name = config["vector_store"]["collection_name"]
extracted_text_dir = config["paths"]["extracted_text_dir"]
chunks_dir = config["paths"]["chunk_dir"]
embeddings_dir = config["paths"]["embeddings_dir"]
SUPPORTED_EXTENSIONS={".pdf",'.txt','.md'}

def parse_args():
    parser = argparse.ArgumentParser(description="Run ingestion pipeline. Exactly one of --file or --dir is required to run.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file",
        "-f",
        help="Path to the file to ingest (.pdf, .txt, .md).")
    group.add_argument(
        "--dir",
        help='Path to directory to ingest all supported files (.pdf, .txt, .md)')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Scan nested directories for .pdf, .txt, .md files. Can only be used if --dir was used.')
    parser.add_argument(
        '--reset-collection',
        action='store_true',
        help='Reset the Chroma collection before ingestion. ')
    
    args=parser.parse_args()
    
    if args.recursive and not args.dir:
        parser.error("--recursive can only be used with --dir.")
    
    return args

def is_supported_file(file_path: str | Path)-> bool:
    if not file_path.is_file():
        return False
    if file_path.name ==".DS_Store":
        return False
    if file_path.name.startswith("."):
        return False
    
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def discover_supported_files(directory_path: str | Path, recursive: bool  = False) -> list[Path]:
    directory_path = Path(directory_path)
    
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory_path}")
    
    if not directory_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {directory_path}")
    
    if recursive:
        candidates = directory_path.rglob("*")
    else:
        candidates = directory_path.iterdir()
    
    files=[path for path in candidates if is_supported_file(path)]
    
    return sorted(files)

def ingest_file_type(file_path: str | Path) -> tuple[dict[str, Any] | None, str]:
    file_path = Path(file_path)
    file_name = file_path.stem
    suffix = file_path.suffix.lower()

    match suffix:
        case ".pdf":
            document = ingest_docling_document(file_path)
        case ".txt":
            document = ingest_text_document(file_path)
        case ".md":
            document = ingest_text_document(file_path)
        case _:
            raise ValueError(f"Unsupported file type: {file_path.name}")

    return document, file_name

def load_vectordb_collection(reset: bool = False):
    client=initialize_vector_db(str(persist_dir))
    
    if reset:
        collection=reset_collection(client,collection_name)
    else:
        collection = get_or_create_collection(client,collection_name)
        
    return collection

def ingest_document(document: str, file_name:str,collection: str):
    log.info(f'Starting Ingestion Pipeline for document: {file_name}')

    if document is None:
        log.warning(f"Document returned None for file: {file_name}")
        return None
    
    metadata=document.get('metadata',{})
    document_id=metadata.get('document_id','')
    
    if not document_id:
        log.warning(f"Document ID missing for file: {file_name}")
        return None
    cleaned_text=document.get("cleaned_text")
    if not cleaned_text:
        log.warning(f"Cleaned text missing or empty for file: {file_name}")
        
    save_extracted_text(document,extracted_text_dir)
    
    chunks=chunk_document(cleaned_text=cleaned_text,document_metadata=metadata,
                        target_size=target_size,overlap_size=overlap_size)
    
    if not chunks:
        log.warning(f'Chunks returned empty for file: {file_name}')
        return None
        
    save_json(chunks, Path(chunks_dir) / f'{document_id}_chunks.json')
    
    embedded_chunks=embed_chunks(chunk_records=chunks,model_name=embedding_model)
    
    if not embedded_chunks: 
        log.warning(f'Embedded chunks returned empty. ')
        return None
    
    save_json(embedded_chunks,Path(embeddings_dir) / f'{document_id}_embeddings.json')
    
    add_records_to_collection(collection,embedded_chunks)
    
    vector_count=get_collection_count(collection)
    
    result = {
        "file_name": metadata.get("file_name", file_name),
        "document_id": document_id,
        "chunks_created": len(chunks),
        "embeddings_created": len(embedded_chunks),
        "vector_db_count": vector_count,
        "status": "success",
        "error": None 
    }
    print("\n==== Document Ingestion Summary ====")
    print(f"Document ID: {result['document_id']}")
    print(f"Document: {result['file_name']}")
    print(f"Chunks created: {result['chunks_created']}")
    print(f"Embeddings created: {result['embeddings_created']}")
    print(f"Vector DB count: {result['vector_db_count']}")

    log.info(f"Ingestion completed for document: {file_name}")

    return result

def process_one_file(file_path: str | Path, collection) -> dict[str, Any]:
    file_path = Path(file_path)

    try:
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        if not file_path.is_file():
            raise IsADirectoryError(f"Input path is not a file: {file_path}")

        if not is_supported_file(file_path):
            raise ValueError(f"Unsupported file type: {file_path.name}")

        log.info(f"Ingesting file: {file_path}")

        document, file_name = ingest_file_type(file_path)

        result = ingest_document(
            document=document,
            file_name=file_name,
            collection=collection,
        )

        if result is None:
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "document_id": None,
                "chunks_created": 0,
                "embeddings_created": 0,
                "status": "failed",
                "error": "Ingestion returned None.",
            }

        result["file_path"] = str(file_path)
        return result

    except Exception as e:
        log.exception(f"Failed to ingest file: {file_path}. Reason: {e}")

        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "document_id": None,
            "chunks_created": 0,
            "embeddings_created": 0,
            "status": "failed",
            "error": str(e),
        }

def print_batch_summary(results: list[dict[str, Any]], collection) -> None:
    processed = [result for result in results if result.get("status") == "success"]
    failed = [result for result in results if result.get("status") == "failed"]

    total_chunks = sum(result.get("chunks_created", 0) for result in processed)
    total_embeddings = sum(result.get("embeddings_created", 0) for result in processed)

    vector_count = get_collection_count(collection)

    print("\n==============================")
    print("==== Ingestion Run Summary ====")
    print("==============================")
    print(f"Files attempted: {len(results)}")
    print(f"Processed: {len(processed)}")
    print(f"Failed: {len(failed)}")
    print(f"Chunks created: {total_chunks}")
    print(f"Embeddings created: {total_embeddings}")
    print(f"Vector DB count: {vector_count}")

    if failed:
        print("\nFailures:")
        for result in failed:
            print(f"- {result.get('file_path')}: {result.get('error')}")

    
def main():
    args = parse_args()

    collection = load_vectordb_collection(reset=args.reset_collection)

    if args.file:
        file_path = Path(args.file)

        if not file_path.exists():
            log.error(f"File does not exist: {file_path}")
            return

        if not file_path.is_file():
            log.error(f"Input path is not a file: {file_path}")
            return

        if not is_supported_file(file_path):
            log.error(f"Unsupported file type: {file_path.name}")
            return

        files_to_process = [file_path]

    else:
        directory_path = Path(args.dir)

        try:
            files_to_process = discover_supported_files(
                directory_path=directory_path,
                recursive=args.recursive,
            )
        except Exception as e:
            log.exception(f"Failed to discover files in directory: {directory_path}. Reason: {e}")
            return

        if not files_to_process:
            log.info(f"No supported files found in directory: {directory_path}")
            print(f"No supported files found in directory: {directory_path}")
            return

        if args.recursive:
            log.info(
                f"Found {len(files_to_process)} valid files at {directory_path} and child directories."
            )
        else:
            log.info(f"Found {len(files_to_process)} valid files at {directory_path}.")

    results = []

    for file_path in files_to_process:
        result = process_one_file(
            file_path=file_path,
            collection=collection,
        )
        results.append(result)

    print_batch_summary(results, collection)

    log.info("Ingestion pipeline run completed.")


if __name__ == "__main__":
    main()
    
