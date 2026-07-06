import argparse
import shutil
from pathlib import Path
from typing import Any

from src.chunking.chunker import chunk_document
from src.embedding.embedder import embed_chunks
from src.ingestion.ingest_docling_document import ingest_docling_document
from src.ingestion.ingest_text import ingest_text_document
from src.ingestion.manifest import (
    compute_file_hash,
    create_empty_manifest,
    is_file_unchanged,
    load_manifest,
    save_manifest,
    update_manifest_record,
)
from src.retrieval.bm25_retriever import get_bm25_index
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.io import save_extracted_text, save_json
from src.utils.logging import setup_logger
from src.vector_store.vectordb import (
    add_records_to_collection,
    delete_by_document_id,
    get_collection_count,
    get_or_create_collection,
    initialize_vector_db,
    reset_collection,
)

log = setup_logger(__name__)
# Config Variables


config = load_config()
target_size = config["chunking"]["target_size"]
overlap_size = config["chunking"]["overlap_size"]
embedding_model = config["models"]["embedding"]
persist_dir = config["paths"]["persist_dir"]
collection_name = config["vector_store"]["collection_name"]
extracted_text_dir = config["paths"]["extracted_text_dir"]
chunks_dir = config["paths"]["chunk_dir"]
embeddings_dir = config["paths"]["embeddings_dir"]
bm25_index_dir = config["paths"]["bm25_index_dir"]
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
manifest_path = config["paths"]["manifest_path"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ingestion pipeline. Exactly one of --file or --dir is required to run."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", "-f", help="Path to the file to ingest (.pdf, .txt, .md).")
    group.add_argument(
        "--dir", help="Path to directory to ingest all supported files (.pdf, .txt, .md)"
    )
    parser.add_argument(
        "--force-reingest",
        action="store_true",
        help="Reprocess files even when the manifest says they are unchanged.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan nested directories for .pdf, .txt, .md files. Can only be used if --dir was used.",
    )
    parser.add_argument(
        "--reset-collection",
        action="store_true",
        help="Reset the Chroma collection before ingestion. Bypasses unchanged-file skipping.",
    )
    parser.add_argument(
        "--reset-manifest",
        action="store_true",
        help="Start this run with an empty ingestion manifest.",
    )
    parser.add_argument(
        "--reset-artifacts",
        action="store_true",
        help="Clear generated artifacts for a full local rebuild; typically used with --dir for rebuilding a corpus.",
    )

    args = parser.parse_args()

    if args.recursive and not args.dir:
        parser.error("--recursive can only be used with --dir.")

    return args


def is_supported_file(file_path: str | Path) -> bool:
    if not file_path.is_file():
        return False
    if file_path.name == ".DS_Store":
        return False
    if file_path.name.startswith("."):
        return False

    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def discover_supported_files(directory_path: str | Path, recursive: bool = False) -> list[Path]:
    directory_path = Path(directory_path)

    if not directory_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory_path}")

    if not directory_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {directory_path}")

    if recursive:
        candidates = directory_path.rglob("*")
    else:
        candidates = directory_path.iterdir()

    files = [path for path in candidates if is_supported_file(path)]

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
    client = initialize_vector_db(str(persist_dir))

    if reset:
        collection = reset_collection(client, collection_name)
    else:
        collection = get_or_create_collection(client, collection_name)

    return collection


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _raw_source_dir(project_root: Path = PROJECT_ROOT) -> Path:
    configured_raw_dir = config.get("paths", {}).get("raw_dir")

    if configured_raw_dir:
        raw_dir = Path(configured_raw_dir).expanduser()
        if not raw_dir.is_absolute():
            raw_dir = project_root / raw_dir
    else:
        raw_dir = project_root / "data" / "raw"

    return _resolve_path(raw_dir)


def _is_safe_deletion_target(path: Path, *, project_root: Path, raw_dir: Path) -> bool:
    if not str(path).strip():
        return False

    try:
        resolved_path = _resolve_path(path)
        resolved_project_root = _resolve_path(project_root)
        resolved_project_data_dir = resolved_project_root / "data"
        resolved_raw_dir = _resolve_path(raw_dir)
        home_dir = _resolve_path(Path.home())
    except (OSError, RuntimeError):
        return False

    if resolved_path == Path(resolved_path.anchor):
        return False

    if resolved_path == home_dir:
        return False

    if resolved_path == resolved_project_root:
        return False

    if resolved_path in resolved_project_root.parents:
        return False

    if resolved_project_root in resolved_path.parents:
        if resolved_project_data_dir not in resolved_path.parents:
            return False

    if resolved_path == resolved_raw_dir:
        return False

    if resolved_raw_dir in resolved_path.parents:
        return False

    return True


def _reset_artifact_targets() -> dict[str, Path]:
    return {
        "chunk_dir": _resolve_path(chunks_dir),
        "embeddings_dir": _resolve_path(embeddings_dir),
        "extracted_text_dir": _resolve_path(extracted_text_dir),
        "bm25_index_dir": _resolve_path(bm25_index_dir),
        "manifest_path": _resolve_path(manifest_path),
    }


def _validate_reset_artifact_targets(targets: dict[str, Path]) -> bool:
    project_root = _resolve_path(PROJECT_ROOT)
    raw_dir = _raw_source_dir(project_root)
    unsafe_targets = [
        (name, path)
        for name, path in targets.items()
        if not _is_safe_deletion_target(path, project_root=project_root, raw_dir=raw_dir)
    ]

    if not unsafe_targets:
        return True

    log.error("Unsafe --reset-artifacts target detected. Aborting artifact reset.")
    for name, path in unsafe_targets:
        log.error(f"Unsafe artifact target refused: {name}={path}")

    return False


def _clear_directory_contents(directory_path: str | Path) -> None:
    directory_path = Path(directory_path)

    if not directory_path.exists():
        log.info(f"Generated artifact directory does not exist; skipping: {directory_path}")
        return

    for child in directory_path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
            log.info(f"Deleted generated artifact directory: {child}")
        else:
            child.unlink()
            log.info(f"Deleted generated artifact file: {child}")


def _remove_directory(directory_path: str | Path) -> None:
    directory_path = Path(directory_path)

    if not directory_path.exists():
        log.info(f"Generated artifact directory does not exist; skipping: {directory_path}")
        return

    shutil.rmtree(directory_path)
    log.info(f"Deleted generated artifact directory: {directory_path}")


def _remove_file(file_path: str | Path) -> None:
    file_path = Path(file_path)

    if not file_path.exists():
        log.info(f"Generated artifact file does not exist; skipping: {file_path}")
        return

    if not file_path.is_file():
        log.info(f"Generated artifact path is not a file; skipping: {file_path}")
        return

    file_path.unlink()
    log.info(f"Deleted generated artifact file: {file_path}")


def reset_generated_artifacts() -> bool:
    log.info("Resetting generated ingestion and retrieval artifacts.")
    targets = _reset_artifact_targets()

    if not _validate_reset_artifact_targets(targets):
        return False

    for directory_path in [
        targets["chunk_dir"],
        targets["embeddings_dir"],
        targets["extracted_text_dir"],
    ]:
        _clear_directory_contents(directory_path)

    _remove_directory(targets["bm25_index_dir"])
    _remove_file(targets["manifest_path"])

    return True


def ingest_document(document: str, file_name: str, collection: str):
    log.info(f"Starting Ingestion Pipeline for document: {file_name}")

    if document is None:
        log.warning(f"Document returned None for file: {file_name}")
        return None

    metadata = document.get("metadata", {})
    document_id = metadata.get("document_id", "")

    if not document_id:
        log.warning(f"Document ID missing for file: {file_name}")
        return None
    cleaned_text = document.get("cleaned_text")
    if not cleaned_text:
        log.warning(f"Cleaned text missing or empty for file: {file_name}")

    save_extracted_text(document, extracted_text_dir)

    chunks = chunk_document(
        cleaned_text=cleaned_text,
        document_metadata=metadata,
        target_size=target_size,
        overlap_size=overlap_size,
    )

    if not chunks:
        log.warning(f"Chunks returned empty for file: {file_name}")
        return None

    save_json(chunks, Path(chunks_dir) / f"{document_id}_chunks.json")

    embedded_chunks = embed_chunks(chunk_records=chunks, model_name=embedding_model)

    if not embedded_chunks:
        log.warning("Embedded chunks returned empty. ")
        return None

    save_json(embedded_chunks, Path(embeddings_dir) / f"{document_id}_embeddings.json")

    delete_by_document_id(collection, document_id)
    add_records_to_collection(collection, embedded_chunks)

    vector_count = get_collection_count(collection)

    result = {
        "file_name": metadata.get("file_name", file_name),
        "document_id": document_id,
        "chunks_created": len(chunks),
        "embeddings_created": len(embedded_chunks),
        "vector_db_count": vector_count,
        "status": "success",
        "error": None,
        "metadata": metadata,
    }
    print("\n==== Document Ingestion Summary ====")
    print(f"Document ID: {result['document_id']}")
    print(f"Document: {result['file_name']}")
    print(f"Chunks created: {result['chunks_created']}")
    print(f"Embeddings created: {result['embeddings_created']}")
    print(f"Vector DB count: {result['vector_db_count']}")

    log.info(f"Ingestion completed for document: {file_name}")

    return result


def _failure_metadata(file_path: Path) -> dict[str, Any]:
    metadata = {
        "document_id": None,
        "file_name": file_path.name,
        "file_path": file_path.as_posix(),
        "file_size": None,
    }

    if file_path.exists() and file_path.is_file():
        metadata["file_size"] = file_path.stat().st_size

    return metadata


def _record_manifest_outcome(
    manifest: dict[str, Any],
    file_path: Path,
    file_hash: str | None,
    result: dict[str, Any],
) -> None:
    update_manifest_record(
        manifest=manifest,
        file_path=file_path,
        document_metadata=result.get("metadata") or _failure_metadata(file_path),
        file_hash=file_hash or "",
        chunks_created=result.get("chunks_created", 0),
        status=result.get("status", "failed"),
    )


def process_one_file(
    file_path: str | Path,
    collection,
    manifest: dict[str, Any],
    skip_unchanged: bool = True,
) -> dict[str, Any]:
    file_path = Path(file_path)
    file_hash = None

    try:
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        if not file_path.is_file():
            raise IsADirectoryError(f"Input path is not a file: {file_path}")

        if not is_supported_file(file_path):
            raise ValueError(f"Unsupported file type: {file_path.name}")

        file_hash = compute_file_hash(file_path)

        if skip_unchanged and is_file_unchanged(manifest, file_path, file_hash):
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "document_id": None,
                "chunks_created": 0,
                "embeddings_created": 0,
                "status": "skipped",
                "error": None,
            }

        log.info(f"Ingesting file: {file_path}")

        document, file_name = ingest_file_type(file_path)

        result = ingest_document(
            document=document,
            file_name=file_name,
            collection=collection,
        )

        if result is None:
            result = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "document_id": None,
                "chunks_created": 0,
                "embeddings_created": 0,
                "status": "failed",
                "error": "Ingestion returned None.",
            }
            _record_manifest_outcome(manifest, file_path, file_hash, result)
            return result

        result["file_path"] = str(file_path)
        _record_manifest_outcome(manifest, file_path, file_hash, result)
        return result

    except Exception as e:
        log.exception(f"Failed to ingest file: {file_path}. Reason: {e}")

        result = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "document_id": None,
            "chunks_created": 0,
            "embeddings_created": 0,
            "status": "failed",
            "error": str(e),
        }
        _record_manifest_outcome(manifest, file_path, file_hash, result)
        return result


def print_batch_summary(results: list[dict[str, Any]], collection) -> None:
    processed = [result for result in results if result.get("status") == "success"]
    skipped = [result for result in results if result.get("status") == "skipped"]
    failed = [result for result in results if result.get("status") == "failed"]

    total_chunks = sum(result.get("chunks_created", 0) for result in processed)
    total_embeddings = sum(result.get("embeddings_created", 0) for result in processed)

    vector_count = get_collection_count(collection)

    print("\n==============================")
    print("==== Ingestion Run Summary ====")
    print("==============================")
    print(f"Files attempted: {len(results)}")
    print(f"Processed: {len(processed)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Failed: {len(failed)}")
    print(f"Chunks created: {total_chunks}")
    print(f"Embeddings created: {total_embeddings}")
    print(f"Vector DB count: {vector_count}")

    if skipped:
        print("\nSkipped unchanged files:")
        for result in skipped:
            print(f"- {result.get('file_path')}")

    if failed:
        print("\nFailures:")
        for result in failed:
            print(f"- {result.get('file_path')}: {result.get('error')}")


def main():
    args = parse_args()

    if args.reset_artifacts:
        if args.file:
            log.warning(
                "--reset-artifacts clears generated artifacts for the full corpus; "
                "with --file, the rebuilt corpus will contain only that file."
            )
        if not reset_generated_artifacts():
            return

    if args.reset_manifest or args.reset_artifacts:
        manifest = create_empty_manifest()
    else:
        manifest = load_manifest(manifest_path)

    reset_collection_requested = args.reset_collection or args.reset_artifacts
    skip_unchanged = not (args.force_reingest or reset_collection_requested)
    if reset_collection_requested and not args.force_reingest:
        log.info("Collection reset requested; unchanged-file skip logic is disabled for this run.")

    collection = load_vectordb_collection(reset=reset_collection_requested)

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
            manifest=manifest,
            skip_unchanged=skip_unchanged,
        )
        results.append(result)
        save_manifest(manifest, manifest_path)

    print_batch_summary(results, collection)
    save_manifest(manifest, manifest_path)

    if any(result.get("status") == "success" for result in results):
        get_bm25_index(chunk_dir=chunks_dir, index_dir=bm25_index_dir)

    log.info("Ingestion pipeline run completed.")


if __name__ == "__main__":
    main()
