import hashlib
import json
from datetime import datetime
from pathlib import Path

from src.utils.logging import setup_logger

log = setup_logger(__name__)


# helper functions for time stamp and creating empty manifests
def current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_empty_manifest() -> dict:
    return {
        "version": 1,
        "updated_at": current_timestamp(),
        "documents": {},
    }


def load_manifest(manifest_path: str | Path) -> dict:
    file_path = Path(manifest_path)
    if not file_path.exists():
        log.info(f"Manifest Path not found: {file_path}. Returning empty manifest.")
        return create_empty_manifest()

    try:
        with file_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
            log.info(f"Manifest file found at {file_path}. Returning file.")
        if "documents" not in manifest:
            manifest["documents"] = {}
        log.info(f"Manifest file loaded from {manifest_path}.")
        return manifest

    except json.decoder.JSONDecodeError as e:
        log.warning(f"Manifest JSON decode error: {e}. Returning empty manifest.")
        return create_empty_manifest()


def save_manifest(manifest: dict, manifest_path: str | Path) -> None:
    file_path = Path(manifest_path)
    if not manifest:
        log.warning("Manifest is empty or None. Nothing saved.")
        return
    manifest["updated_at"] = current_timestamp()
    manifest.setdefault("version", 1)
    manifest.setdefault("documents", {})
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    log.info(f"Manifest saved to {file_path}.")


def compute_file_hash(file_path: str | Path) -> str | None:
    file_path = Path(file_path)

    if not file_path.exists():
        log.info(f"File Path not found: {file_path}. ")
        return None
    if not file_path.is_file():
        log.info(f"Path is not a file: {file_path}. ")
        return None

    hash_object = hashlib.sha256()

    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hash_object.update(chunk)

    return hash_object.hexdigest()


def normalize_manifest_path(file_path: str | Path) -> str:
    """
    Normalize a file path before using it as a manifest key.
    """
    return Path(file_path).as_posix()


def get_manifest_record(manifest: dict, file_path: str | Path) -> dict | None:
    """
    Return the manifest record for a file if it exists.
    """

    if not manifest:
        return None

    documents = manifest.get("documents", {})
    normalized_path = normalize_manifest_path(file_path)
    return documents.get(normalized_path)


def is_file_unchanged(manifest: dict, file_path: str | Path, current_file_hash: str | None) -> bool:
    """
    Return True if the manifest has a successful record for this file
    and the stored hash matches the current file hash.
    """

    if not current_file_hash:
        return False

    existing_record = get_manifest_record(manifest, file_path)

    if not existing_record:
        log.info(f"No existing manifest record for {file_path}. ")
        return False

    previous_hash = existing_record.get("file_hash")
    previous_status = existing_record.get("status")

    if previous_hash == current_file_hash and previous_status == "success":
        log.info(f"File unchanged since last successful ingestion: {file_path}")
        return True

    log.info(f"File changed or previous ingestion was not successful: {file_path}")
    return False


def update_manifest_record(
    manifest: dict,
    file_path: str | Path,
    document_metadata: dict,
    file_hash: str,
    chunks_created: int,
    status: str,
) -> dict:
    """
    Add or update one document record in the ingestion manifest.
    """
    manifest.setdefault("version", 1)
    manifest.setdefault("documents", {})

    normalized_path = normalize_manifest_path(file_path)

    record = {
        "document_id": document_metadata.get("document_id"),
        "file_name": document_metadata.get("file_name"),
        "file_path": document_metadata.get("file_path", str(file_path)),
        "file_hash": file_hash,
        "file_size": document_metadata.get("file_size"),
        "source_type": document_metadata.get("source_type"),
        "raw_source_type": document_metadata.get("raw_source_type"),
        "course": document_metadata.get("course"),
        "title": document_metadata.get("title"),
        "chunks_created": chunks_created,
        "ingested_at": current_timestamp(),
        "status": status,
    }

    manifest["documents"][normalized_path] = record
    log.info(
        f"Manifest updated for {normalized_path} | status={status} | chunks_created={chunks_created}"
    )

    return manifest
