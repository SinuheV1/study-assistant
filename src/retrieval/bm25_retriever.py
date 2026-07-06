import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bm25s

from src.utils.logging import setup_logger

log = setup_logger(__name__)

BM25_METADATA_FILE = "metadata.json"


@dataclass(frozen=True)
class BM25IndexBundle:
    retriever: Any | None
    records: list[dict]
    fingerprint: str | None
    metadata: dict[str, Any]
    index_dir: Path | None = None


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


def _bm25s_version() -> str:
    return str(getattr(bm25s, "__version__", "unknown"))


def compute_bm25_fingerprint(chunk_dir: str | Path) -> dict[str, Any]:
    chunk_dir = Path(chunk_dir)
    files = []

    if chunk_dir.exists():
        for file_path in sorted(chunk_dir.glob("*.json")):
            if not file_path.is_file():
                continue

            stat = file_path.stat()
            files.append(
                {
                    "path": file_path.name,
                    "mtime_ns": stat.st_mtime_ns,
                    "size": stat.st_size,
                }
            )

    fingerprint_source = {
        "bm25s_version": _bm25s_version(),
        "files": files,
    }
    fingerprint_text = json.dumps(fingerprint_source, sort_keys=True, separators=(",", ":"))

    return {
        **fingerprint_source,
        "fingerprint": _hash_fingerprint(fingerprint_text),
    }


def _hash_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def save_bm25_index(
    retriever,
    chunk_records: list[dict],
    index_dir: str | Path,
    fingerprint_metadata: dict[str, Any],
) -> BM25IndexBundle:
    index_dir = Path(index_dir)
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=".bm25_index_", dir=index_dir.parent))

    try:
        retriever.save(temp_dir, show_progress=False)

        metadata = {
            "fingerprint": fingerprint_metadata.get("fingerprint"),
            "bm25s_version": fingerprint_metadata.get("bm25s_version"),
            "files": fingerprint_metadata.get("files", []),
            "record_count": len(chunk_records),
            "records": chunk_records,
        }

        with (temp_dir / BM25_METADATA_FILE).open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)

        if index_dir.exists():
            shutil.rmtree(index_dir)

        temp_dir.rename(index_dir)

        log.info(f"Saved BM25 index to {index_dir}")
        return BM25IndexBundle(
            retriever=retriever,
            records=chunk_records,
            fingerprint=metadata["fingerprint"],
            metadata=metadata,
            index_dir=index_dir,
        )
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def build_and_save_bm25_index(
    chunk_records: list[dict],
    index_dir: str | Path,
    fingerprint_metadata: dict[str, Any] | None = None,
) -> BM25IndexBundle:
    if not chunk_records:
        log.warning("No chunk records provided. BM25 index artifact was not created.")
        return BM25IndexBundle(
            retriever=None,
            records=[],
            fingerprint=None,
            metadata={},
            index_dir=Path(index_dir),
        )

    index_dir = Path(index_dir)
    fingerprint_metadata = fingerprint_metadata or {
        "fingerprint": None,
        "bm25s_version": _bm25s_version(),
        "files": [],
    }
    retriever = build_bm25_index(chunk_records)

    return save_bm25_index(
        retriever=retriever,
        chunk_records=chunk_records,
        index_dir=index_dir,
        fingerprint_metadata=fingerprint_metadata,
    )


def load_bm25_index(index_dir: str | Path) -> BM25IndexBundle:
    index_dir = Path(index_dir)
    metadata_path = index_dir / BM25_METADATA_FILE

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    records = metadata.get("records")
    if not isinstance(records, list):
        raise ValueError(f"BM25 metadata is missing records: {metadata_path}")

    retriever = bm25s.BM25.load(index_dir, load_corpus=False)

    return BM25IndexBundle(
        retriever=retriever,
        records=records,
        fingerprint=metadata.get("fingerprint"),
        metadata=metadata,
        index_dir=index_dir,
    )


def get_bm25_index(chunk_dir: str | Path, index_dir: str | Path) -> BM25IndexBundle:
    fingerprint_metadata = compute_bm25_fingerprint(chunk_dir)
    expected_fingerprint = fingerprint_metadata["fingerprint"]
    index_dir = Path(index_dir)

    try:
        bundle = load_bm25_index(index_dir)
        if bundle.fingerprint == expected_fingerprint:
            log.info(f"Loaded fresh BM25 index from {index_dir}")
            return bundle

        log.info("BM25 index fingerprint is stale. Rebuilding index.")
    except Exception as exc:
        log.warning(f"Could not load BM25 index from {index_dir}: {exc}. Rebuilding index.")

    chunk_records = load_chunk_records(str(chunk_dir)) or []
    if not chunk_records:
        return BM25IndexBundle(
            retriever=None,
            records=[],
            fingerprint=expected_fingerprint,
            metadata=fingerprint_metadata,
            index_dir=index_dir,
        )

    return build_and_save_bm25_index(
        chunk_records=chunk_records,
        index_dir=index_dir,
        fingerprint_metadata=fingerprint_metadata,
    )


def bm25_retrieve_from_index(query: str, index_bundle: BM25IndexBundle, top_k: int) -> list[str]:
    if not index_bundle.records or index_bundle.retriever is None:
        log.warning("No BM25 index records provided to BM25 retriever.")
        return []

    tokenized_query = bm25s.tokenize(query)
    results, scores = index_bundle.retriever.retrieve(tokenized_query, k=top_k)
    bm25_results = []

    for rank, (chunk_index, score) in enumerate(zip(results[0], scores[0]), start=1):
        chunk_record = index_bundle.records[int(chunk_index)]
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
