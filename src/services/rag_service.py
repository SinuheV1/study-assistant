from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from src.generation.generator import format_page_range
from src.reranking.reranker import rerank_results
from src.retrieval.bm25_retriever import load_chunk_records
from src.retrieval.hybrid_retrieval import hybrid_retrieve
from src.retrieval.retriever import retrieve_relevant_chunks
from src.utils.logging import setup_logger
from src.vector_store.vectordb import get_or_create_collection, initialize_vector_db


log = setup_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PERSIST_DIR = PROJECT_ROOT / "data" / "processed" / "vector_store"
DEFAULT_CHUNK_DIR = PROJECT_ROOT / "data" / "processed" / "chunks"
DEFAULT_COLLECTION_NAME = "study_assistant_chunks"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:4b"
DEFAULT_RERANKER_MODEL = "mixedbread-ai/mxbai-rerank-base-v1"


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def _settings() -> dict[str, Any]:
    return {
        "persist_dir": _env_path("RAG_PERSIST_DIR", DEFAULT_PERSIST_DIR),
        "chunk_dir": _env_path("RAG_CHUNK_DIR", DEFAULT_CHUNK_DIR),
        "collection_name": os.getenv(
            "RAG_COLLECTION_NAME",
            DEFAULT_COLLECTION_NAME,
        ),
        "embedding_model": os.getenv(
            "RAG_EMBEDDING_MODEL",
            DEFAULT_EMBEDDING_MODEL,
        ),
        "reranker_model": os.getenv(
            "RAG_RERANKER_MODEL",
            DEFAULT_RERANKER_MODEL,
        ),
    }


def _load_collection():
    settings = _settings()
    client = initialize_vector_db(str(settings["persist_dir"]))
    return get_or_create_collection(client, settings["collection_name"])


def _infer_week(metadata: dict[str, Any]) -> str | None:
    week = metadata.get("week")
    if week:
        return str(week)

    file_path = str(metadata.get("file_path") or "")
    if not file_path:
        return None

    for part in Path(file_path).parts:
        normalized = part.strip().lower().replace("-", "_")
        if re.fullmatch(r"week_?\d+", normalized):
            digits = re.search(r"\d+", normalized)
            if digits:
                return f"week_{int(digits.group()):02d}"
            return normalized

    return None


def _matches_filters(
    metadata: dict[str, Any],
    course: str | None = None,
    week: str | None = None,
) -> bool:
    if course and str(metadata.get("course") or "").lower() != course.lower():
        return False

    if week:
        inferred_week = _infer_week(metadata)
        if not inferred_week or inferred_week.lower() != week.lower():
            return False

    return True


def _metadata_source_id(metadata: dict[str, Any]) -> str:
    return (
        metadata.get("document_id")
        or metadata.get("file_path")
        or metadata.get("file_name")
        or metadata.get("title")
        or "unknown_source"
    )


def _iter_chroma_metadatas(batch_size: int = 1000) -> list[dict[str, Any]]:
    collection = _load_collection()
    total = collection.count()
    metadatas: list[dict[str, Any]] = []

    for offset in range(0, total, batch_size):
        batch = collection.get(
            include=["metadatas"],
            limit=batch_size,
            offset=offset,
        )
        metadatas.extend(batch.get("metadatas") or [])

    return metadatas


def _iter_chunk_file_records() -> list[dict[str, Any]]:
    settings = _settings()
    chunk_dir = settings["chunk_dir"]

    if not chunk_dir.exists():
        return []

    records: list[dict[str, Any]] = []

    for file_path in sorted(chunk_dir.glob("*.json")):
        if not file_path.is_file():
            continue

        try:
            with file_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            log.warning(f"Could not read chunk artifact {file_path}: {exc}")
            continue

        if isinstance(data, list):
            records.extend(record for record in data if isinstance(record, dict))

    return records


def _document_records_from_metadatas(
    metadatas: list[dict[str, Any]],
    course: str | None = None,
    week: str | None = None,
) -> list[dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}

    for metadata in metadatas:
        if not isinstance(metadata, dict):
            continue

        if not _matches_filters(metadata, course=course, week=week):
            continue

        source_id = _metadata_source_id(metadata)
        document = documents.setdefault(
            source_id,
            {
                "source_id": source_id,
                "title": metadata.get("title"),
                "course": metadata.get("course"),
                "week": _infer_week(metadata),
                "source_type": metadata.get("source_type"),
                "source_path": metadata.get("file_path"),
                "file_name": metadata.get("file_name"),
                "chunk_count": 0,
            },
        )
        document["chunk_count"] += 1

    return sorted(
        documents.values(),
        key=lambda item: (
            str(item.get("course") or ""),
            str(item.get("week") or ""),
            str(item.get("title") or item.get("file_name") or ""),
        ),
    )


def _document_records_from_chunks(
    course: str | None = None,
    week: str | None = None,
) -> list[dict[str, Any]]:
    metadatas = []

    for record in _iter_chunk_file_records():
        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            metadatas.append(metadata)

    return _document_records_from_metadatas(
        metadatas=metadatas,
        course=course,
        week=week,
    )


def health_check_service() -> dict[str, Any]:
    settings = _settings()
    response = {
        "status": "ok",
        "vector_db_path": str(settings["persist_dir"]),
        "embedding_model": settings["embedding_model"],
        "collection_name": settings["collection_name"],
        "warnings": [],
    }

    try:
        collection = _load_collection()
        response["total_chunks"] = collection.count()
    except Exception as exc:
        response["status"] = "error"
        response["warnings"].append(f"Vector database unavailable: {exc}")

    return response


def collection_stats_service() -> dict[str, Any]:
    settings = _settings()
    metadatas: list[dict[str, Any]] = []
    warnings: list[str] = []
    total_chunks = 0

    try:
        collection = _load_collection()
        total_chunks = collection.count()
        metadatas = _iter_chroma_metadatas()
    except Exception as exc:
        warnings.append(f"Could not read Chroma metadata: {exc}")
        chunk_records = _iter_chunk_file_records()
        metadatas = [
            record.get("metadata", {})
            for record in chunk_records
            if isinstance(record.get("metadata"), dict)
        ]
        total_chunks = len(chunk_records)

    documents = _document_records_from_metadatas(metadatas)
    courses = sorted(
        {
            str(metadata.get("course"))
            for metadata in metadatas
            if metadata.get("course")
        }
    )
    weeks = sorted(
        {
            week
            for metadata in metadatas
            for week in [_infer_week(metadata)]
            if week
        }
    )

    return {
        "collection_name": settings["collection_name"],
        "total_chunks": total_chunks,
        "courses": courses,
        "weeks": weeks,
        "source_count": len(documents),
        "warnings": warnings,
    }


def list_indexed_documents_service(
    course: str | None = None,
    week: str | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []

    try:
        documents = _document_records_from_metadatas(
            metadatas=_iter_chroma_metadatas(),
            course=course,
            week=week,
        )
    except Exception as exc:
        warnings.append(f"Could not read Chroma metadata: {exc}")
        documents = _document_records_from_chunks(course=course, week=week)

    return {
        "documents": documents,
        "warnings": warnings,
    }


def _score_for_result(result: dict[str, Any]) -> float | None:
    for key in ("rerank_score", "hybrid_score", "similarity", "bm25_score"):
        value = result.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

    return None


def _format_search_result(result: dict[str, Any]) -> dict[str, Any]:
    metadata = result.get("metadata") or {}
    page_start = metadata.get("page_start")
    page_end = metadata.get("page_end")

    if page_start in [None, ""] and page_end in [None, ""]:
        page = None
    elif page_start == page_end or page_end in [None, ""]:
        page = page_start
    else:
        page = format_page_range(page_start, page_end)

    return {
        "rank": result.get("rank"),
        "chunk_id": result.get("chunk_id") or metadata.get("chunk_id"),
        "chunk_text": result.get("chunk_text", ""),
        "source": metadata.get("file_name") or metadata.get("title"),
        "source_path": metadata.get("file_path"),
        "source_title": metadata.get("title"),
        "source_type": metadata.get("source_type"),
        "course": metadata.get("course"),
        "week": _infer_week(metadata),
        "section": metadata.get("section"),
        "page": page,
        "page_start": page_start,
        "page_end": page_end,
        "score": _score_for_result(result),
        "metadata": metadata,
    }


def _filter_results(
    results: list[dict[str, Any]],
    course: str | None = None,
    week: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    filtered = [
        result
        for result in results
        if _matches_filters(result.get("metadata") or {}, course=course, week=week)
    ]

    for index, result in enumerate(filtered, start=1):
        result["rank"] = index

    if limit is not None:
        return filtered[:limit]

    return filtered


def search_notes_service(
    query: str,
    course: str | None = None,
    week: str | None = None,
    top_k: int = 8,
    use_hybrid: bool = True,
    use_reranker: bool = True,
) -> dict[str, Any]:
    if not query or not query.strip():
        return {
            "query": query,
            "results": [],
            "warnings": ["Query is empty."],
        }

    settings = _settings()
    warnings: list[str] = []
    safe_top_k = max(1, min(int(top_k), 50))
    has_filters = bool(course or week)
    retrieval_top_k = safe_top_k * 5 if has_filters else safe_top_k

    try:
        collection = _load_collection()
        collection_count = collection.count()
    except Exception as exc:
        return {
            "query": query,
            "results": [],
            "warnings": [f"Vector database unavailable: {exc}"],
        }

    if collection_count == 0:
        return {
            "query": query,
            "results": [],
            "warnings": ["Collection is empty."],
        }

    retrieval_top_k = min(retrieval_top_k, collection_count)

    try:
        if use_hybrid:
            chunk_records = load_chunk_records(str(settings["chunk_dir"])) or []
            if course or week:
                chunk_records = [
                    record
                    for record in chunk_records
                    if _matches_filters(
                        record.get("metadata") or {},
                        course=course,
                        week=week,
                    )
                ]

            if not chunk_records:
                warnings.append(
                    "No chunk artifacts matched filters; falling back to dense retrieval."
                )
                results = retrieve_relevant_chunks(
                    query=query,
                    collection=collection,
                    model_name=settings["embedding_model"],
                    top_k=retrieval_top_k,
                )
            else:
                results = hybrid_retrieve(
                    query=query,
                    collection=collection,
                    chunk_records=chunk_records,
                    embedding_model=settings["embedding_model"],
                    dense_k=retrieval_top_k,
                    bm25_k=retrieval_top_k,
                    top_k=retrieval_top_k,
                    alpha=0.6,
                )
        else:
            results = retrieve_relevant_chunks(
                query=query,
                collection=collection,
                model_name=settings["embedding_model"],
                top_k=retrieval_top_k,
            )
    except Exception as exc:
        return {
            "query": query,
            "results": [],
            "warnings": [f"Retrieval failed: {exc}"],
        }

    results = _filter_results(
        results,
        course=course,
        week=week,
        limit=None if use_reranker else safe_top_k,
    )

    if use_reranker and results:
        try:
            results = rerank_results(
                query=query,
                retrieved_results=results,
                model_name=settings["reranker_model"],
                top_k=safe_top_k,
            )
        except Exception as exc:
            warnings.append(f"Reranker failed; returned retrieval results: {exc}")
            results = results[:safe_top_k]

    return {
        "query": query,
        "results": [_format_search_result(result) for result in results[:safe_top_k]],
        "warnings": warnings,
    }

