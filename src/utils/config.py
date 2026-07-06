from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

PATH_KEYS = {
    ("paths", "persist_dir"),
    ("paths", "chunk_dir"),
    ("paths", "extracted_text_dir"),
    ("paths", "embeddings_dir"),
    ("paths", "evaluation_queries"),
    ("paths", "manifest_path"),
    ("paths", "bm25_index_dir"),
}

ENV_OVERRIDES = {
    "RAG_PERSIST_DIR": ("paths", "persist_dir"),
    "RAG_CHUNK_DIR": ("paths", "chunk_dir"),
    "RAG_COLLECTION_NAME": ("vector_store", "collection_name"),
    "RAG_MANIFEST_PATH": ("paths", "manifest_path"),
    "RAG_BM25_INDEX_DIR": ("paths", "bm25_index_dir"),
    "RAG_EMBEDDING_MODEL": ("models", "embedding"),
    "RAG_RERANKER_MODEL": ("models", "reranker"),
}


def _set_nested(config: dict[str, Any], keys: tuple[str, str], value: Any) -> None:
    section, key = keys
    config.setdefault(section, {})[key] = value


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")

    config = deepcopy(loaded)

    for env_name, keys in ENV_OVERRIDES.items():
        value = os.getenv(env_name)
        if value:
            _set_nested(config, keys, value)

    for keys in PATH_KEYS:
        section, key = keys
        value = config.get(section, {}).get(key)
        if value:
            config[section][key] = _resolve_project_path(value)

    return config
