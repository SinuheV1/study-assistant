import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.utils.config import ENV_OVERRIDES, PROJECT_ROOT, load_config


def clean_env(overrides=None):
    env = {key: value for key, value in os.environ.items() if key not in ENV_OVERRIDES}
    if overrides:
        env.update(overrides)
    return env


class ConfigTest(unittest.TestCase):
    def test_load_config_defaults(self):
        with patch.dict(os.environ, clean_env(), clear=True):
            config = load_config()

        processed_dir = Path("data") / "processed"
        self.assertEqual(
            config["paths"]["persist_dir"],
            PROJECT_ROOT / processed_dir / "vector_store",
        )
        self.assertEqual(
            config["paths"]["chunk_dir"],
            PROJECT_ROOT / processed_dir / "chunks",
        )
        self.assertEqual(config["models"]["embedding"], "qwen3-embedding:4b")
        self.assertEqual(config["models"]["llm"], "qwen3.6:27b")
        self.assertEqual(config["models"]["reranker"], "mixedbread-ai/mxbai-rerank-base-v1")
        self.assertEqual(config["vector_store"]["collection_name"], "study_assistant_chunks")
        self.assertEqual(config["chunking"]["target_size"], 900)
        self.assertEqual(config["chunking"]["overlap_size"], 75)
        self.assertEqual(config["retrieval"]["top_k"], 4)
        self.assertEqual(config["evaluation"]["top_k"], 3)
        self.assertEqual(config["evaluation"]["candidate_k"], 8)

    def test_load_config_env_overrides(self):
        overrides = {
            "RAG_PERSIST_DIR": "tmp/vector_store",
            "RAG_CHUNK_DIR": "/tmp/chunks",
            "RAG_COLLECTION_NAME": "override_collection",
            "RAG_EMBEDDING_MODEL": "override_embedding",
            "RAG_RERANKER_MODEL": "override_reranker",
        }

        with patch.dict(os.environ, clean_env(overrides), clear=True):
            config = load_config()

        self.assertEqual(config["paths"]["persist_dir"], PROJECT_ROOT / "tmp/vector_store")
        self.assertEqual(config["paths"]["chunk_dir"], Path("/tmp/chunks"))
        self.assertEqual(config["vector_store"]["collection_name"], "override_collection")
        self.assertEqual(config["models"]["embedding"], "override_embedding")
        self.assertEqual(config["models"]["reranker"], "override_reranker")

    def test_load_config_resolves_relative_paths_against_project_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(
                """
paths:
  persist_dir: relative/vector_store
  chunk_dir: /tmp/chunks
  extracted_text_dir: relative/extracted_texts
  embeddings_dir: relative/embeddings
  evaluation_queries: evaluation/queries.json
models:
  embedding: qwen3-embedding:4b
  llm: qwen3.6:27b
  reranker: mixedbread-ai/mxbai-rerank-base-v1
vector_store:
  collection_name: study_assistant_chunks
chunking:
  target_size: 900
  overlap_size: 75
retrieval:
  top_k: 4
  preview_chars: 300
  candidate_k: 12
  bm25_k: 8
  dense_k: 8
  hybrid_alpha: 0.6
""",
                encoding="utf-8",
            )

            with patch.dict(os.environ, clean_env(), clear=True):
                config = load_config(config_path)

        self.assertEqual(config["paths"]["persist_dir"], PROJECT_ROOT / "relative/vector_store")
        self.assertEqual(config["paths"]["chunk_dir"], Path("/tmp/chunks"))
        self.assertEqual(
            config["paths"]["evaluation_queries"],
            PROJECT_ROOT / "evaluation/queries.json",
        )

    def test_load_config_missing_file_error(self):
        missing_path = Path(tempfile.gettempdir()) / "missing-study-assistant-config.yaml"

        with self.assertRaisesRegex(FileNotFoundError, "Config file not found"):
            load_config(missing_path)


if __name__ == "__main__":
    unittest.main()
