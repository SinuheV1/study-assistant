from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STALE_MODEL_STRINGS = ("all-MiniLM", "llama3.2", "ms-marco")


def import_script_module(script_name: str):
    module_path = PROJECT_ROOT / "scripts" / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(script_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


evaluate_rag = import_script_module("evaluate_rag")


def test_relevant_scripts_import_without_running_pipelines():
    for module_name in [
        "evaluate_rag",
        "run_chunking_ab_test",
        "run_query_pipeline",
        "run_ingestion_pipeline",
    ]:
        import_script_module(module_name)


def test_evaluate_rag_pure_helpers():
    assert (
        evaluate_rag.keyword_score("Linear Regression uses Least Squares", ["linear", "squares"])
        == 1.0
    )
    assert evaluate_rag.keyword_score("No keywords needed", []) == 0.0
    assert evaluate_rag.safe_average([1.0, 3.0]) == 2.0
    assert evaluate_rag.safe_average([]) == 0.0
    assert evaluate_rag.get_query_group("lexical_exact_term") == "lexical"
    assert evaluate_rag.get_query_group("semantic_big_picture") == "semantic"


def test_scripts_do_not_reintroduce_stale_model_strings():
    hits = []

    for path in sorted((PROJECT_ROOT / "scripts").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for stale_model in STALE_MODEL_STRINGS:
            if stale_model in text:
                hits.append((path.name, stale_model))

    assert hits == []


def test_full_ingestion_smoke_uses_throwaway_vector_store():
    text = (PROJECT_ROOT / "scripts" / "test_full_ingestion_vector_workflow.py").read_text(
        encoding="utf-8"
    )

    assert "vector_store_smoke" in text
    assert '"data/processed/vector_store"' not in text
    assert "'data/processed/vector_store'" not in text


def test_chunking_ab_test_keeps_isolated_vector_store():
    text = (PROJECT_ROOT / "scripts" / "run_chunking_ab_test.py").read_text(encoding="utf-8")

    assert "vector_store_ab" in text


def test_chunking_ab_test_does_not_use_production_bm25_cache():
    text = (PROJECT_ROOT / "scripts" / "run_chunking_ab_test.py").read_text(encoding="utf-8")

    assert "bm25_index_dir" not in text
    assert "get_bm25_index" not in text
