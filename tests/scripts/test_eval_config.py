from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

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


class FakeBM25Index:
    records = [{"chunk_id": "chunk_bm25", "chunk_text": "linear squares", "metadata": {}}]


class FakeCompletedProcess:
    def __init__(self, stdout: str = "", returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode


def fake_git_run(
    status_stdout: str = "", status_returncode: int = 0, raise_on_status: bool = False
):
    def run(args, **kwargs):
        git_args = args[1:]

        if git_args == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return FakeCompletedProcess(stdout="main\n")
        if git_args == ["rev-parse", "HEAD"]:
            return FakeCompletedProcess(stdout="abc123\n")
        if git_args == ["status", "--short"]:
            if raise_on_status:
                raise RuntimeError("git status unavailable")
            return FakeCompletedProcess(stdout=status_stdout, returncode=status_returncode)

        raise AssertionError(f"Unexpected git command: {args}")

    return run


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


def test_evaluate_rag_help_works(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["evaluate_rag.py", "--help"])

    with pytest.raises(SystemExit):
        evaluate_rag.parse_args()

    output = capsys.readouterr().out
    assert "--mode" in output
    assert "--limit" in output
    assert "--group" in output
    assert "--pipelines" in output
    assert "--save-results" in output
    assert "--no-save-results" in output
    assert "--results-dir" in output
    assert "--run-name" in output


def test_evaluate_rag_default_args_are_full_mode(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate_rag.py"])

    args = evaluate_rag.parse_args()

    assert args.mode == "full"
    assert args.limit is None
    assert args.group is None
    assert args.pipelines is None
    assert args.save_results is True


def _patch_fast_evaluation(monkeypatch, tmp_path, queries: list[dict] | None = None):
    queries = queries or [
        {
            "id": "semantic_test",
            "query": "What is least squares?",
            "expected_keywords": ["linear", "squares"],
        }
    ]
    queries_path = tmp_path / "queries.json"
    queries_path.write_text(
        json.dumps({"queries": queries}),
        encoding="utf-8",
    )
    result = {
        "rank": 1,
        "chunk_id": "chunk_linear",
        "chunk_text": "linear squares",
        "metadata": {
            "file_name": "lecture.pdf",
            "page_start": 7,
            "page_end": 8,
            "section": "Least Squares",
            "source_type": "lecture_pdfs",
        },
        "similarity": 0.9,
    }
    calls = {
        "retrieve": [],
        "rerank": [],
        "hybrid": [],
        "generate": [],
    }

    def fake_retrieve(**kwargs):
        calls["retrieve"].append(kwargs)
        return [result]

    def fake_rerank(**kwargs):
        calls["rerank"].append(kwargs)
        return [result]

    def fake_hybrid(**kwargs):
        calls["hybrid"].append(kwargs)
        return [result]

    def fake_generate(*args, **kwargs):
        calls["generate"].append((args, kwargs))
        return "linear squares"

    monkeypatch.setattr(evaluate_rag, "evaluation_queries_path", queries_path)
    monkeypatch.setattr(evaluate_rag, "eval_results_directory", tmp_path / "configured_results")
    monkeypatch.setattr(evaluate_rag, "initialize_vector_db", lambda path: object())
    monkeypatch.setattr(evaluate_rag, "get_or_create_collection", lambda client, name: object())
    monkeypatch.setattr(evaluate_rag, "get_bm25_index", lambda **kwargs: FakeBM25Index())
    monkeypatch.setattr(evaluate_rag, "load_chunk_records", lambda path: FakeBM25Index.records)
    monkeypatch.setattr(evaluate_rag, "retrieve_relevant_chunks", fake_retrieve)
    monkeypatch.setattr(evaluate_rag, "rerank_results", fake_rerank)
    monkeypatch.setattr(evaluate_rag, "hybrid_retrieve", fake_hybrid)
    monkeypatch.setattr(evaluate_rag, "generate_answer", fake_generate)
    monkeypatch.setattr(
        evaluate_rag,
        "get_git_metadata",
        lambda: {"branch": "test-branch", "commit": "abc123", "dirty": False},
    )
    return calls


def _mixed_queries():
    return [
        {
            "id": "semantic_first",
            "query": "What is least squares?",
            "expected_keywords": ["linear", "squares"],
        },
        {
            "id": "lexical_exact_term",
            "query": "Define RSS.",
            "expected_keywords": ["linear", "squares"],
        },
        {
            "id": "semantic_second",
            "query": "What is supervised learning?",
            "expected_keywords": ["linear", "squares"],
        },
    ]


def test_default_run_saves_artifact_to_configured_results_dir(tmp_path, monkeypatch):
    _patch_fast_evaluation(monkeypatch, tmp_path)

    artifact = evaluate_rag.run_evaluation()
    files = list((tmp_path / "configured_results").glob("*_eval_run.json"))

    assert len(files) == 1
    saved = json.loads(files[0].read_text(encoding="utf-8"))
    assert artifact["run_id"] == saved["run_id"]
    assert saved["schema_version"] == "1.1"
    assert saved["git"] == {"branch": "test-branch", "commit": "abc123", "dirty": False}
    assert saved["run_options"] == {
        "mode": "full",
        "limit": None,
        "group": None,
        "pipelines": ["dense", "dense_reranker", "hybrid", "hybrid_reranker"],
    }
    assert saved["timing"]["started_at"]
    assert saved["timing"]["finished_at"]
    assert saved["timing"]["total_seconds"] >= 0.0
    assert saved["config"]["embedding_model"] == "qwen3-embedding:4b"
    assert saved["config"]["generation_model"] == "qwen3.6:27b"
    assert saved["config"]["reranker_model"] == "mixedbread-ai/mxbai-rerank-base-v1"
    assert saved["summary"]["all"]["query_count"] == 1
    assert saved["summary"]["groups"]["semantic"]["query_count"] == 1
    assert saved["queries"][0]["pipeline"] == "Dense Baseline"
    assert saved["queries"][0]["generation_score"] == 1.0
    assert saved["queries"][0]["retrieval_seconds"] >= 0.0
    assert saved["queries"][0]["generation_seconds"] >= 0.0
    assert saved["queries"][0]["total_seconds"] >= 0.0
    assert saved["queries"][0]["top_chunk_ids"] == ["chunk_linear"]
    assert saved["queries"][0]["top_sources"][0]["file_name"] == "lecture.pdf"


def test_retrieval_mode_skips_generation_and_writes_null_generation_scores(tmp_path, monkeypatch):
    _patch_fast_evaluation(monkeypatch, tmp_path)

    def fail_generate(*args, **kwargs):
        raise AssertionError("generation should not run in retrieval mode")

    monkeypatch.setattr(evaluate_rag, "generate_answer", fail_generate)

    artifact = evaluate_rag.run_evaluation(mode="retrieval", save_results=False)

    assert artifact["run_options"]["mode"] == "retrieval"
    assert {record["generation_score"] for record in artifact["queries"]} == {None}
    assert {record["generation_seconds"] for record in artifact["queries"]} == {0.0}
    assert artifact["summary"]["all"]["pipelines"]["Dense Baseline"]["generation"] is None
    assert artifact["timing"]["pipelines"]["Dense Baseline"]["generation_seconds"] == 0.0


def test_full_mode_calls_generation(tmp_path, monkeypatch):
    calls = _patch_fast_evaluation(monkeypatch, tmp_path)

    artifact = evaluate_rag.run_evaluation(mode="full", save_results=False)

    assert artifact["run_options"]["mode"] == "full"
    assert len(calls["generate"]) == 4
    assert artifact["queries"][0]["generation_score"] == 1.0
    assert artifact["queries"][0]["generation_seconds"] >= 0.0


def test_limit_filters_queries_after_group_filtering(tmp_path, monkeypatch):
    _patch_fast_evaluation(monkeypatch, tmp_path, queries=_mixed_queries())

    artifact = evaluate_rag.run_evaluation(limit=2, save_results=False)

    assert artifact["run_options"]["limit"] == 2
    assert {record["query_id"] for record in artifact["queries"]} == {
        "semantic_first",
        "lexical_exact_term",
    }
    assert artifact["summary"]["all"]["query_count"] == 2


def test_group_semantic_filters_queries(tmp_path, monkeypatch):
    _patch_fast_evaluation(monkeypatch, tmp_path, queries=_mixed_queries())

    artifact = evaluate_rag.run_evaluation(group="semantic", save_results=False)

    assert artifact["run_options"]["group"] == "semantic"
    assert {record["query_id"] for record in artifact["queries"]} == {
        "semantic_first",
        "semantic_second",
    }
    assert {record["group"] for record in artifact["queries"]} == {"semantic"}
    assert artifact["summary"]["all"]["query_count"] == 2


def test_group_lexical_filters_queries(tmp_path, monkeypatch):
    _patch_fast_evaluation(monkeypatch, tmp_path, queries=_mixed_queries())

    artifact = evaluate_rag.run_evaluation(group="lexical", save_results=False)

    assert artifact["run_options"]["group"] == "lexical"
    assert {record["query_id"] for record in artifact["queries"]} == {"lexical_exact_term"}
    assert {record["group"] for record in artifact["queries"]} == {"lexical"}
    assert artifact["summary"]["all"]["query_count"] == 1


def test_single_pipeline_filter_runs_only_hybrid(tmp_path, monkeypatch):
    calls = _patch_fast_evaluation(monkeypatch, tmp_path)

    artifact = evaluate_rag.run_evaluation(pipelines="hybrid", save_results=False)

    assert artifact["run_options"]["pipelines"] == ["hybrid"]
    assert artifact["pipelines"] == ["Hybrid"]
    assert {record["pipeline"] for record in artifact["queries"]} == {"Hybrid"}
    assert calls["retrieve"] == []
    assert calls["rerank"] == []
    assert len(calls["hybrid"]) == 1


def test_multiple_pipeline_filter_runs_only_requested_pipelines(tmp_path, monkeypatch):
    calls = _patch_fast_evaluation(monkeypatch, tmp_path)

    artifact = evaluate_rag.run_evaluation(
        mode="retrieval",
        pipelines="hybrid,hybrid_reranker",
        save_results=False,
    )

    assert artifact["run_options"]["pipelines"] == ["hybrid", "hybrid_reranker"]
    assert artifact["pipelines"] == ["Hybrid", "Hybrid + Reranker"]
    assert {record["pipeline"] for record in artifact["queries"]} == {
        "Hybrid",
        "Hybrid + Reranker",
    }
    assert calls["retrieve"] == []
    assert len(calls["hybrid"]) == 2
    assert len(calls["rerank"]) == 1
    assert calls["generate"] == []


def test_invalid_pipeline_key_fails_clearly():
    with pytest.raises(Exception, match="Invalid pipeline key"):
        evaluate_rag.parse_pipeline_keys("hybrid,unknown")


def test_timing_metadata_includes_run_pipeline_and_query_levels(tmp_path, monkeypatch):
    _patch_fast_evaluation(monkeypatch, tmp_path)

    artifact = evaluate_rag.run_evaluation(mode="retrieval", pipelines="hybrid", save_results=False)

    assert artifact["timing"]["started_at"]
    assert artifact["timing"]["finished_at"]
    assert artifact["timing"]["total_seconds"] >= 0.0
    assert artifact["timing"]["pipelines"]["Hybrid"]["query_count"] == 1
    assert artifact["timing"]["pipelines"]["Hybrid"]["retrieval_seconds"] >= 0.0
    assert artifact["timing"]["pipelines"]["Hybrid"]["generation_seconds"] == 0.0
    assert artifact["timing"]["pipelines"]["Hybrid"]["total_seconds"] >= 0.0
    assert artifact["queries"][0]["retrieval_seconds"] >= 0.0
    assert artifact["queries"][0]["generation_seconds"] == 0.0
    assert artifact["queries"][0]["total_seconds"] >= 0.0


def test_no_save_results_does_not_write_artifact(tmp_path, monkeypatch):
    _patch_fast_evaluation(monkeypatch, tmp_path)

    evaluate_rag.run_evaluation(save_results=False)

    assert not (tmp_path / "configured_results").exists()


def test_results_dir_override_writes_to_override_dir(tmp_path, monkeypatch):
    _patch_fast_evaluation(monkeypatch, tmp_path)
    override_dir = tmp_path / "override_results"

    artifact = evaluate_rag.run_evaluation(results_dir=override_dir)

    files = list(override_dir.glob("*_eval_run.json"))
    assert len(files) == 1
    assert artifact["config"]["paths"]["eval_results_dir"] == str(override_dir)
    assert not (tmp_path / "configured_results").exists()


def test_run_name_is_slugged_into_filename(tmp_path, monkeypatch):
    _patch_fast_evaluation(monkeypatch, tmp_path)

    artifact = evaluate_rag.run_evaluation(run_name="ISLP Ch2!!")
    files = list((tmp_path / "configured_results").glob("*_islp-ch2_eval_run.json"))

    assert len(files) == 1
    assert artifact["run_id"].endswith("_islp-ch2")


def test_git_metadata_clean_tree_sets_dirty_false(monkeypatch):
    monkeypatch.setattr(evaluate_rag.subprocess, "run", fake_git_run(status_stdout=""))

    assert evaluate_rag.get_git_metadata() == {
        "branch": "main",
        "commit": "abc123",
        "dirty": False,
    }


def test_git_metadata_dirty_tree_sets_dirty_true(monkeypatch):
    monkeypatch.setattr(evaluate_rag.subprocess, "run", fake_git_run(status_stdout=" M file.py\n"))

    assert evaluate_rag.get_git_metadata() == {
        "branch": "main",
        "commit": "abc123",
        "dirty": True,
    }


def test_git_metadata_status_failure_sets_dirty_none(monkeypatch):
    monkeypatch.setattr(
        evaluate_rag.subprocess,
        "run",
        fake_git_run(status_stdout="fatal: not a git repository\n", status_returncode=128),
    )

    assert evaluate_rag.get_git_metadata() == {
        "branch": "main",
        "commit": "abc123",
        "dirty": None,
    }


def test_git_metadata_status_exception_sets_dirty_none(monkeypatch):
    monkeypatch.setattr(evaluate_rag.subprocess, "run", fake_git_run(raise_on_status=True))

    assert evaluate_rag.get_git_metadata() == {
        "branch": "main",
        "commit": "abc123",
        "dirty": None,
    }


def test_git_metadata_failure_does_not_crash(monkeypatch):
    def fail_run(*args, **kwargs):
        raise RuntimeError("git unavailable")

    monkeypatch.setattr(evaluate_rag.subprocess, "run", fail_run)

    assert evaluate_rag.get_git_metadata() == {
        "branch": "unknown",
        "commit": "unknown",
        "dirty": None,
    }
