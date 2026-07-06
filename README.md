# 📘 Local RAG-Based Study Assistant

A **local-first Retrieval-Augmented Generation (RAG) system** for querying unstructured documents using vector search and local LLMs, with built-in evaluation and chunking A/B testing.

---

## 🚀 Overview

This project implements an end-to-end RAG pipeline that:

- Ingests unstructured PDF, Markdown, and text documents
- Converts PDFs to structured Markdown using Docling
- Cleans and normalizes extracted text
- Chunks documents into semantically meaningful segments
- Generates embeddings for vector similarity search
- Supports section-aware metadata retrieval and debugging
- Retrieves relevant context with metadata-aware source tracing
- Generates grounded answers using a local LLM (Ollama)
- Evaluates retrieval and generation quality
- Optimizes chunking strategies through A/B testing
- Supports CLI-based querying and retrieval debugging
- Supports dense, BM25, hybrid, and reranked retrieval modes
- Combines semantic vector search with lexical BM25 search for hybrid retrieval
- Evaluates retrieval performance across semantic and lexical query groups
- Exposes read-only local MCP tools for agent-driven retrieval debugging

---

## 🧠 System Architecture

```text
PDF / TXT / MD
        ↓
 Ingestion Layer
(Docling or Text Parser)
        ↓
   Text Cleaning
        ↓
     Chunking
        ↓
    Embeddings
        ↓
    Chroma Vector DB
        ↓
    User Query
        ↓
    Dense Vector Retrieval
        +
    BM25 Lexical Retrieval
        ↓
    Optional Hybrid Score Fusion
        ↓
    Optional Cross-Encoder Reranking
        ↓
    Context Builder
        ↓
    LLM Generation
        ↓
    Answer + Sources
```

---

## ⚙️ Tech Stack

- **Language:** Python 
- **LLM Runtime:** Ollama 
- **Generation Model:** `qwen3.6:27b` 
- **Embedding Model:** `qwen3-embedding:4b` 
- **Vector Database:** ChromaDB 
- **PDF Parsing:** Docling 
- **Keyword Retrieval:** BM25 via `bm25s` 
- **Hybrid Retrieval:** Dense + BM25 score fusion 
- **Reranking:** Cross-encoder reranker (`mixedbread-ai/mxbai-rerank-base-v1`) 
- **Generation API:** Ollama `/api/chat` 
- **MCP Server:** Local stdio MCP tools via `mcp`
- **Evaluation:** Custom retrieval + generation benchmarking 
- **CLI Interface:** argparse 
- **Sentence Splitting:** NLTK

---

## 📂 Project Structure

```text
study-assistant/
│
├── configs/
│   └── config.yaml
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── src/
│   ├── ingestion/
│   ├── chunking/
│   ├── embedding/
│   ├── retrieval/
│   ├── reranking/
│   ├── generation/
│   ├── services/
│   ├── mcp_server/
│   ├── vector_store/
│   └── utils/
│       └── config.py
│
├── scripts/
│   ├── run_ingestion_pipeline.py
│   ├── run_query_pipeline.py
│   ├── evaluate_rag.py
│   └── run_chunking_ab_test.py
│
├── evaluation/
│   ├── queries.json
│   └── results/                  # generated/reference eval artifacts
│
├── data/
│   ├── raw/                      # source documents, not generated
│   ├── processed/                # generated local artifacts
│   │   └── bm25_index/           # generated BM25 cache
│   └── ingestion_manifest.json   # generated local manifest
│
├── tests/
│
├── pyproject.toml
├── CHANGELOG.md
└── README.md
```

---

## ⚙️ Configuration

`configs/config.yaml` is the shared source of truth for model names, paths, chunking defaults, retrieval settings, generation settings, and evaluation settings. `src/utils/config.py` loads that config, resolves project-relative paths, and applies the existing `RAG_*` environment variable overrides.

Scripts and services now read from the shared config instead of duplicating hardcoded model and path constants. This keeps evaluation, smoke scripts, and query workflows more reproducible without turning the project into a full production configuration system.

---

## ✅ Testing and CI

The repo includes a pytest suite with 66 passing tests covering chunking, ingestion helpers, manifest logic, retrieval helpers, generation formatting, service helpers, script regression checks, and one Chroma `EphemeralClient` integration test.

Tests are designed to be hermetic where possible and avoid Ollama, real model downloads, network calls, real PDFs, and production vector stores. Ruff is used for linting and formatting, and GitHub Actions runs pytest plus Ruff checks on push and pull request.

---

## 🔄 Pipeline Workflow

### 1. Ingestion Pipeline

Run ingestion from the command line:

```bash
python -m scripts.run_ingestion_pipeline -f data/raw/lecture_pdfs/Lecture_01.pdf
```

Reset and rebuild collection:

```bash
python -m scripts.run_ingestion_pipeline \
-f data/raw/lecture_pdfs/Lecture_01.pdf \
--reset-collection
```

Supported file types:

- `.pdf`
- `.txt`
- `.md`

### Ingestion Flow

```text
File → Extract Text → Clean Text → Chunk → Embed → Store in ChromaDB
```

### Ingestion Manifest

The ingestion pipeline uses a local manifest at:

```text
data/ingestion_manifest.json
```

Successful ingestions are tracked by source file hash. On later runs, unchanged files with a successful manifest record are skipped, failed files are retried, and changed files are reprocessed. For changed documents, old Chroma vectors are replaced only after new parsing, chunking, and embedding succeeds, which avoids dropping the previous working vectors on a failed re-ingestion.

The manifest tracks each ingested document using:

- `document_id`
- `file_name`
- `file_path`
- `file_hash`
- `file_size`
- `source_type`
- `course`
- `title`
- `chunks_created`
- `ingested_at`
- `status`

File hashes are computed with SHA-256 over the document contents, not just the file path. This allows the pipeline to detect whether a source file has changed since the last successful ingestion.

The manifest supports incremental behavior such as:

```text
file unchanged → skip ingestion
file changed   → re-ingest
new file       → ingest
failed status  → retry
```

The manifest is local runtime state and should not be committed.

Useful ingestion reset flags:

| Flag | Behavior |
|---|---|
| `--force-reingest` | Reprocess files without deleting all generated artifacts |
| `--reset-manifest` | Start the run with a fresh manifest |
| `--reset-collection` | Reset the Chroma collection only |
| `--reset-artifacts` | Safely clear generated chunks, embeddings, extracted texts, BM25 cache, manifest, and reset Chroma for a clean local rebuild |

`--reset-artifacts` never deletes `data/raw` and refuses unsafe deletion targets such as source-document paths or repo-critical directories.


### Textbook PDF Ingestion

Textbook PDFs are parsed with Docling using provenance-aware page metadata. For textbook-style PDFs, the ingestion pipeline preserves page boundaries from Docling’s internal document structure and converts them into page-marked Markdown before chunking.

Textbook chunks include:

- `source_type`
- `course`
- `file_name`
- `chapter`
- `section`
- `page_start`
- `page_end`
- `chunk_id`

This enables page-aware textbook citations such as:

```text
ISLP_chapter_2.pdf, K -Nearest Neighbors, pages 22-23
```

---

## 🔍 Retrieval + Generation Flow

```text
User Query
    ↓
Embed Query with Ollama Embedding Model
    ↓
Dense Retrieval from ChromaDB
    +
BM25 Retrieval from Chunk Artifacts
    ↓
Optional Hybrid Score Fusion
    ↓
Optional Cross-Encoder Reranking
    ↓
Select Final Top-K Chunks
    ↓
Build Metadata-Rich Context Block
    ↓
Generate Answer with Ollama Chat API
    ↓
Append Deterministic Source Citations
    ↓
Return Answer + Sources
```

Generation uses Ollama’s /api/chat endpoint with separate system and user messages. System instructions define study-assistant behavior, while the user message contains retrieved context and the question.

---

## 💬 Query Pipeline

Run interactive query retrieval and generation:

```bash
python -m scripts.run_query_pipeline \
-q "What is supervised learning?"
```

### Retrieval-Only Debugging

```bash
python -m scripts.run_query_pipeline \
-q "What is supervised learning?" \
--no-generate \
--show-context
```

### Optional Reranking

```bash
python -m scripts.run_query_pipeline \
-q "What is supervised learning?" \
--use-reranker \
--candidate-k 8 \
--show-sources
```

### Optional Hybrid Retrieval

```bash
python -m scripts.run_query_pipeline \
-q "What is least squares?" \
--use-hybrid \
--dense-k 8 \
--bm25-k 8 \
--hybrid-alpha 0.6 \
--show-sources
```

### Hybrid + Reranker

```bash
python -m scripts.run_query_pipeline \
-q "What is supervised learning?" \
--use-hybrid \
--use-reranker \
--dense-k 8 \
--bm25-k 8 \
--hybrid-alpha 0.7 \
--candidate-k 8 \
--top-k 4 \
--show-sources
```

---

### Available CLI Options

| Option | Purpose |
|---|---|
| `--top-k` | Control retrieval depth |
| `--show-sources` | Print retrieved chunk previews |
| `--show-context` | Print full retrieved chunks |
| `--no-generate` | Disable LLM generation |
| `--model` | Specify Ollama model |
| `--embedding-model` | Specify embedding model |
| `--collection` | Select Chroma collection |
| `--persist-dir` | Override vector DB location |
| `--use-reranker` | Enable cross-encoder reranking |
| `--candidate-k` | Number of dense retrieval candidates before reranking |
| `--reranker-model` | Specify reranker model |
| `--use-hybrid` | Enable hybrid dense + BM25 retrieval |
| `--dense-k` | Number of dense candidates before hybrid fusion |
| `--bm25-k` | Number of BM25 candidates before hybrid fusion |
| `--hybrid-alpha` | Dense retrieval weight in hybrid scoring |

---

### Persisted BM25 Index

Hybrid retrieval uses BM25 lexical retrieval alongside dense vector retrieval. The BM25 index is persisted under:

```text
data/processed/bm25_index/
```

This lets hybrid retrieval reuse a saved BM25 index instead of rebuilding BM25 from chunk JSON artifacts on every query. The cache is invalidated using a fingerprint of the chunk artifact files, so missing, stale, or corrupt BM25 cache artifacts rebuild automatically.

Filtered or course-specific retrieval paths can still use ad-hoc filtered BM25 behavior where that preserves current ranking semantics. The BM25 cache is generated local runtime state and should not be committed.

---

### Recommended Retrieval Modes


| Use Case                                         | Recommended Mode                                                                |
| ------------------------------------------------ | ------------------------------------------------------------------------------- |
| Fast baseline retrieval                          | Dense retrieval                                                                 |
| Exact technical terms / formulas / section names | Hybrid retrieval                                                                |
| Best evaluated quality                           | Hybrid + Reranker                                                               |
| Retrieval debugging / experiments                | Compare Dense, Hybrid, and Hybrid + Reranker                                    |
| Current practical default                        | Hybrid retrieval, with reranker optional when quality matters more than latency |

---

### Retrieval Debugging Modes

#### Source Preview Mode

```bash
--show-sources
```

Displays:

- final rank
- chunk ID
- course
- file name
- source type
- semantic section metadata
- short preview text
- dense retrieval rank
- dense similarity
- BM25 rank
- BM25 score
- hybrid score
- rerank score
- textbook chapter metadata
- textbook page ranges
- deterministic source citations
- retrieval policy filtering for exercise-style sections

#### Full Context Mode

```bash
--show-context
```

Displays:

- full retrieved chunk text
- section-aware metadata
- ranking information
- dense retrieval rank
- rerank score
- reranked final rank

This enables detailed retrieval debugging and chunk quality inspection.

---

## 🔌 Local MCP Server

The project includes a lightweight local MCP server for read-only RAG operations. It lets an MCP client inspect the vector database and run retrieval without memorizing CLI commands.

Phase 1 tools:

- `health_check`
- `collection_stats`
- `list_indexed_documents`
- `search_notes`

Run the server locally over stdio:

```bash
python -m src.mcp_server.server
```

The MCP layer wraps the existing RAG code and does not replace the CLI scripts. Phase 1 is read-only: it does not ingest files, reset the vector database, generate study guides, or execute arbitrary shell commands.

See `MCP_USAGE.md` for setup, tool behavior, configuration, and troubleshooting.

---

## 📊 Evaluation Framework

The project supports retrieval experimentation through:
- chunking A/B testing
- reranker comparisons
- retrieval debugging
- metadata-aware retrieval inspection
- qualitative + quantitative evaluation

Evaluation compares:

- Dense Baseline
- Dense + Reranker
- Hybrid
- Hybrid + Reranker

- retrieval score
- generation score
- reranking deltas
- top-result ranking changes

Run the default full evaluation:

```bash
python -m scripts.evaluate_rag
```

Full mode is the default and preserves the current behavior: it runs retrieval,
reranking, generation, retrieval scoring, and generation scoring. Retrieval-only
mode skips the generation model entirely, so it is much faster for checking
retrieval quality. It still uses the configured retrieval models and embeddings,
and may load the reranker when reranker pipelines are selected.

Useful eval controls:

- `--mode full|retrieval`
- `--limit N`
- `--group semantic|lexical`
- `--pipelines dense,dense_reranker,hybrid,hybrid_reranker`

By default, evaluation also saves a timestamped JSON artifact under:

```text
evaluation/results/
```

The terminal summary still prints as before. Saved artifacts use schema version `1.1` and include a run ID, timestamp, git metadata, config/model metadata, `run_options`, run/pipeline/query timing, summary metrics, grouped metrics, pipeline results, and compact per-query records with source metadata, IDs, scores, and timing. Retrieval-only artifacts store generation scores as `null`. Artifacts avoid full retrieved chunk text to keep files smaller and easier to review.

Examples:

```bash
python -m scripts.evaluate_rag

python -m scripts.evaluate_rag \
  --mode retrieval \
  --limit 5 \
  --pipelines hybrid,hybrid_reranker \
  --run-name retrieval-smoke

python -m scripts.evaluate_rag \
  --results-dir evaluation/results/dev \
  --run-name dev-eval

python -m scripts.evaluate_rag \
  --mode full \
  --limit 3 \
  --pipelines hybrid \
  --run-name full-smoke

python -m scripts.evaluate_rag \
  --group semantic \
  --mode retrieval

python -m scripts.evaluate_rag --no-save-results
```

Generated eval artifacts should be committed deliberately only when they are intended as reference runs. Avoid `git add .` for evaluation outputs.

Evaluation uses:

```text
evaluation/queries.json
```

Metrics measured:

- Retrieval keyword coverage
- Generation keyword coverage
- Per-query retrieval performance
- Average retrieval score
- Average generation score

---
### Current Evaluation Results

Latest local evaluation run:

```text
Date: 2026-07-05
Command: python -m scripts.evaluate_rag
Queries: 28 total, grouped into 14 semantic and 14 lexical / hybrid queries
Embedding model: qwen3-embedding:4b
Generation model: qwen3.6:27b
Reranker: mixedbread-ai/mxbai-rerank-base-v1
Metric: keyword coverage over expected retrieval/generation terms
```

> Note: These results replace earlier README tables produced before the shared config migration. Older reranker results are not directly comparable because the project previously used a different reranker/model configuration.
> Future reference evaluation numbers should come from committed `evaluation/results/*_eval_run.json` artifacts.

#### All Queries

| Pipeline          | Retrieval | Generation |
| ----------------- | --------: | ---------: |
| Dense Baseline    |      0.21 |       0.27 |
| Dense + Reranker  |      0.19 |       0.27 |
| Hybrid            |      0.65 |       0.59 |
| Hybrid + Reranker |      0.81 |       0.75 |

#### Semantic Queries

| Pipeline          | Retrieval | Generation |
| ----------------- | --------: | ---------: |
| Dense Baseline    |      0.27 |       0.15 |
| Dense + Reranker  |      0.22 |       0.13 |
| Hybrid            |      0.50 |       0.46 |
| Hybrid + Reranker |      0.67 |       0.61 |

#### Lexical / Hybrid Queries

| Pipeline          | Retrieval | Generation |
| ----------------- | --------: | ---------: |
| Dense Baseline    |      0.16 |       0.39 |
| Dense + Reranker  |      0.17 |       0.42 |
| Hybrid            |      0.81 |       0.71 |
| Hybrid + Reranker |      0.94 |       0.88 |

#### Observations

* Hybrid retrieval significantly outperformed dense-only retrieval on this corpus and query set.
* Hybrid + Reranker was the strongest overall pipeline, improving all-query retrieval from `0.21` to `0.81` and generation from `0.27` to `0.75`.
* Dense + Reranker did not improve over Dense Baseline, which suggests reranking weak dense candidates is not enough by itself.
* Reranking was most useful after hybrid retrieval, where the candidate pool already contained stronger lexical and semantic matches.
* Lexical / exact-term queries benefited the most from hybrid retrieval.
* The current evaluation metric is keyword coverage, which is useful for regression testing but limited. A future evaluation version should use a labeled gold set with rank-aware metrics such as recall@k, MRR, and nDCG.
---
## 🧪 Chunking A/B Testing

Run A/B testing:

```bash
python -m scripts.run_chunking_ab_test
```

### Tested Configurations

| Config | Chunk Size | Overlap |
|---|---|---|
| A | 700 | 50 |
| B | 900 | 75 |

### Results

| Config | Chunks | Retrieval Score | Generation Score |
|---|---|---|---|
| A_700_50 | 94 | 0.67 | 0.53 |
| B_900_75 | 71 | 0.79 | 0.62 |

### Outcome

```text
Winner: B_900_75
```

Key findings:

- Larger chunks improved semantic coherence
- Reduced fragmentation of conceptual sections
- Improved retrieval quality by ~12%
- Improved generation quality by ~9%

---

## Text Cleaning Pipeline

The ingestion pipeline performs deterministic preprocessing to improve retrieval quality:

- HTML artifact removal
- OCR noise filtering
- image placeholder cleanup
- markdown normalization
- whitespace normalization
- wrapped-line reconstruction

Benefits:

- cleaner semantic chunks
- improved retrieval observability
- reduced OCR contamination
- better section metadata quality

---
## 📈 Key Learnings

- Chunking strategy significantly impacts retrieval quality
- Retrieval quality is often the primary bottleneck in RAG systems
- Evaluation is critical — naive RAG demos can be misleading
- Larger chunks improved conceptual question answering in this corpus
- Standardized document objects simplify multi-format ingestion
- Retrieval debugging tools significantly improve observability
- Top-k retrieval depth materially impacts answer grounding quality
- Transcript-style text introduces semantic noise and OCR artifacts
- Semantic section metadata improves retrieval traceability and debugging
- Cross-encoder rerankers can significantly reorder dense retrieval outputs
- Hybrid retrieval significantly improved both retrieval and generation quality on the current corpus.
- Reranking is most useful after hybrid retrieval, where the candidate pool already contains stronger matches.
- Dense-only retrieval remains useful as a fast baseline, but it underperformed hybrid retrieval in the latest evaluation.
- Keyword-coverage evaluation is useful for early regression testing, but a labeled gold set with rank-aware metrics is needed for stronger retrieval claims.
- Semantic chunk precision remains a major retrieval bottleneck
- BM25 improves retrieval for exact lecture terms, formulas, abbreviations, and section names
- Hybrid retrieval improves lexical query performance without replacing dense retrieval
- Grouped evaluation is necessary because semantic and lexical queries stress different retrieval behaviors
- Textbook PDFs require page-aware metadata to support trustworthy citations
- Docling provenance is more useful than plain Markdown export for textbook retrieval
- Strict heading detection reduces section pollution from captions, glossary fragments, and margin text
- Exercise sections can pollute lexical retrieval and should be penalized or filtered for normal study queries
- Source metadata should be passed into the generation context, not only printed during retrieval debugging
- Local model choice materially affects latency, answer quality, and hardware requirements.
- Ollama chat generation is better suited than raw prompt generation for the RAG answer step because it separates system instructions from user context.
- Thinking-capable models may spend generation budget on internal reasoning unless token budgets and model options are configured carefully.
- Manifest-based ingestion is necessary before scaling from single-file ingestion to folder-level multi-document ingestion.
- File content hashing is more reliable than file-path tracking for detecting changed source documents.
---

## 🔒 Local-First Design

- No external LLM API required
- Runs fully on local hardware using Ollama
- Suitable for sensitive/private documents
- Modular ingestion supports multiple document types

---

## 🚧 Future Improvements

- Improve semantic section-boundary chunking
- Section-aware reranking
- Improve hybrid retrieval weighting and query-type routing
- Add automatic routing between dense and hybrid retrieval
- Add section-boundary-aware chunking
- Query rewriting / expansion
- Multi-document retrieval
- Improved reranking strategies
- Add manifest-aware vector DB cleanup for deleted or renamed files
- Metadata filtering
- Better OCR/transcript cleanup
- Latency optimization
- FastAPI inference layer
- Streamlit or web UI

---

## 🏁 Getting Started

### 1. Install dependencies

Use Python 3.12. The local environment is direnv-managed:

```bash
direnv allow
python -m pip install -e .
```

Docling and Torch-related packages are large and may take a while to install.
NLTK `punkt` data is downloaded automatically on first textbook ingestion.

### 2. Start Ollama

```bash
ollama serve
```

### 3. Pull local models

```bash
ollama pull qwen3-embedding:4b
ollama pull qwen3.6:27b

Optional reranker model is downloaded automatically from Hugging Face on first use:

mixedbread-ai/mxbai-rerank-base-v1
```

### 4. Run ingestion

```bash
python -m scripts.run_ingestion_pipeline -f data/raw/lecture_pdfs/Lecture_01.pdf
```

### 5. Run query pipeline

```bash
python -m scripts.run_query_pipeline \
-q "What is supervised learning?"
```

### 6. Run the local MCP server

```bash
python -m src.mcp_server.server
```

### 7. Run evaluation

```bash
python -m scripts.evaluate_rag
```

### 8. Run chunking A/B tests

```bash
python -m scripts.run_chunking_ab_test
```

---

## 💡 Why This Project Matters

This project goes beyond a simple chatbot:

- End-to-end ML system design
- Multi-format ingestion architecture
- Real evaluation framework
- Measurable retrieval optimization
- Local LLM deployment
- Retrieval experimentation via A/B testing
- Retrieval debugging and observability tooling
- Metadata-aware retrieval workflows
- Two-stage retrieval pipelines with reranking
- Hybrid retrieval with dense + BM25 score fusion
- Grouped retrieval evaluation across semantic and lexical query types
- Configurable retrieval modes for experimentation

---

## 📌 Summary

```text
Built a local-first RAG-based study assistant with: 
- multi-format document ingestion 
- deterministic text cleaning 
- metadata-aware chunking 
- vector retrieval with ChromaDB 
- Ollama-based embedding generation 
- BM25 keyword retrieval with `bm25s` 
- hybrid dense + BM25 retrieval 
- optional cross-encoder reranking 
- local answer generation with Ollama `/api/chat` 
- retrieval evaluation and A/B testing frameworks 
- retrieval debugging and observability tooling 
- provenance-aware textbook PDF ingestion with Docling 
- sentence-aware textbook chunking with page, chapter, and section metadata 
- deterministic source citations with textbook page ranges 
- ingestion manifest utilities with SHA-256 file hashing 
- groundwork for folder-level incremental ingestion 

This project focuses on retrieval systems engineering, experimentation, and evaluation rather than simple chatbot generation workflows.
```
