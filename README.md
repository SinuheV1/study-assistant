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
├── src/
│   ├── ingestion/
│   │   ├── ingest_docling_document.py
│   │   ├── ingest_text.py
│   │   ├── clean_text.py
│   │   └── manifest.py
│   │
│   ├── chunking/
│   ├── embedding/
│   ├── retrieval/
│   ├── reranking/
│   ├── generation/
│   ├── services/
│   ├── mcp_server/
│   ├── vector_store/
│   └── utils/
│
├── scripts/
│   ├── run_ingestion_pipeline.py
│   ├── run_query_pipeline.py
│   ├── evaluate_rag.py
│   └── run_chunking_ab_test.py
│
├── evaluation/
│   ├── queries.json
│   └── results/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── ingestion_manifest.json
└── README.md
```

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

The ingestion pipeline includes manifest utilities for repeatable and incremental document ingestion.

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

This enables future folder-level ingestion behavior such as:

```text
file unchanged → skip ingestion
file changed   → re-ingest
new file       → ingest
failed status  → retry
```

The manifest is designed to support scalable multi-document ingestion without repeatedly reprocessing unchanged PDFs, transcripts, or notes.


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

---

```md
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

### Recommended Retrieval Modes

| Use Case | Recommended Mode |
|---|---|
| Broad conceptual questions | Dense retrieval |
| Exact technical terms / formulas / section names | Hybrid retrieval |
| Retrieval experiments | Hybrid and reranker modes |
| Current default | Dense retrieval |

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

Dense Retrieval Baseline
        vs
Dense Retrieval + Cross-Encoder Reranking

- retrieval score
- generation score
- reranking deltas
- top-result ranking changes

Run evaluation:

```bash
python -m scripts.evaluate_rag
```

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

#### All Queries

| Pipeline | Retrieval | Generation |
|---|---:|---:|
| Dense Baseline | 0.87 | 0.76 |
| Dense + Reranker | 0.87 | 0.70 |
| Hybrid | 0.87 | 0.78 |
| Hybrid + Reranker | 0.86 | 0.75 |

#### Semantic Queries

| Pipeline | Retrieval | Generation |
|---|---:|---:|
| Dense Baseline | 0.78 | 0.70 |
| Dense + Reranker | 0.78 | 0.62 |
| Hybrid | 0.74 | 0.71 |
| Hybrid + Reranker | 0.76 | 0.67 |

#### Lexical / Hybrid Queries

| Pipeline | Retrieval | Generation |
|---|---:|---:|
| Dense Baseline | 0.96 | 0.83 |
| Dense + Reranker | 0.95 | 0.78 |
| Hybrid | 1.00 | 0.84 |
| Hybrid + Reranker | 0.95 | 0.83 |

Observations:
- Dense retrieval remains the best default for broad semantic questions.
- Hybrid retrieval improves lexical / exact technical-term retrieval.
- Hybrid improved lexical retrieval from `0.96` to `1.00`.
- Reranking changes top results often but does not currently improve evaluation scores.
- Retrieval mode should remain configurable instead of forcing one global strategy.
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
- Better retrieval ranking does not always improve downstream generation quality
- Semantic chunk precision remains a major retrieval bottleneck
- Dense retrieval performs best as the default semantic search baseline
- BM25 improves retrieval for exact lecture terms, formulas, abbreviations, and section names
- Hybrid retrieval improves lexical query performance without replacing dense retrieval
- Reranking is useful for experimentation but is not currently beneficial as a default
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
- Add saved evaluation result artifacts
- Query rewriting / expansion
- Multi-document retrieval
- Improved reranking strategies
- Wire ingestion manifest into folder-level ingestion 
- Add skip-unchanged-file logic to the ingestion pipeline 
- Add force-reingest and reset-manifest CLI options 
- Add manifest-aware vector DB cleanup for deleted or renamed files
- Metadata filtering
- Better OCR/transcript cleanup
- Latency optimization
- FastAPI inference layer
- Streamlit or web UI

---

## 🏁 Getting Started

### 1. Install dependencies

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
