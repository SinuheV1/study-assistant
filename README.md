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
User Query → Retrieval → Context Builder → LLM → Answer + Sources
```

---

## ⚙️ Tech Stack

- **Language:** Python
- **LLM (local):** Ollama (Llama 3)
- **Vector Database:** ChromaDB
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **PDF Parsing:** Docling
- **Evaluation:** Custom retrieval + generation benchmarking
- **CLI Interface:** argparse

---

## 📂 Project Structure

```text
study-assistant/
│
├── src/
│   ├── ingestion/
│   │   ├── ingest_docling_document.py
│   │   ├── ingest_text.py
│   │   └── clean_text.py
│   │
│   ├── chunking/
│   ├── embedding/
│   ├── retrieval/
│   ├── generation/
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
│   └── processed/
│
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

---

## 🔍 Retrieval + Generation Flow

```text
User Query
    ↓
Embed Query
    ↓
Retrieve Top-K Chunks
    ↓
Build Context Block
    ↓
Generate Answer with Ollama
    ↓
Return Answer + Sources
```

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

### Retrieval Debugging Modes

#### Source Preview Mode

```bash
--show-sources
```

Displays:

- similarity score
- chunk ID
- course
- file name
- source type
- semantic section metadata
- short preview text

#### Full Context Mode

```bash
--show-context
```

Displays:

- full retrieved chunk text
- section-aware metadata
- ranking information

This enables detailed retrieval debugging and chunk quality inspection.

---

## 📊 Evaluation Framework

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

---

## 🔒 Local-First Design

- No external LLM API required
- Runs fully on local hardware using Ollama
- Suitable for sensitive/private documents
- Modular ingestion supports multiple document types

---

## 🚧 Future Improvements

- Improve heading detection heuristics
- Section-aware reranking
- Hybrid retrieval (BM25 + vector search)
- Query rewriting / expansion
- Multi-document retrieval
- Retrieval reranking
- Metadata filtering
- Folder-level ingestion
- Better OCR/transcript cleanup
- Latency optimization
- FastAPI inference layer
- Streamlit or web UI

---

## 🏁 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama

```bash
ollama serve
```

### 3. Pull model

```bash
ollama pull llama3.2:3b
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

### 6. Run evaluation

```bash
python -m scripts.evaluate_rag
```

### 7. Run chunking A/B tests

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

---

## 📌 Summary

```text
Built a local-first RAG system with ingestion, retrieval, gener