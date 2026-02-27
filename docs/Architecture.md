# System Architecture

## High-Level Overview

User Query
   ↓
Retrieve Relevant Chunks (Vector DB)
   ↓
Local LLM (Ollama)
   ↓
Optional Escalation (Claude/OpenAI API)
   ↓
Final Response

---

## Core Components

### 1. Ingestion Layer
- PDF parsing
- Transcript cleaning
- Chunking (800–1200 chars, overlap 150–200)
- Metadata tagging

### 2. Embeddings
- Model: all-MiniLM-L6-v2
- Stored in Chroma or FAISS
- Cached locally

### 3. Retrieval
- Top-k search (k=3–6)
- Score thresholding
- Later: hybrid search (BM25 + vector)

### 4. Generation

#### Default
- Local model via Ollama
- Mistral 7B or LLaMA 3.1 8B (quantized)

#### Optional Escalation
- Claude API (Anthropic)
- OpenAI API
- Triggered manually or by confidence rule

---

## Hardware Plan

Phase 1:
- MacBook Pro (M1, 8GB)

Phase 2:
- Mac mini (M4, 16GB+)
- External SSD for corpus storage

Optional Later:
- Raspberry Pi for orchestration layer

---

## Design Principles

- Local-first execution
- API usage only when necessary
- Clear separation of retrieval vs generation
- Modular backend switching
- Explicit tradeoff documentation