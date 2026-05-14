# Changelog

All notable changes to the Local RAG Study Assistant project will be documented in this file.

---

# 2026-05-13

## Added

### Query Pipeline
- Added `run_query_pipeline.py`
- Added CLI query interface:
  ```bash
  python -m scripts.run_query_pipeline -q "<query>"
  ```
- Added retrieval-only debugging mode with:
  - `--no-generate`
  - `--show-sources`
  - `--show-context`
- Added configurable:
  - `--top-k`
  - `--model`
  - `--embedding-model`
  - `--collection`
  - `--persist-dir`

---

### Retrieval Debugging
- Added `print_sources()` for compact retrieval previews
- Added `print_context()` for full retrieved chunk inspection
- Added retrieval metadata printing:
  - rank
  - chunk_id
  - similarity score
  - file_name
  - source_type

---

### Metadata Improvements
- Added `file_name` propagation into chunk metadata
- Improved metadata-aware retrieval inspection
- Added source attribution visibility in query outputs

---

### Ingestion Pipeline
- Added argparse CLI support to `run_ingestion_pipeline.py`
- Added multi-format ingestion support:
  - `.pdf`
  - `.txt`
  - `.md`
- Added:
  ```bash
  --reset-collection
  ```
  flag for rebuilding Chroma collections directly from CLI

---

### Retrieval Improvements
- Increased default retrieval depth experimentation from:
  ```text
  top_k = 2 → top_k = 4
  ```
- Improved retrieval grounding quality for conceptual queries

---

### Evaluation & Experimentation
- Completed chunking A/B testing workflow
- Compared:
  - `700/50`
  - `900/75`
- Identified `900/75` as best-performing configuration

Results:
| Config | Retrieval | Generation |
|---|---|---|
| 700/50 | 0.67 | 0.53 |
| 900/75 | 0.79 | 0.62 |

---

### Documentation
- Expanded README with:
  - query pipeline usage
  - retrieval debugging modes
  - CLI examples
  - reset collection workflow
  - metadata-aware retrieval
  - updated architecture and learnings

---

## Fixed

- Fixed duplicate source printing during `--show-context`
- Fixed chunk metadata missing `file_name`
- Fixed argparse configuration issues in query pipeline
- Fixed collection reset workflow integration
- Fixed retrieval preview formatting
- Fixed ingestion pipeline reset handling

---

## Learned

- Retrieval quality is highly sensitive to chunking strategy
- Larger chunk sizes improved semantic coherence
- Retrieval debugging tools significantly improve observability
- Transcript-style text introduces semantic retrieval noise
- Top-k retrieval depth materially impacts answer grounding quality
