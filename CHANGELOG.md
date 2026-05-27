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

---

# 2026-05-14

## Added

### Section-Aware Metadata
- Added section-aware chunk metadata
- Added 'section' and 'sections' metadata fields
- Added semantic section propogation during chunk construction
- Added normalized heading extraction for section tracking

### Query Output Improvements
- Added Course metadata to retrieval debugging output
- Added Section metadata to retrieval debugging output
- Added multi-section visibility via 'sections'

### Retrieval Observability
- Improved source traceability across:
  - course
  - source type
  - file name
  - semantic section
  - chunk ID

---

## Changed

### Chunking Architecture
- Refactored chunk builder to return structured chunk objects instead of plain text strings
- Added section-aware chunk construction workflow
- Improved heading attachment handling during chunk assembly

---

## Validated

- Confirmed section metadata propogates correctly into:
  - chunk records
  - embeddings
  - Chroma metadata
  - retrieval outputs

- Successfully re-ingested and retrieved:
  - course metadata
  - section metadata
  - source metadata

---

## Known Issues

- 'is_heading()' currently over-detects some OCR/image artifacts:
  - '<!-- image -->'
  - malformed OCR text
- Retrieval quality still impaacted by transcript/PDF extraction noise

---

# 2026-05-19

## Added

### Heading Detection Improvements
- Improved `is_heading()` heuristics for semantic section detection
- Added markdown heading prioritization
- Added OCR artifact rejection heuristics
- Added merged-token OCR detection
- Added sentence punctuation rejection for headings
- Added list-item rejection for heading detection
- Added heading normalization cleanup

---

## Changed

### Chunking & Section Metadata
- Improved section-aware chunk metadata quality
- Reduced false-positive section assignments from OCR artifacts
- Improved semantic section propagation during chunk construction

### Retrieval Debugging
- Retrieval output now surfaces cleaner section metadata
- Improved retrieval observability for multi-section chunks

---

## Fixed

- Fixed OCR artifacts incorrectly becoming semantic sections:
  - `Whatarethedesired`
  - `<!-- image -->`

- Fixed heading regex matching behavior
- Fixed false-positive heading detection from malformed transcript text

---

## Known Issues

- Some chunks still span multiple semantic sections
- Primary section assignment currently favors most recent section in chunk
- Transcript/PDF extraction noise still impacts chunk cleanliness

---

## Learned

- Retrieval quality is highly sensitive to heading detection heuristics
- OCR/transcript cleanup materially affects semantic chunk quality
- Structure-aware chunking introduces new retrieval engineering challenges
- Semantic metadata improves retrieval debugging and observability

---

# 2026-05-20

## Added

### Text Cleaning Pipeline
- Added deterministic text preprocessing pipeline in `clean_text.py`
- Added HTML comment cleanup
- Added image artifact removal
- Added markdown heading normalization
- Added whitespace normalization
- Added trailing whitespace cleanup
- Added wrapped-line reconstruction for OCR/transcript exports
- Added short noise-line filtering
- Added OCR merged-token detection heuristics
- Added cleaning statistics logging

---

## Changed

### Ingestion Quality
- Improved extracted text cleanliness before chunking
- Reduced OCR artifact propagation into chunks
- Improved semantic chunk readability
- Improved section metadata quality after preprocessing

### Retrieval Quality
- Improved retrieval context readability
- Reduced retrieval noise from malformed OCR text
- Improved retrieval observability during evaluation

---

## Fixed

- Fixed markdown heading normalization behavior
- Fixed HTML comment cleanup regex
- Fixed artifact leakage into retrieval chunks
- Fixed malformed OCR headings affecting chunk metadata

---

## Evaluation

### Post-cleaning Evaluation Results
| Metric | Score |
|---|---|
| Avg Retrieval Score | 0.78 |
| Avg Generation Score | 0.62 |

Notes:
- Retrieval quality became noticeably cleaner and more interpretable
- OCR artifacts were substantially reduced
- Retrieval bottleneck is shifting from preprocessing toward ranking quality

---

## Learned

- Corpus cleanliness strongly impacts retrieval quality
- Deterministic preprocessing improves observability and debugging
- OCR artifacts degrade semantic chunk quality
- Retrieval evaluation should combine quantitative scores with qualitative inspection

---

# 2026-05-21

## Added

### Reranking Pipeline
- Added optional cross-encoder reranking pipeline
- Added `src/reranking/reranker.py`
- Added support for:
  - `--use-reranker`
  - `--reranker-model`
  - `--candidate-k`
- Added local reranker support using:
  - `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Added reranker-aware retrieval debugging metadata:
  - dense_rank
  - dense_similarity
  - rerank_score

---

### Evaluation Improvements
- Expanded `evaluate_rag.py` to compare:
  - dense retrieval baseline
  - dense + reranker pipeline
- Added reranker evaluation metrics:
  - retrieval deltas
  - generation deltas
  - top-result change tracking
- Added side-by-side retrieval pipeline benchmarking

---

## Changed

### Query Pipeline
- Updated query pipeline to support two-stage retrieval:
  ```text
  dense retrieval → reranking → generation

## Learned 

- Cross-encoder rerankers can substantially reorder dense retrieval results
- Improved retrieval ranking does not necessarily improve downstream generation quality
- Retrieval evaluation requires both quantitative metrics and qualitative chunk inspection
- Chunk contamination and semantic boundary overlap still impact retrieval precision
- Retrieval bottlenecks have shifted from OCR/text-cleaning issues toward semantic chunk quality
- Dense retrieval and reranking optimize different aspects of retrieval quality:
  - dense retrieval improves recall
  - reranking improves ranking precision
- Observability tooling (dense rank, rerank score, retrieval previews) is critical for debugging RAG systems
- Optional/feature-flagged experimentation enables safe iterative development of retrieval pipelines


# 2026-05-26

## Added

### Hybrid Retrieval
- Added standalone BM25 retrieval using saved chunk JSON artifacts
- Added `src/retrieval/bm25_retriever.py`
- Added `src/retrieval/hybrid_retrieval.py`
- Added hybrid retrieval pipeline combining:
  - dense vector retrieval from ChromaDB
  - BM25 lexical retrieval
  - score normalization
  - chunk deduplication by `chunk_id`
  - weighted hybrid scoring
- Added hybrid retrieval CLI support:
  - `--use-hybrid`
  - `--dense-k`
  - `--bm25-k`
  - `--hybrid-alpha`
- Added support for combining hybrid retrieval with reranking:
  ```text
  dense retrieval + BM25 retrieval → hybrid scoring → optional reranking → final top-k

## Learned

-Dense retrieval remains the strongest default for broad semantic questions
- Hybrid retrieval improves exact technical-term retrieval
- BM25 is useful for lecture-specific vocabulary, formulas, abbreviations, and section-title queries
- Hybrid search improved lexical retrieval from 0.96 to 1.00
- Reranking changes rankings frequently but does not currently improve evaluation scores on this corpus
- Retrieval modes should remain configurable instead of forcing one strategy globally
- Grouped evaluation is necessary because semantic and lexical queries stress different retrieval behaviors
- The best current architecture is:
  ```text
  dense retrieval as default
  hybrid retrieval as optional for exact technical terms
  reranking as experimental