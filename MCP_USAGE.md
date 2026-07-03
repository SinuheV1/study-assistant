# RAG Study Assistant MCP Server

This MCP server exposes the local RAG Study Assistant as read-only tools for an MCP client such as OMP Agent.

It wraps the existing Python RAG code. It does not replace the CLI scripts, add a web app, or implement study guide generation.

## Phase 1 Tools

- `health_check`: checks the configured vector database and returns basic system settings.
- `collection_stats`: returns collection count, indexed courses, inferred weeks, and source count.
- `list_indexed_documents`: lists indexed documents from Chroma metadata, with chunk JSON fallback.
- `search_notes`: retrieves chunks using hybrid retrieval and reranking by default.

Phase 1 does not include ingestion, answer generation, or vector database reset tools.

## Run Locally

Install dependencies in your project environment:

```bash
pip install -r requirements.txt
```

Run the MCP server over stdio:

```bash
python -m src.mcp_server.server
```

If your environment uses `python3`, use:

```bash
python3 -m src.mcp_server.server
```

## Optional Configuration

The server uses these defaults:

- `RAG_PERSIST_DIR`: `data/processed/vector_store`
- `RAG_CHUNK_DIR`: `data/processed/chunks`
- `RAG_COLLECTION_NAME`: `study_assistant_chunks`
- `RAG_EMBEDDING_MODEL`: `qwen3-embedding:4b`
- `RAG_RERANKER_MODEL`: `mixedbread-ai/mxbai-rerank-base-v1`

Relative paths are resolved from the project root.

## Example Prompts

- "Check whether my RAG study assistant is healthy."
- "List indexed ANLP documents."
- "Search my notes for cross entropy."
- "Show the chunks retrieved for entropy in ANLP week 2."

## Safety Notes

Phase 1 is read-only. It does not expose tools that ingest files, delete data, reset the vector database, or execute shell commands.

The Study Guide Generation Agent is intentionally separate and unchanged.

## Troubleshooting

- If `health_check` reports a vector database error, confirm the Chroma directory exists and dependencies are installed.
- If `search_notes` returns a reranker warning, the tool falls back to retrieval results instead of failing.
- If `search_notes` fails during embedding, confirm Ollama is running and the configured embedding model is available.

