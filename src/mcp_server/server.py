from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from src.services.rag_service import (
    collection_stats_service,
    health_check_service,
    list_indexed_documents_service,
    search_notes_service,
)


mcp = FastMCP("rag-study-assistant")


@mcp.tool()
def health_check() -> dict:
    """Check whether the local RAG Study Assistant is available."""
    return health_check_service()


@mcp.tool()
def collection_stats() -> dict:
    """Return basic statistics for the configured Chroma collection."""
    return collection_stats_service()


@mcp.tool()
def list_indexed_documents(
    course: str | None = None,
    week: str | None = None,
) -> dict:
    """List documents currently indexed in the vector database."""
    return list_indexed_documents_service(course=course, week=week)


@mcp.tool()
def search_notes(
    query: str,
    course: str | None = None,
    week: str | None = None,
    top_k: int = 8,
    use_hybrid: bool = True,
    use_reranker: bool = True,
) -> dict:
    """Retrieve relevant chunks without generating a final answer."""
    return search_notes_service(
        query=query,
        course=course,
        week=week,
        top_k=top_k,
        use_hybrid=use_hybrid,
        use_reranker=use_reranker,
    )


if __name__ == "__main__":
    mcp.run()

