"""
Antigravity Project Memory: Cross-conversation memory and algorithmic BM25 RAG MCP server.
"""

from .retrieval_engine import (
    DocumentChunk,
    ProjectRetrievalEngine,
    SearchResult,
    normalize_workspace_uri,
)
from .server import (
    get_conversation_artifact,
    get_conversation_transcript,
    list_project_conversations,
    mcp,
    resolve_workspace,
    search_project_memory,
)

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "mcp",
    "search_project_memory",
    "list_project_conversations",
    "get_conversation_transcript",
    "get_conversation_artifact",
    "resolve_workspace",
    "ProjectRetrievalEngine",
    "SearchResult",
    "DocumentChunk",
    "normalize_workspace_uri",
]
