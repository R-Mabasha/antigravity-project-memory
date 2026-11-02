"""
Model Context Protocol (MCP) Server for Antigravity Project Memory.

Exposes high-value cross-session memory tools for Antigravity workspaces:
- search_project_memory: BM25 Okapi + symbol boost + passage stitching over past chats & artifacts
- list_project_conversations: List all past sessions recorded for a workspace
- get_conversation_transcript: Clean dialogue history without UI XML clutter
- get_conversation_artifact: Read full markdown ADRs, specs, and walkthroughs from brain
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .retrieval_engine import (
    DEFAULT_BRAIN_DIR,
    DEFAULT_DB_PATH,
    DocumentChunk,
    ProjectRetrievalEngine,
    SearchResult,
    normalize_workspace_uri,
)

# Initialize FastMCP Server
mcp = FastMCP(
    name="antigravity-project-memory",
    instructions=(
        "Antigravity Project Memory provides cross-session retrieval over past conversations, "
        "architectural decisions, tool executions, and brain artifacts for the active workspace."
    ),
)

# Shared singleton retrieval engine
engine = ProjectRetrievalEngine(
    db_path=DEFAULT_DB_PATH,
    brain_dir=DEFAULT_BRAIN_DIR,
)


def resolve_workspace(workspace_path: Optional[str] = None) -> str:
    """
    Resolves the workspace root path following the fallback hierarchy:
    1. Explicit workspace_path argument
    2. ANTIGRAVITY_WORKSPACE environment variable
    3. Current working directory (os.getcwd())
    Normalizes Windows drive paths and file:// URIs.
    """
    if workspace_path and workspace_path.strip():
        return normalize_workspace_uri(workspace_path)
    env_ws = os.environ.get("ANTIGRAVITY_WORKSPACE")
    if env_ws and env_ws.strip():
        return normalize_workspace_uri(env_ws)
    return normalize_workspace_uri(os.getcwd())


@mcp.tool()
def search_project_memory(
    query: str,
    workspace_path: Optional[str] = None,
    top_k: int = 5,
    char_budget: int = 2500,
) -> str:
    """Search past Antigravity conversations, decisions, and artifacts for the active workspace using BM25 Okapi and code-symbol boosting.

    Args:
        query: Search keywords, code identifiers, file paths, or architectural questions.
        workspace_path: Optional workspace path or URI. If omitted, auto-detected from ANTIGRAVITY_WORKSPACE or current working directory.
        top_k: Maximum number of relevant passages to return (default: 5).
        char_budget: Maximum character budget per stitched passage (default: 2500).

    Returns:
        Formatted markdown context containing high-relevance conversation dialogue, actions, and artifacts with relevance scores and provenance metadata.
    """
    clean_query = query.strip()
    if not clean_query:
        return "Error: query parameter cannot be empty."

    target_ws = resolve_workspace(workspace_path)

    # Auto-load workspace if not loaded or workspace changed
    if engine.indexed_workspace != target_ws or engine.index.doc_count == 0:
        indexed_chunks = engine.load_workspace(target_ws)
    else:
        indexed_chunks = engine.index.doc_count

    result: SearchResult = engine.search(
        query=clean_query,
        top_k=max(1, min(top_k, 25)),
        stitch_budget=max(500, min(char_budget, 10000)),
    )

    if not result.passages:
        return (
            f"# Antigravity Project Memory Search\n\n"
            f"- **Query**: `{clean_query}`\n"
            f"- **Workspace**: `{target_ws}`\n"
            f"- **Indexed**: {result.conversations_indexed} conversations ({result.total_chunks_indexed} semantic chunks)\n"
            f"- **Latency**: {result.timings_ms.get('total_ms', 0):.2f} ms\n\n"
            f"**No matching passages found.** Try broader keywords or inspect available conversations via `list_project_conversations`."
        )

    out_lines = [
        f"# Antigravity Project Memory Search Results",
        f"- **Query**: `{clean_query}`",
        f"- **Workspace**: `{target_ws}`",
        f"- **Matched**: {result.total_found} relevant passages ({result.timings_ms.get('total_ms', 0):.2f} ms retrieval latency)",
        f"- **Index Stats**: {result.conversations_indexed} conversations indexed ({result.total_chunks_indexed} chunks)",
        "",
        "---",
    ]

    for idx, passage in enumerate(result.passages, 1):
        type_badge = passage.chunk_type.upper()
        out_lines.append(f"## {idx}. [{type_badge}] {passage.title}")
        out_lines.append(
            f"- **Relevance Score**: `{passage.score:.2f}` "
            f"(BM25: `{passage.raw_bm25:.2f}`, Symbol Boost: `{passage.symbol_boost:.2f}x`, Recency: `{passage.recency_factor:.2f}`)"
        )
        out_lines.append(f"- **Conversation ID**: `{passage.conversation_id}`")
        out_lines.append(f"- **Created**: `{passage.created_at}`")
        out_lines.append(f"- **Source**: `{passage.source}`")
        if passage.metadata.get("tools_used"):
            tools_str = ", ".join(f"`{t}`" for t in passage.metadata["tools_used"])
            out_lines.append(f"- **Tools Used**: {tools_str}")
        out_lines.append("")
        out_lines.append(passage.content)
        out_lines.append("")
        out_lines.append("---")

    return "\n".join(out_lines)


@mcp.tool()
def list_project_conversations(
    workspace_path: Optional[str] = None,
    limit: int = 20,
) -> str:
    """List all past conversation threads recorded in Antigravity for the workspace.

    Args:
        workspace_path: Optional workspace path or URI. If omitted, auto-detected from ANTIGRAVITY_WORKSPACE or current working directory.
        limit: Maximum number of conversation threads to return (default: 20).

    Returns:
        Formatted markdown list of past conversations with IDs, titles, dates, step counts, and status.
    """
    target_ws = resolve_workspace(workspace_path)
    convs = engine.discover_workspace_conversations(target_ws)

    if not convs:
        return (
            f"# Antigravity Conversations\n\n"
            f"No recorded conversations found for workspace `{target_ws}`.\n"
            f"*(Checked database: `{engine.db_path}`)*"
        )

    capped_limit = max(1, min(limit, 100))
    display_convs = convs[:capped_limit]

    out_lines = [
        f"# Antigravity Conversations for Workspace",
        f"- **Workspace**: `{target_ws}`",
        f"- **Total Found**: {len(convs)} conversation threads (displaying top {len(display_convs)})",
        "",
        "| # | Conversation Title | Conversation ID | Last Modified | Steps | Status |",
        "| :-: | :--- | :--- | :--- | :-: | :-: |",
    ]

    for idx, c in enumerate(display_convs, 1):
        title = c["title"].replace("|", "\\|")
        cid = c["conversation_id"]
        mod_date = c["last_modified_time"][:19].replace("T", " ") if c["last_modified_time"] else "N/A"
        steps = c["step_count"]
        status = c["status"] or "active"
        out_lines.append(f"| {idx} | **{title}** | `{cid}` | {mod_date} | {steps} | `{status}` |")

    out_lines.append("")
    out_lines.append("### Conversation Previews")
    for idx, c in enumerate(display_convs, 1):
        preview = c.get("preview", "").strip().replace("\n", " ")
        if preview:
            if len(preview) > 160:
                preview = preview[:160] + "..."
            out_lines.append(f"{idx}. **{c['title']}** (`{c['conversation_id']}`):")
            out_lines.append(f"   > {preview}")

    return "\n".join(out_lines)


@mcp.tool()
def get_conversation_transcript(
    conversation_id: str,
    max_turns: int = 30,
) -> str:
    """Extract the clean human-readable dialogue history of a past Antigravity session.

    Args:
        conversation_id: The UUID of the conversation (from search_project_memory or list_project_conversations).
        max_turns: Maximum number of recent dialogue turns to extract (default: 30).

    Returns:
        Clean markdown dialogue history (user requests, tools executed, and assistant responses) stripped of raw UI XML and thought blocks.
    """
    clean_cid = conversation_id.strip()
    if not clean_cid:
        return "Error: conversation_id cannot be empty."

    data = engine.get_transcript_turns(clean_cid, max_turns=max_turns)
    if not data or not data["turns"]:
        conv_dir = os.path.join(engine.brain_dir, clean_cid)
        return (
            f"Error: Transcript not found for conversation `{clean_cid}`.\n"
            f"Looked in `{conv_dir}/.system_generated/logs/transcript.jsonl`."
        )

    out_lines = [
        f"# Conversation Transcript: {data['title']}",
        f"- **Conversation ID**: `{clean_cid}`",
        f"- **Total Turns**: {data['total_turns']} (Showing {'last ' if data['is_truncated'] else ''}{data['returned_turns_count']})",
        f"- **Last Modified**: {data['last_modified_time']}",
        "",
        "---",
    ]

    for turn in data["turns"]:
        t_idx = turn["turn_index"]
        t_date = turn["created_at"]
        out_lines.append(f"## Turn {t_idx} ({t_date})")
        out_lines.append(f"### User Request\n{turn['user_text']}\n")
        if turn["tool_summary"]:
            out_lines.append(f"### Actions Executed\n{turn['tool_summary']}\n")
        if turn["assistant_text"]:
            out_lines.append(f"### Assistant Response\n{turn['assistant_text']}\n")
        out_lines.append("---")

    return "\n".join(out_lines)


@mcp.tool()
def get_conversation_artifact(
    conversation_id: str,
    artifact_name: str,
) -> str:
    """Read and return full markdown specifications, ADRs, or walkthroughs from an Antigravity conversation brain directory.

    Args:
        conversation_id: The UUID of the conversation owning the artifact.
        artifact_name: Name of the artifact file (e.g. 'walkthrough.md', 'task.md', or 'implementation_plan').

    Returns:
        Full text content of the artifact with path metadata, or a list of available artifacts if not found.
    """
    clean_cid = conversation_id.strip()
    clean_art = artifact_name.strip()
    if not clean_cid or not clean_art:
        return "Error: conversation_id and artifact_name cannot be empty."

    art_data = engine.read_artifact(clean_cid, clean_art)
    if art_data:
        return (
            f"# Artifact: {art_data['file_name']}\n"
            f"- **Conversation ID**: `{clean_cid}`\n"
            f"- **Path**: `{art_data['full_path']}`\n"
            f"- **Size**: {art_data['size_bytes']} bytes | **Modified**: {art_data['modified_at']}\n\n"
            f"---\n\n"
            f"{art_data['content']}"
        )

    # Not found -> list available artifacts
    available = engine.list_artifacts(clean_cid)
    if not available:
        return (
            f"Artifact `{clean_art}` not found.\n\n"
            f"No artifacts (*.md, *.txt, *.json) exist in brain directory for conversation `{clean_cid}`."
        )

    lines = [
        f"Artifact `{clean_art}` not found in conversation `{clean_cid}`.",
        "",
        "### Available Artifacts in this Conversation:",
    ]
    for item in available:
        lines.append(f"- `{item['rel_path']}` ({item['size_bytes']} bytes, modified {item['modified_at']})")

    return "\n".join(lines)


def main() -> None:
    """Entry point for the antigravity-project-memory CLI."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
