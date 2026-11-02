"""
Unit and integration tests for antigravity-project-memory MCP server.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

# Ensure src/ is on sys.path for direct test runs
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from antigravity_project_memory.retrieval_engine import normalize_workspace_uri
from antigravity_project_memory.server import (
    engine,
    get_conversation_artifact,
    get_conversation_transcript,
    list_project_conversations,
    mcp,
    resolve_workspace,
    search_project_memory,
)


class TestWorkspaceResolution(unittest.TestCase):
    """Tests workspace normalization and fallback hierarchy."""

    def test_normalize_workspace_uri_cases(self) -> None:
        win_path = r"c:\Ai Readiness assesment"
        slash_path = r"C:/Ai Readiness assesment/"
        uri_lower = "file:///c%3A/Ai%20Readiness%20assesment"
        uri_upper = "file:///C:/Ai%20Readiness%20assesment"

        expected = os.path.normcase(os.path.normpath(r"c:\Ai Readiness assesment"))
        self.assertEqual(normalize_workspace_uri(win_path), expected)
        self.assertEqual(normalize_workspace_uri(slash_path), expected)
        self.assertEqual(normalize_workspace_uri(uri_lower), expected)
        self.assertEqual(normalize_workspace_uri(uri_upper), expected)

    def test_normalize_relative_path(self) -> None:
        curr = normalize_workspace_uri(".")
        expected = os.path.normcase(os.path.abspath("."))
        self.assertEqual(curr, expected)

    def test_resolve_workspace_hierarchy(self) -> None:
        # 1. Explicit argument wins
        explicit = resolve_workspace(r"C:\Custom\Workspace")
        self.assertEqual(explicit, normalize_workspace_uri(r"C:\Custom\Workspace"))

        # 2. ANTIGRAVITY_WORKSPACE env var
        with patch.dict(os.environ, {"ANTIGRAVITY_WORKSPACE": r"C:\Env\Workspace"}):
            env_resolved = resolve_workspace(None)
            self.assertEqual(env_resolved, normalize_workspace_uri(r"C:\Env\Workspace"))

        # 3. Fallback to os.getcwd()
        with patch.dict(os.environ, {}, clear=True):
            cwd_resolved = resolve_workspace(None)
            self.assertEqual(cwd_resolved, normalize_workspace_uri(os.getcwd()))


class TestServerMCPTools(unittest.TestCase):
    """Tests the 4 MCP tools against live Antigravity conversations."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = os.getcwd()
        cls.known_cid = "fea0b2e7-abe8-45cf-9924-9d06b7c77d24"
        cls.artifact_cid = "88e26d56-6df0-490d-81b5-13399a0f1088"

    def test_1_list_project_conversations(self) -> None:
        result = list_project_conversations(self.workspace, limit=10)
        self.assertIn("# Antigravity Conversations", result)
        self.assertIn("Total Found", result)
        self.assertIn("| Conversation Title |", result)
        # Should include known conversation
        self.assertIn(self.known_cid, result)

    def test_list_conversations_empty(self) -> None:
        result = list_project_conversations(r"C:\NonExistent\Directory_123456", limit=10)
        self.assertIn("No recorded conversations found", result)

    def test_2_search_project_memory(self) -> None:
        result = search_project_memory(
            query="chat memory antigravity",
            workspace_path=self.workspace,
            top_k=3,
        )
        self.assertIn("# Antigravity Project Memory Search Results", result)
        self.assertIn("Relevance Score", result)
        self.assertIn(self.known_cid, result)

    def test_search_project_memory_empty_query(self) -> None:
        result = search_project_memory(query="   ", workspace_path=self.workspace)
        self.assertTrue(result.startswith("Error:"))

    def test_3_get_conversation_transcript(self) -> None:
        result = get_conversation_transcript(self.known_cid, max_turns=5)
        self.assertIn("# Conversation Transcript:", result)
        self.assertIn("### User Request", result)
        self.assertIn("### Assistant Response", result)
        self.assertIn("## Turn ", result)

    def test_get_conversation_transcript_invalid_id(self) -> None:
        result = get_conversation_transcript("invalid-uuid-99999", max_turns=5)
        self.assertIn("Error: Transcript not found", result)

    def test_4_get_conversation_artifact(self) -> None:
        # Test exact match
        result = get_conversation_artifact(self.artifact_cid, "prompt_draft.md")
        self.assertIn("# Artifact: prompt_draft.md", result)
        self.assertIn("Teamwork Project Prompt", result)

        # Test match without extension
        result_no_ext = get_conversation_artifact(self.artifact_cid, "prompt_draft")
        self.assertIn("# Artifact: prompt_draft.md", result_no_ext)

    def test_get_conversation_artifact_missing_shows_list(self) -> None:
        result = get_conversation_artifact(self.artifact_cid, "non_existent_file.md")
        self.assertIn("not found in conversation", result)
        self.assertIn("prompt_draft.md", result)

    def test_fastmcp_async_call_tool(self) -> None:
        """Verifies tool invocation via the FastMCP async protocol."""
        async def run_protocol() -> None:
            # Test calling search_project_memory through FastMCP
            res, _ = await mcp.call_tool(
                "search_project_memory",
                {"query": "memory", "top_k": 2}
            )
            self.assertTrue(len(res) > 0)
            self.assertIn("Antigravity Project Memory Search Results", res[0].text)

            # Test calling list_project_conversations through FastMCP
            res_list, _ = await mcp.call_tool(
                "list_project_conversations",
                {"limit": 5}
            )
            self.assertTrue(len(res_list) > 0)
            self.assertIn("Antigravity Conversations", res_list[0].text)

        asyncio.run(run_protocol())


if __name__ == "__main__":
    unittest.main()
