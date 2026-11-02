"""
Unit and integration tests for antigravity-project-memory MCP server.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

# Ensure src/ is on sys.path for direct test runs
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from antigravity_project_memory.retrieval_engine import BM25Index, normalize_workspace_uri
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


import json
import tempfile

class TestServerMCPTools(unittest.TestCase):
    """Tests the 4 MCP tools using hermetic fixture data (runs cleanly locally and in CI)."""

    _temp_dir: tempfile.TemporaryDirectory | None = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = os.path.normpath(os.getcwd())
        cls.known_cid = "test-conv-uuid-101"
        cls.artifact_cid = cls.known_cid

        # Create isolated temporary fixture so tests pass in any CI runner or clean VM
        cls._temp_dir = tempfile.TemporaryDirectory()
        temp_root = cls._temp_dir.name
        fixture_db = os.path.join(temp_root, "conversation_summaries.db")
        fixture_brain = os.path.join(temp_root, "brain")

        # Initialize SQLite schema & test conversation
        conn = sqlite3.connect(fixture_db)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                conversation_id TEXT PRIMARY KEY,
                title TEXT,
                preview TEXT,
                step_count INTEGER,
                last_modified_time TEXT,
                workspace_uris TEXT,
                status TEXT
            )
            """
        )
        ws_uri = f"file:///{normalize_workspace_uri(cls.workspace).replace(os.sep, '/')}"
        conn.execute(
            "INSERT INTO conversation_summaries VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                cls.known_cid,
                "Test Architecture Conversation",
                "Chat memory antigravity architecture preview",
                4,
                "2026-11-02T12:00:00Z",
                json.dumps([ws_uri]),
                "active",
            ),
        )
        conn.commit()
        conn.close()

        # Create mock transcript and artifact
        conv_dir = os.path.join(fixture_brain, cls.known_cid)
        logs_dir = os.path.join(conv_dir, ".system_generated", "logs")
        os.makedirs(logs_dir, exist_ok=True)

        transcript_file = os.path.join(logs_dir, "transcript.jsonl")
        turn_events = [
            {
                "step_index": 0,
                "source": "USER_EXPLICIT",
                "type": "USER_INPUT",
                "content": "<USER_REQUEST>How should we configure chat memory antigravity?</USER_REQUEST>",
                "created_at": "2026-11-02T12:00:00Z",
            },
            {
                "step_index": 1,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "content": "Configure the FastMCP server using BM25 Okapi retrieval.",
                "created_at": "2026-11-02T12:00:05Z",
            },
        ]
        with open(transcript_file, "w", encoding="utf-8") as f:
            for event in turn_events:
                f.write(json.dumps(event) + "\n")

        # Create mock artifact
        artifact_file = os.path.join(conv_dir, "prompt_draft.md")
        with open(artifact_file, "w", encoding="utf-8") as f:
            f.write("# Teamwork Project Prompt\n\nInitial draft for multi-agent execution.\n")

        # Point engine to hermetic fixtures
        engine.db_path = fixture_db
        engine.brain_dir = fixture_brain
        engine.index = BM25Index()
        engine.indexed_workspace = ""

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._temp_dir:
            cls._temp_dir.cleanup()

    def test_1_list_project_conversations(self) -> None:
        result = list_project_conversations(self.workspace, limit=10)
        self.assertIn("# Antigravity Conversations", result)
        self.assertIn("Total Found", result)
        self.assertIn("| Conversation Title |", result)
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
        result = get_conversation_artifact(self.artifact_cid, "prompt_draft.md")
        self.assertIn("# Artifact: prompt_draft.md", result)
        self.assertIn("Teamwork Project Prompt", result)

        result_no_ext = get_conversation_artifact(self.artifact_cid, "prompt_draft")
        self.assertIn("# Artifact: prompt_draft.md", result_no_ext)

    def test_get_conversation_artifact_missing_shows_list(self) -> None:
        result = get_conversation_artifact(self.artifact_cid, "non_existent_file.md")
        self.assertIn("not found in conversation", result)
        self.assertIn("prompt_draft.md", result)

    def test_fastmcp_async_call_tool(self) -> None:
        """Verifies tool invocation via the FastMCP async protocol."""
        async def run_protocol() -> None:
            res, _ = await mcp.call_tool(
                "search_project_memory",
                {"query": "memory", "top_k": 2}
            )
            self.assertTrue(len(res) > 0)
            self.assertIn("Antigravity Project Memory Search Results", res[0].text)

            res_list, _ = await mcp.call_tool(
                "list_project_conversations",
                {"limit": 5}
            )
            self.assertTrue(len(res_list) > 0)
            self.assertIn("Antigravity Conversations", res_list[0].text)

        asyncio.run(run_protocol())


if __name__ == "__main__":
    unittest.main()
