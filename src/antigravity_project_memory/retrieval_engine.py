"""
Pure-Python Local Algorithmic Retrieval Engine for Antigravity Project Memory.

Zero external neural or LLM dependencies (no PyTorch, Hugging Face, CUDA, or external APIs).
Uses exact BM25 Okapi, code identifier / symbol boosting, exponential recency decay,
semantic dialogue turn parsing, markdown artifact chunking, and passage stitching.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import urllib.parse
import urllib.request


# =====================================================================
# Constants & Regular Expressions
# =====================================================================

DEFAULT_DB_PATH = os.environ.get(
    "ANTIGRAVITY_DB_PATH",
    os.path.expanduser(r"~\.gemini\antigravity\conversation_summaries.db")
)
DEFAULT_BRAIN_DIR = os.environ.get(
    "ANTIGRAVITY_BRAIN_DIR",
    os.path.expanduser(r"~\.gemini\antigravity\brain")
)

# Standard English stopwords for Information Retrieval (preserving technical semantics)
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had",
    "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "with", "would", "you",
    "your", "yours", "yourself", "yourselves"
}

# Regex for file paths with extensions or path separators
RE_FILE_PATH = re.compile(
    r"(?:[a-zA-Z]:[\\/]|/|[\w\-.]+[\\/])[\w\-./\\]+\.[a-zA-Z0-9_]{1,10}\b|"
    r"\b[\w\-.]+\.(?:py|json|md|txt|html|pdf|ts|js|sql|sh|yaml|yml|xml|docx|pptx|java|go|rs|c|cpp|h)\b",
    re.IGNORECASE
)

# Regex for camelCase / PascalCase identifiers
RE_CAMEL_CASE = re.compile(
    r"\b(?:[a-z]+[A-Z][a-zA-Z0-9]*|[A-Z][a-z0-9]+[A-Z][a-zA-Z0-9]*)\b"
)

# Regex for snake_case identifiers
RE_SNAKE_CASE = re.compile(
    r"\b[a-zA-Z0-9]+(?:_[a-zA-Z0-9]+)+\b"
)

# Regex for UPPER_CASE constants
RE_UPPER_CONST = re.compile(
    r"\b[A-Z][A-Z0-9_]{2,}\b"
)

# Splitting camelCase into subwords: e.g. "ProjectRetrievalEngine" -> ["Project", "Retrieval", "Engine"]
RE_CAMEL_SPLIT = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z0-9]|\b)")

# UI Marker cleaning regexes
RE_USER_REQUEST_TAG = re.compile(r"<USER_REQUEST>(.*?)</USER_REQUEST>", re.DOTALL | re.IGNORECASE)
RE_ADDITIONAL_METADATA_TAG = re.compile(r"<ADDITIONAL_METADATA>.*?</ADDITIONAL_METADATA>", re.DOTALL | re.IGNORECASE)
RE_SETTINGS_CHANGE_TAG = re.compile(r"<USER_SETTINGS_CHANGE>.*?</USER_SETTINGS_CHANGE>", re.DOTALL | re.IGNORECASE)
RE_SYSTEM_MSG_TAG = re.compile(r"<SYSTEM_MESSAGE>.*?</SYSTEM_MESSAGE>", re.DOTALL | re.IGNORECASE)
RE_THOUGHT_TAG = re.compile(r"<thought>.*?</thought>", re.DOTALL | re.IGNORECASE)
RE_MENTION_TAG = re.compile(r"@\[(.*?)\]")


# =====================================================================
# Data Structures
# =====================================================================

@dataclass
class DocumentChunk:
    """Represents a discrete semantic chunk of conversation dialogue or artifact."""
    chunk_id: str                      # Globally unique ID: "{conversation_id}:{type}:{index}"
    conversation_id: str               # Antigravity conversation UUID
    workspace_uri: str                 # Workspace URI or normalized path
    chunk_type: str                    # "dialogue" | "artifact"
    chunk_index: int                   # Sequence position within document (0, 1, 2...)
    end_chunk_index: int               # Same as chunk_index initially; grows upon passage stitching
    title: str                         # Conversation title or artifact filename/header
    content: str                       # Clean textual content
    source: str                        # File path on disk (transcript.jsonl or artifact.md)
    created_at: str                    # ISO 8601 timestamp string
    metadata: Dict[str, Any] = field(default_factory=dict) # Turn details, tools used, header breadcrumbs
    score: float = 0.0                 # Final composite retrieval score
    raw_bm25: float = 0.0              # Base BM25 Okapi relevance score
    symbol_boost: float = 1.0          # Multiplier from code identifier & file path matching
    recency_factor: float = 1.0        # Multiplier from exponential/linear age decay
    chunk_ids: List[str] = field(default_factory=list) # List of original chunk IDs if stitched

    def __post_init__(self) -> None:
        if not self.chunk_ids:
            self.chunk_ids = [self.chunk_id]
        if self.end_chunk_index < self.chunk_index:
            self.end_chunk_index = self.chunk_index

    def clone(self) -> DocumentChunk:
        """Create a deep copy of the chunk for stitching without mutating original."""
        return DocumentChunk(
            chunk_id=self.chunk_id,
            conversation_id=self.conversation_id,
            workspace_uri=self.workspace_uri,
            chunk_type=self.chunk_type,
            chunk_index=self.chunk_index,
            end_chunk_index=self.end_chunk_index,
            title=self.title,
            content=self.content,
            source=self.source,
            created_at=self.created_at,
            metadata=dict(self.metadata),
            score=self.score,
            raw_bm25=self.raw_bm25,
            symbol_boost=self.symbol_boost,
            recency_factor=self.recency_factor,
            chunk_ids=list(self.chunk_ids),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk into a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class SearchResult:
    """Encapsulates retrieval results, query analytics, and performance benchmarks."""
    passages: List[DocumentChunk]
    total_found: int
    query: str
    query_tokens: List[str]
    query_symbols: List[str]
    timings_ms: Dict[str, float]
    workspace_matched: str
    conversations_indexed: int
    total_chunks_indexed: int


# =====================================================================
# Utilities: Normalization, Extraction, and Recency
# =====================================================================

def normalize_workspace_uri(uri_or_path: str) -> str:
    """
    Normalizes workspace URIs and local file paths to a canonical lowercase path.
    Handles 'file:///c%3A/foo', 'file:///C:/foo', 'c:\\foo', 'C:/foo', relative paths, and current dir.
    """
    if not uri_or_path:
        return ""
    target = uri_or_path.strip()
    if target.startswith("file://"):
        target = target[7:]
    target = urllib.parse.unquote(target)
    
    # Handle leading slash on Windows (e.g. /c:/path -> c:/path)
    if len(target) >= 3 and target[0] in ("/\\") and target[2] == ":":
        target = target[1:]
    
    # Resolve relative paths (e.g. '.', './sub') to absolute path
    if not os.path.isabs(target) and not (len(target) >= 2 and target[1] == ":"):
        target = os.path.abspath(target)

    norm = os.path.normpath(target)
    return os.path.normcase(norm)


def parse_iso_timestamp(ts_str: Optional[str]) -> datetime:
    """Parse an ISO 8601 timestamp string into a timezone-aware UTC datetime."""
    if not ts_str:
        return datetime.now(timezone.utc)
    ts_clean = ts_str.strip()
    try:
        if ts_clean.endswith("Z"):
            ts_clean = ts_clean[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts_clean)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def compute_recency_factor(
    doc_timestamp: str,
    reference_time: Optional[datetime] = None,
    half_life_days: float = 30.0,
    floor: float = 0.5,
    mode: str = "exponential"
) -> float:
    """
    Computes a recency decay multiplier in [floor, 1.0].
    
    Formula (Exponential):
        delta_t_days = max(0, (ref_time - doc_time).days)
        factor = floor + (1.0 - floor) * 2^(-delta_t_days / half_life_days)
    
    Formula (Linear):
        factor = max(floor, 1.0 - (delta_t_days / (half_life_days * 3)) * (1.0 - floor))
    """
    if mode == "none" or half_life_days <= 0:
        return 1.0

    ref_dt = reference_time or datetime.now(timezone.utc)
    doc_dt = parse_iso_timestamp(doc_timestamp)
    delta_seconds = max(0.0, (ref_dt - doc_dt).total_seconds())
    delta_days = delta_seconds / 86400.0

    if mode == "exponential":
        decay = math.pow(2.0, -delta_days / half_life_days)
        return float(floor + (1.0 - floor) * decay)
    elif mode == "linear":
        max_days = half_life_days * 3.0
        ratio = min(1.0, delta_days / max_days)
        return float(1.0 - ratio * (1.0 - floor))
    else:
        return 1.0


def extract_code_symbols(text: str) -> Dict[str, Set[str]]:
    """
    Extracts code identifiers, file paths, snake_case, camelCase, and UPPER_CASE constants.
    """
    paths = set(RE_FILE_PATH.findall(text))
    camels = set(RE_CAMEL_CASE.findall(text))
    snakes = set(RE_SNAKE_CASE.findall(text))
    uppers = set(RE_UPPER_CONST.findall(text))
    
    # Clean file paths if needed
    cleaned_paths = {p.strip().replace("\\", "/") for p in paths if p.strip()}

    return {
        "paths": cleaned_paths,
        "camelCase": camels,
        "snake_case": snakes,
        "UPPER_CONST": uppers,
        "all_symbols": cleaned_paths | camels | snakes | uppers
    }


# =====================================================================
# BM25 Okapi Algorithm & Index
# =====================================================================

class BM25Index:
    """
    Pure-Python implementation of BM25 Okapi with Lucene-style non-negative IDF,
    subword/identifier tokenization, and code symbol awareness.
    
    Formula:
        IDF(q) = ln( (N - n(q) + 0.5) / (n(q) + 0.5) + 1.0 )
        TF_norm(q, D) = (f(q, D) * (k1 + 1)) / (f(q, D) + k1 * (1 - b + b * (|D| / avgdl)))
        BM25(D, Q) = sum_{q in Q} IDF(q) * TF_norm(q, D)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents: List[DocumentChunk] = []
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0
        self.doc_count: int = 0
        
        # Inverted index: term -> list of (doc_index, term_frequency)
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = collections.defaultdict(list)
        # Term document frequency: term -> number of documents containing term
        self.doc_frequencies: Dict[str, int] = collections.defaultdict(int)
        # Precomputed IDF
        self.idf_cache: Dict[str, float] = {}

    def tokenize(self, text: str, filter_stopwords: bool = True) -> List[str]:
        """
        Tokenizes text into normalized lexical units:
        - Splits on whitespace and non-identifier punctuation
        - Preserves and decomposes snake_case and camelCase
        - Preserves file names and extensions (e.g. 'retrieval_engine.py')
        - Lowercases all tokens
        - Optionally filters standard stopwords
        """
        if not text:
            return []

        raw_tokens = re.findall(r"[A-Za-z0-9_.\-/\\]+", text)
        processed: List[str] = []

        for raw in raw_tokens:
            cleaned = raw.strip(".,;:!?()[]{}'\"`~")
            if not cleaned:
                continue

            # Check if token is a file path or file name
            if "/" in cleaned or "\\" in cleaned or ("." in cleaned and not cleaned.endswith(".")):
                norm_path = cleaned.replace("\\", "/").lower()
                processed.append(norm_path)
                base = os.path.basename(norm_path)
                if base != norm_path:
                    processed.append(base)
                # Also index extension and subparts
                parts = re.split(r"[/._\-]+", norm_path)
                for p in parts:
                    if len(p) >= 2:
                        processed.append(p)
                continue

            # Check snake_case
            if "_" in cleaned:
                processed.append(cleaned.lower())
                sub_parts = cleaned.split("_")
                for sp in sub_parts:
                    if len(sp) >= 2:
                        processed.append(sp.lower())
                continue

            # Check camelCase / PascalCase
            camels = RE_CAMEL_SPLIT.findall(cleaned)
            if len(camels) > 1:
                processed.append(cleaned.lower())
                for c in camels:
                    if len(c) >= 2:
                        processed.append(c.lower())
                continue

            # Regular word/number
            processed.append(cleaned.lower())

        if filter_stopwords:
            filtered = [t for t in processed if t not in STOPWORDS]
            # If all tokens were filtered out, preserve original to avoid empty queries
            return filtered if filtered else processed

        return processed

    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Indexes a batch of DocumentChunk instances."""
        start_idx = len(self.documents)
        self.documents.extend(chunks)

        for offset, chunk in enumerate(chunks):
            doc_idx = start_idx + offset
            tokens = self.tokenize(chunk.content + " " + chunk.title)
            self.doc_lengths.append(len(tokens))

            # Count term frequencies in this document
            tf_counter = collections.Counter(tokens)
            for term, count in tf_counter.items():
                self.inverted_index[term].append((doc_idx, count))
                self.doc_frequencies[term] += 1

        self.doc_count = len(self.documents)
        total_len = sum(self.doc_lengths)
        self.avgdl = (total_len / self.doc_count) if self.doc_count > 0 else 0.0
        
        # Reset IDF cache
        self.idf_cache.clear()
        for term, df in self.doc_frequencies.items():
            # Robertson-Spärck Jones Okapi formula with +1.0 Lucene non-negative smoothing
            self.idf_cache[term] = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)

    def score_document(self, doc_idx: int, query_tokens: List[str]) -> float:
        """Calculate raw BM25 Okapi score for a specific document."""
        if doc_idx >= self.doc_count or self.avgdl == 0.0:
            return 0.0

        score = 0.0
        doc_len = self.doc_lengths[doc_idx]
        denom_part = self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))

        # Check terms in doc
        for q in query_tokens:
            idf = self.idf_cache.get(q, 0.0)
            if idf <= 0.0:
                continue

            # Look up TF
            tf = 0
            for d_id, freq in self.inverted_index.get(q, []):
                if d_id == doc_idx:
                    tf = freq
                    break

            if tf > 0:
                tf_norm = (tf * (self.k1 + 1.0)) / (tf + denom_part)
                score += idf * tf_norm

        return score

    def search_candidates(self, query: str, top_n: int = 50) -> List[Tuple[int, float]]:
        """
        Fast candidate retrieval using inverted index.
        Returns list of (doc_idx, raw_bm25_score) sorted descending by score.
        """
        query_tokens = self.tokenize(query)
        if not query_tokens or self.doc_count == 0:
            return []

        doc_scores: Dict[int, float] = collections.defaultdict(float)
        
        for q in query_tokens:
            idf = self.idf_cache.get(q, 0.0)
            if idf <= 0.0:
                continue

            postings = self.inverted_index.get(q, [])
            for doc_idx, tf in postings:
                doc_len = self.doc_lengths[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                tf_norm = (tf * (self.k1 + 1.0)) / denom
                doc_scores[doc_idx] += idf * tf_norm

        if not doc_scores:
            return []

        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]


# =====================================================================
# Code Symbol & Path Boosting
# =====================================================================

def calculate_symbol_boost(
    query: str,
    chunk: DocumentChunk,
    query_symbols: Dict[str, Set[str]]
) -> float:
    """
    Computes an empirical boost multiplier (1.0 to 3.0) based on code symbols:
    - Path matches (+0.60): Exact file path or filename found in chunk
    - Code identifiers (+0.35 each): camelCase, snake_case, UPPER_CONST matches
    - Exact query phrase (+0.45): Exact multi-word query in chunk content
    - Title / Header match (+0.30): Query symbols present in section title/header
    """
    boost = 1.0
    content_lower = chunk.content.lower()
    title_lower = chunk.title.lower()
    header_path = str(chunk.metadata.get("header_path", "")).lower()

    # 1. File Path & Filename Match
    for path in query_symbols.get("paths", set()):
        path_lower = path.lower()
        base_name = os.path.basename(path_lower)
        if path_lower in content_lower or base_name in content_lower:
            boost += 0.60
            break
        elif base_name in title_lower or base_name in header_path:
            boost += 0.40
            break

    # 2. Code Identifier Matches
    matched_idents = 0
    for ident in query_symbols.get("all_symbols", set()):
        ident_lower = ident.lower()
        if len(ident) >= 3 and (ident_lower in content_lower or ident in chunk.content):
            matched_idents += 1
            boost += 0.35
            if matched_idents >= 3:
                break

    # 3. Exact Multi-word Phrase Match
    clean_q = query.strip().lower()
    words = clean_q.split()
    if len(words) >= 2:
        if clean_q in content_lower:
            boost += 0.45
        elif len(words) >= 3:
            # Check bigram subphrases
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                if bigram in content_lower and bigram not in STOPWORDS:
                    boost += 0.15
                    break

    # 4. Title / Header Match
    for word in words:
        if len(word) >= 3 and word not in STOPWORDS:
            if word in title_lower or word in header_path:
                boost += 0.15
                break

    # Cap boost multiplier between 1.0 and 3.0
    return min(3.0, boost)


# =====================================================================
# Passage Stitching Algorithm
# =====================================================================

def stitch_passages(passages: List[DocumentChunk], budget: int = 2500) -> List[DocumentChunk]:
    """
    Stitches adjacent chunks belonging to the same conversation / artifact document into
    a single cohesive passage bounded by a character budget, maintaining chronological order.
    """
    if not passages:
        return []

    # Group passages by document identity:
    # Dialogue turns belong to conversation_id; Artifacts belong to conversation_id + file_name
    groups: Dict[str, List[DocumentChunk]] = {}
    for passage in passages:
        doc_key = f"{passage.conversation_id}::{passage.source}"
        groups.setdefault(doc_key, []).append(passage)

    output: List[DocumentChunk] = []

    for doc_key, group in groups.items():
        # Sort sequentially by chunk_index
        group.sort(key=lambda p: p.chunk_index)
        
        current = group[0].clone()
        for following in group[1:]:
            combined_len = len(current.content) + 2 + len(following.content)
            # Check adjacency and budget constraint
            if following.chunk_index == current.end_chunk_index + 1 and combined_len <= budget:
                current.content = current.content + "\n\n" + following.content
                current.end_chunk_index = following.end_chunk_index
                current.chunk_ids.extend(following.chunk_ids)
                current.score = max(current.score, following.score)
                current.raw_bm25 = max(current.raw_bm25, following.raw_bm25)
                current.symbol_boost = max(current.symbol_boost, following.symbol_boost)
                current.recency_factor = max(current.recency_factor, following.recency_factor)
            else:
                output.append(current)
                current = following.clone()
        output.append(current)

    # Sort final stitched passages by descending composite relevance score
    return sorted(output, key=lambda p: p.score, reverse=True)


# =====================================================================
# Antigravity Transcript & Artifact Parser
# =====================================================================

class AntigravityTranscriptParser:
    """
    Parses raw Antigravity transcripts (transcript.jsonl) into structured semantic dialogue units
    and parses brain artifact files (*.md, *.txt) into indexed sections.
    """

    def __init__(self, max_chunk_chars: int = 1800, overlap_chars: int = 200) -> None:
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars

    @staticmethod
    def clean_user_text(raw_text: str) -> str:
        """Strips Antigravity UI XML wrappers, metadata blocks, and cleans mentions."""
        if not raw_text:
            return ""

        # Check for explicit <USER_REQUEST> wrapper
        m = RE_USER_REQUEST_TAG.search(raw_text)
        if m:
            text = m.group(1)
        else:
            text = raw_text

        # Remove UI metadata tags
        text = RE_ADDITIONAL_METADATA_TAG.sub("", text)
        text = RE_SETTINGS_CHANGE_TAG.sub("", text)
        text = RE_SYSTEM_MSG_TAG.sub("", text)

        # Clean @[item] mentions to bare item name
        text = RE_MENTION_TAG.sub(r"\1", text)

        return text.strip()

    @staticmethod
    def clean_model_text(raw_text: str) -> str:
        """Cleans internal thought tags or system messages from model responses."""
        if not raw_text:
            return ""
        text = RE_THOUGHT_TAG.sub("", raw_text)
        text = RE_SYSTEM_MSG_TAG.sub("", text)
        return text.strip()

    @staticmethod
    def summarize_tool_calls(tool_calls: List[Dict[str, Any]]) -> str:
        """
        Generates a concise semantic summary of tool calls executed during a turn
        without polluting the index with redundant command-line stdout.
        """
        if not tool_calls:
            return ""

        summaries = []
        for tc in tool_calls:
            name = tc.get("tool_name") or tc.get("name") or "tool"
            args = tc.get("args") or {}
            
            # Extract most informative argument
            arg_summary = ""
            if "CommandLine" in args:
                cmd = str(args["CommandLine"]).strip().strip('"').strip("'")
                summary_label = args.get("toolSummary") or args.get("toolAction") or cmd[:60]
                arg_summary = f"{summary_label} (`{cmd[:80]}`)"
            elif "FilePath" in args or "file_path" in args or "Path" in args or "path" in args:
                p = args.get("FilePath") or args.get("file_path") or args.get("Path") or args.get("path")
                arg_summary = f"path=`{p}`"
            elif "Url" in args or "url" in args:
                u = args.get("Url") or args.get("url")
                arg_summary = f"url=`{u}`"
            elif "query" in args or "Query" in args:
                q = args.get("query") or args.get("Query")
                arg_summary = f"query=`{q}`"
            else:
                arg_summary = ", ".join(f"{k}={str(v)[:40]}" for k, v in list(args.items())[:2])

            summaries.append(f"- `{name}`: {arg_summary}")

        return "\n".join(summaries)

    def parse_transcript(
        self,
        transcript_path: str,
        conversation_id: str,
        workspace_uri: str,
        conv_title: str,
        last_modified: str
    ) -> List[DocumentChunk]:
        """
        Parses transcript.jsonl into ordered semantic dialogue units (User Request + Tools + Assistant Answer).
        """
        if not os.path.exists(transcript_path):
            return []

        steps = []
        try:
            with open(transcript_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        steps.append(json.loads(line))
        except Exception:
            return []

        if not steps:
            return []

        chunks: List[DocumentChunk] = []
        current_turn_index = 0
        i = 0
        n = len(steps)

        while i < n:
            step = steps[i]
            s_type = step.get("type")
            s_source = step.get("source")

            if s_type == "USER_INPUT" or s_source == "USER_EXPLICIT":
                user_step_index = step.get("step_index", i)
                raw_user_content = step.get("content") or ""
                cleaned_user = self.clean_user_text(raw_user_content)
                user_time = step.get("created_at") or last_modified

                # Collect tool calls and assistant response until next USER_INPUT
                j = i + 1
                tool_calls_in_turn = []
                model_responses = []
                tools_used_set = set()

                while j < n and steps[j].get("type") != "USER_INPUT" and steps[j].get("source") != "USER_EXPLICIT":
                    s_next = steps[j]
                    # Check tool calls
                    if s_next.get("tool_calls"):
                        for tc in s_next["tool_calls"]:
                            tool_calls_in_turn.append(tc)
                            t_name = tc.get("tool_name") or tc.get("name")
                            if t_name:
                                tools_used_set.add(t_name)

                    # Check assistant response
                    if s_next.get("type") == "PLANNER_RESPONSE" and s_next.get("content"):
                        clean_resp = self.clean_model_text(s_next["content"])
                        if clean_resp:
                            model_responses.append(clean_resp)

                    j += 1

                # Construct dialogue unit
                assistant_text = "\n\n".join(model_responses).strip()
                tool_summary = self.summarize_tool_calls(tool_calls_in_turn)

                turn_body_parts = []
                turn_body_parts.append(f"### User Request\n{cleaned_user}")
                if tool_summary:
                    turn_body_parts.append(f"### Actions Executed\n{tool_summary}")
                if assistant_text:
                    turn_body_parts.append(f"### Assistant Response\n{assistant_text}")

                full_turn_text = "\n\n".join(turn_body_parts)

                # Chunk if dialogue turn exceeds maximum character budget
                if len(full_turn_text) <= self.max_chunk_chars:
                    chunk = DocumentChunk(
                        chunk_id=f"{conversation_id}:dialogue:{len(chunks)}",
                        conversation_id=conversation_id,
                        workspace_uri=workspace_uri,
                        chunk_type="dialogue",
                        chunk_index=len(chunks),
                        end_chunk_index=len(chunks),
                        title=f"{conv_title} - Turn {current_turn_index + 1}",
                        content=full_turn_text,
                        source=transcript_path,
                        created_at=user_time,
                        metadata={
                            "turn_index": current_turn_index,
                            "user_step": user_step_index,
                            "tools_used": list(tools_used_set),
                            "tool_calls_count": len(tool_calls_in_turn),
                        }
                    )
                    chunks.append(chunk)
                else:
                    # Split into sub-chunks while maintaining turn header and continuity
                    sub_parts = self._split_text_into_chunks(full_turn_text, self.max_chunk_chars, self.overlap_chars)
                    for part_idx, part_text in enumerate(sub_parts):
                        chunk = DocumentChunk(
                            chunk_id=f"{conversation_id}:dialogue:{len(chunks)}",
                            conversation_id=conversation_id,
                            workspace_uri=workspace_uri,
                            chunk_type="dialogue",
                            chunk_index=len(chunks),
                            end_chunk_index=len(chunks),
                            title=f"{conv_title} - Turn {current_turn_index + 1} (Part {part_idx + 1})",
                            content=part_text,
                            source=transcript_path,
                            created_at=user_time,
                            metadata={
                                "turn_index": current_turn_index,
                                "part_index": part_idx,
                                "user_step": user_step_index,
                                "tools_used": list(tools_used_set),
                            }
                        )
                        chunks.append(chunk)

                current_turn_index += 1
                i = j
            else:
                i += 1

        return chunks

    def parse_transcript_turns_raw(
        self,
        transcript_path: str,
        last_modified: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Extracts high-level dialogue turns without splitting into chunk length budgets.
        Ideal for get_conversation_transcript tool.
        """
        if not os.path.exists(transcript_path):
            return []

        steps = []
        try:
            with open(transcript_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        steps.append(json.loads(line))
        except Exception:
            return []

        turns: List[Dict[str, Any]] = []
        current_turn_index = 0
        i = 0
        n = len(steps)

        while i < n:
            step = steps[i]
            s_type = step.get("type")
            s_source = step.get("source")

            if s_type == "USER_INPUT" or s_source == "USER_EXPLICIT":
                raw_user = step.get("content") or ""
                cleaned_user = self.clean_user_text(raw_user)
                ts = step.get("created_at") or last_modified

                j = i + 1
                tool_calls_in_turn = []
                model_responses = []

                while j < n and steps[j].get("type") != "USER_INPUT" and steps[j].get("source") != "USER_EXPLICIT":
                    s_next = steps[j]
                    if s_next.get("tool_calls"):
                        for tc in s_next["tool_calls"]:
                            tool_calls_in_turn.append(tc)
                    if s_next.get("type") == "PLANNER_RESPONSE" and s_next.get("content"):
                        clean_resp = self.clean_model_text(s_next["content"])
                        if clean_resp:
                            model_responses.append(clean_resp)
                    j += 1

                assistant_text = "\n\n".join(model_responses).strip()
                tool_summary = self.summarize_tool_calls(tool_calls_in_turn)

                turns.append({
                    "turn_index": current_turn_index + 1,
                    "created_at": ts,
                    "user_text": cleaned_user,
                    "tool_summary": tool_summary,
                    "tool_calls_count": len(tool_calls_in_turn),
                    "assistant_text": assistant_text,
                })

                current_turn_index += 1
                i = j
            else:
                i += 1

        return turns

    def parse_artifact(
        self,
        artifact_path: str,
        conversation_id: str,
        workspace_uri: str,
        conv_title: str,
        last_modified: str
    ) -> List[DocumentChunk]:
        """
        Parses markdown and text artifacts into structured chunks using header hierarchy.
        """
        if not os.path.exists(artifact_path):
            return []

        try:
            with open(artifact_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            return []

        if not content.strip():
            return []

        file_name = os.path.basename(artifact_path)
        chunks: List[DocumentChunk] = []

        # Split markdown by top-level headers (# or ##)
        header_sections = self._split_markdown_headers(content)

        for sec_idx, (breadcrumbs, sec_text) in enumerate(header_sections):
            header_title = f"{file_name} > {' > '.join(breadcrumbs)}" if breadcrumbs else file_name
            
            if len(sec_text) <= self.max_chunk_chars:
                chunk = DocumentChunk(
                    chunk_id=f"{conversation_id}:artifact:{file_name}:{len(chunks)}",
                    conversation_id=conversation_id,
                    workspace_uri=workspace_uri,
                    chunk_type="artifact",
                    chunk_index=len(chunks),
                    end_chunk_index=len(chunks),
                    title=f"[Artifact] {header_title}",
                    content=f"### Artifact: {header_title}\n\n{sec_text}",
                    source=artifact_path,
                    created_at=last_modified,
                    metadata={
                        "file_name": file_name,
                        "header_path": " > ".join(breadcrumbs),
                        "section_index": sec_idx,
                    }
                )
                chunks.append(chunk)
            else:
                sub_parts = self._split_text_into_chunks(sec_text, self.max_chunk_chars, self.overlap_chars)
                for part_idx, p_text in enumerate(sub_parts):
                    chunk = DocumentChunk(
                        chunk_id=f"{conversation_id}:artifact:{file_name}:{len(chunks)}",
                        conversation_id=conversation_id,
                        workspace_uri=workspace_uri,
                        chunk_type="artifact",
                        chunk_index=len(chunks),
                        end_chunk_index=len(chunks),
                        title=f"[Artifact] {header_title} (Part {part_idx + 1})",
                        content=f"### Artifact: {header_title} (Part {part_idx + 1})\n\n{p_text}",
                        source=artifact_path,
                        created_at=last_modified,
                        metadata={
                            "file_name": file_name,
                            "header_path": " > ".join(breadcrumbs),
                            "part_index": part_idx,
                        }
                    )
                    chunks.append(chunk)

        return chunks

    def _split_markdown_headers(self, text: str) -> List[Tuple[List[str], str]]:
        """Splits markdown document into sections preserving header breadcrumbs."""
        lines = text.splitlines()
        sections: List[Tuple[List[str], str]] = []
        current_headers: List[str] = []
        current_lines: List[str] = []

        header_regex = re.compile(r"^(#{1,4})\s+(.+)$")

        for line in lines:
            m = header_regex.match(line)
            if m:
                # Flush existing accumulated lines
                if current_lines:
                    sec_content = "\n".join(current_lines).strip()
                    if sec_content:
                        sections.append((list(current_headers), sec_content))
                    current_lines = []

                level = len(m.group(1))
                title = m.group(2).strip()
                # Maintain breadcrumb stack depth
                current_headers = current_headers[: level - 1]
                current_headers.append(title)
                current_lines.append(line)
            else:
                current_lines.append(line)

        if current_lines:
            sec_content = "\n".join(current_lines).strip()
            if sec_content:
                sections.append((list(current_headers), sec_content))

        if not sections:
            sections.append(([], text.strip()))

        return sections

    def _split_text_into_chunks(self, text: str, max_chars: int, overlap_chars: int) -> List[str]:
        """Splits text into paragraph-aware chunks bounded by max_chars with overlap."""
        paragraphs = text.split("\n\n")
        chunks = []
        current_part: List[str] = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if current_len + len(para) + 2 > max_chars and current_part:
                chunks.append("\n\n".join(current_part))
                # Retain tail paragraph for overlap if applicable
                tail = current_part[-1] if len(current_part[-1]) <= overlap_chars else ""
                current_part = [tail, para] if tail else [para]
                current_len = sum(len(p) for p in current_part) + (len(current_part) - 1) * 2
            else:
                current_part.append(para)
                current_len += len(para) + 2

        if current_part:
            chunks.append("\n\n".join(current_part))

        return chunks if chunks else [text[:max_chars]]


# =====================================================================
# Main Retrieval Engine Interface
# =====================================================================

class ProjectRetrievalEngine:
    """
    Production-grade local algorithmic retrieval engine for Antigravity workspaces.
    Discovers conversations, indexes transcripts & artifacts, and performs
    scored, boosted, recency-decayed, and stitched context retrieval.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        brain_dir: str = DEFAULT_BRAIN_DIR,
        k1: float = 1.5,
        b: float = 0.75,
        recency_half_life_days: float = 30.0,
        recency_floor: float = 0.5,
        recency_mode: str = "exponential"
    ) -> None:
        self.db_path = os.path.expanduser(db_path)
        self.brain_dir = os.path.expanduser(brain_dir)
        self.k1 = k1
        self.b = b
        self.recency_half_life_days = recency_half_life_days
        self.recency_floor = recency_floor
        self.recency_mode = recency_mode

        self.index = BM25Index(k1=self.k1, b=self.b)
        self.parser = AntigravityTranscriptParser()
        self.indexed_workspace: str = ""
        self.indexed_conversations: List[Dict[str, Any]] = []

    def get_database_connection(self) -> sqlite3.Connection:
        """Opens a read-only connection to conversation_summaries.db."""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found at {self.db_path}")
        uri = f"file:{urllib.request.pathname2url(self.db_path)}?mode=ro"
        return sqlite3.connect(uri, uri=True)

    def discover_workspace_conversations(self, target_workspace: str) -> List[Dict[str, Any]]:
        """
        Queries conversation_summaries.db for all conversations matching target workspace.
        """
        canonical_target = normalize_workspace_uri(target_workspace)
        if not os.path.exists(self.db_path):
            return []

        conn = self.get_database_connection()
        c = conn.cursor()

        results = []
        query = (
            "SELECT conversation_id, title, preview, workspace_uris, "
            "step_count, last_modified_time, status FROM conversation_summaries "
            "ORDER BY last_modified_time DESC"
        )

        for row in c.execute(query):
            cid, title, preview, uris_str, step_count, last_mod, status = row
            try:
                uris = json.loads(uris_str)
            except Exception:
                uris = []

            for u in uris:
                if normalize_workspace_uri(u) == canonical_target:
                    results.append({
                        "conversation_id": cid,
                        "title": title or "Untitled Conversation",
                        "preview": preview or "",
                        "workspace_uri": u,
                        "step_count": step_count or 0,
                        "last_modified_time": last_mod or "",
                        "status": status or "",
                    })
                    break

        conn.close()
        return results

    def get_conversation_metadata(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Fetches metadata for a specific conversation by ID."""
        if not os.path.exists(self.db_path):
            return None

        conn = self.get_database_connection()
        c = conn.cursor()
        c.execute(
            "SELECT conversation_id, title, preview, workspace_uris, "
            "step_count, last_modified_time, status FROM conversation_summaries "
            "WHERE conversation_id = ? LIMIT 1",
            (conversation_id,)
        )
        row = c.fetchone()
        conn.close()

        if not row:
            return None

        cid, title, preview, uris_str, step_count, last_mod, status = row
        try:
            uris = json.loads(uris_str)
        except Exception:
            uris = []

        return {
            "conversation_id": cid,
            "title": title or "Untitled Conversation",
            "preview": preview or "",
            "workspace_uris": uris,
            "step_count": step_count or 0,
            "last_modified_time": last_mod or "",
            "status": status or "",
        }

    def load_workspace(
        self,
        workspace_path_or_uri: str,
        max_conversations: Optional[int] = None
    ) -> int:
        """
        Loads and indexes all conversation dialogue units and artifacts for the workspace.
        Returns total number of chunks indexed.
        """
        t0 = time.monotonic()
        convs = self.discover_workspace_conversations(workspace_path_or_uri)
        if max_conversations:
            convs = convs[:max_conversations]

        self.indexed_workspace = normalize_workspace_uri(workspace_path_or_uri)
        self.indexed_conversations = convs

        all_chunks: List[DocumentChunk] = []

        for conv in convs:
            cid = conv["conversation_id"]
            title = conv["title"]
            w_uri = conv["workspace_uri"]
            last_mod = conv["last_modified_time"]
            conv_brain_dir = os.path.join(self.brain_dir, cid)

            # 1. Parse dialogue transcript
            transcript_path = os.path.join(
                conv_brain_dir, ".system_generated", "logs", "transcript.jsonl"
            )
            if os.path.exists(transcript_path):
                dialogue_chunks = self.parser.parse_transcript(
                    transcript_path=transcript_path,
                    conversation_id=cid,
                    workspace_uri=w_uri,
                    conv_title=title,
                    last_modified=last_mod
                )
                all_chunks.extend(dialogue_chunks)

            # 2. Parse artifacts (.md and .txt directly in brain/<cid> or artifacts/)
            if os.path.isdir(conv_brain_dir):
                for root, dirs, files in os.walk(conv_brain_dir):
                    # Skip internal system logs
                    if ".system_generated" in root or ".user_uploaded" in root:
                        continue
                    for f in files:
                        if f.endswith(".md") or f.endswith(".txt"):
                            art_path = os.path.join(root, f)
                            art_chunks = self.parser.parse_artifact(
                                artifact_path=art_path,
                                conversation_id=cid,
                                workspace_uri=w_uri,
                                conv_title=title,
                                last_modified=last_mod
                            )
                            all_chunks.extend(art_chunks)

        # Build BM25 index
        self.index = BM25Index(k1=self.k1, b=self.b)
        self.index.add_documents(all_chunks)

        return len(all_chunks)

    def search(
        self,
        query: str,
        workspace_path: Optional[str] = None,
        top_k: int = 10,
        stitch_budget: int = 2500,
        context_budget: int = 8000,
        min_score: float = 0.05
    ) -> SearchResult:
        """
        Executes hybrid retrieval:
        1. Ensures the target workspace is loaded/indexed
        2. BM25 Okapi candidate selection
        3. Symbol & Code Identifier Boosting
        4. Recency Decay
        5. Passage Stitching (chronological, bounded by stitch_budget)
        6. Global context budget bounding
        """
        t_start = time.monotonic()
        timings: Dict[str, float] = {}

        # Auto-load workspace if specified or different
        if workspace_path:
            norm_ws = normalize_workspace_uri(workspace_path)
            if norm_ws != self.indexed_workspace or self.index.doc_count == 0:
                self.load_workspace(norm_ws)

        # Step 1: Tokenize query and extract query code symbols
        t0 = time.monotonic()
        query_tokens = self.index.tokenize(query)
        query_symbols = extract_code_symbols(query)
        timings["tokenize_ms"] = round((time.monotonic() - t0) * 1000, 3)

        if not query_tokens or self.index.doc_count == 0:
            return SearchResult(
                passages=[],
                total_found=0,
                query=query,
                query_tokens=query_tokens,
                query_symbols=list(query_symbols["all_symbols"]),
                timings_ms={"total_ms": round((time.monotonic() - t_start) * 1000, 3)},
                workspace_matched=self.indexed_workspace,
                conversations_indexed=len(self.indexed_conversations),
                total_chunks_indexed=self.index.doc_count,
            )

        # Step 2: Retrieve candidate documents via inverted index
        t1 = time.monotonic()
        candidates = self.index.search_candidates(query, top_n=60)
        timings["bm25_candidates_ms"] = round((time.monotonic() - t1) * 1000, 3)

        # Step 3: Compute boosts and recency factor for candidates
        t2 = time.monotonic()
        scored_chunks: List[DocumentChunk] = []
        ref_time = datetime.now(timezone.utc)

        for doc_idx, raw_score in candidates:
            chunk = self.index.documents[doc_idx].clone()
            chunk.raw_bm25 = raw_score

            # Symbol boost
            sym_boost = calculate_symbol_boost(query, chunk, query_symbols)
            chunk.symbol_boost = sym_boost

            # Recency factor
            rec_factor = compute_recency_factor(
                doc_timestamp=chunk.created_at,
                reference_time=ref_time,
                half_life_days=self.recency_half_life_days,
                floor=self.recency_floor,
                mode=self.recency_mode
            )
            chunk.recency_factor = rec_factor

            # Composite score
            chunk.score = raw_score * sym_boost * rec_factor

            if chunk.score >= min_score:
                scored_chunks.append(chunk)

        scored_chunks.sort(key=lambda p: p.score, reverse=True)
        timings["boost_and_recency_ms"] = round((time.monotonic() - t2) * 1000, 3)

        # Step 4: Stitch adjacent passages
        t3 = time.monotonic()
        stitched = stitch_passages(scored_chunks, budget=stitch_budget)
        timings["stitch_ms"] = round((time.monotonic() - t3) * 1000, 3)

        # Step 5: Enforce global context character budget
        selected: List[DocumentChunk] = []
        remaining_budget = context_budget

        for p in stitched[:top_k]:
            p_len = len(p.content)
            if p_len <= remaining_budget or not selected:
                selected.append(p)
                remaining_budget -= p_len
            else:
                break

        timings["total_ms"] = round((time.monotonic() - t_start) * 1000, 3)

        return SearchResult(
            passages=selected,
            total_found=len(selected),
            query=query,
            query_tokens=query_tokens,
            query_symbols=list(query_symbols["all_symbols"]),
            timings_ms=timings,
            workspace_matched=self.indexed_workspace,
            conversations_indexed=len(self.indexed_conversations),
            total_chunks_indexed=self.index.doc_count,
        )

    def list_artifacts(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Discovers all artifact files (*.md, *.txt, *.json) for a conversation."""
        conv_dir = os.path.join(self.brain_dir, conversation_id)
        if not os.path.isdir(conv_dir):
            return []

        artifacts: List[Dict[str, Any]] = []
        for root, dirs, files in os.walk(conv_dir):
            if ".system_generated" in root or ".user_uploaded" in root:
                continue
            for f in files:
                if f.endswith(".metadata.json"):
                    continue
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, conv_dir)
                try:
                    stat = os.stat(full_path)
                    mtime_dt = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                    artifacts.append({
                        "file_name": f,
                        "rel_path": rel_path.replace("\\", "/"),
                        "full_path": full_path,
                        "size_bytes": stat.st_size,
                        "modified_at": mtime_dt.isoformat(),
                    })
                except OSError:
                    continue

        return sorted(artifacts, key=lambda a: a["file_name"].lower())

    def read_artifact(self, conversation_id: str, artifact_name: str) -> Optional[Dict[str, Any]]:
        """Reads a specific artifact by filename or path within the conversation brain."""
        conv_dir = os.path.join(self.brain_dir, conversation_id)
        if not os.path.isdir(conv_dir):
            return None

        # Clean search target
        target = artifact_name.strip().replace("\\", "/").lstrip("/")
        base_target = os.path.basename(target).lower()
        base_no_ext = os.path.splitext(base_target)[0]

        candidates = self.list_artifacts(conversation_id)

        matched_path = None
        matched_item = None

        # 1. Exact match on relative path or file name
        for item in candidates:
            if item["rel_path"].lower() == target.lower() or item["file_name"].lower() == base_target:
                matched_item = item
                matched_path = item["full_path"]
                break

        # 2. Match without extension (e.g. 'walkthrough' matches 'walkthrough.md')
        if not matched_path:
            for item in candidates:
                cand_no_ext = os.path.splitext(item["file_name"].lower())[0]
                if cand_no_ext == base_no_ext or cand_no_ext == target.lower():
                    matched_item = item
                    matched_path = item["full_path"]
                    break

        if not matched_path or not os.path.isfile(matched_path):
            return None

        try:
            with open(matched_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return {
                "file_name": matched_item["file_name"],
                "rel_path": matched_item["rel_path"],
                "full_path": matched_path,
                "size_bytes": matched_item["size_bytes"],
                "modified_at": matched_item["modified_at"],
                "content": content,
            }
        except Exception:
            return None

    def get_transcript_turns(
        self,
        conversation_id: str,
        max_turns: int = 30
    ) -> Optional[Dict[str, Any]]:
        """Retrieves cleaned dialogue turns for a specific conversation."""
        meta = self.get_conversation_metadata(conversation_id)
        conv_dir = os.path.join(self.brain_dir, conversation_id)
        transcript_path = os.path.join(conv_dir, ".system_generated", "logs", "transcript.jsonl")

        if not os.path.exists(transcript_path):
            # Check transcript_full.jsonl fallback
            full_path = os.path.join(conv_dir, ".system_generated", "logs", "transcript_full.jsonl")
            if os.path.exists(full_path):
                transcript_path = full_path
            else:
                return None

        last_mod = meta.get("last_modified_time", "") if meta else ""
        turns = self.parser.parse_transcript_turns_raw(transcript_path, last_modified=last_mod)

        # Slice to max_turns (last max_turns)
        total_turns = len(turns)
        selected_turns = turns[-max_turns:] if total_turns > max_turns else turns

        return {
            "conversation_id": conversation_id,
            "title": meta.get("title", "Untitled Conversation") if meta else "Untitled Conversation",
            "workspace_uris": meta.get("workspace_uris", []) if meta else [],
            "last_modified_time": last_mod,
            "total_turns": total_turns,
            "returned_turns_count": len(selected_turns),
            "is_truncated": total_turns > max_turns,
            "turns": selected_turns,
        }
