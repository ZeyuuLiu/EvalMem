# -*- coding: utf-8 -*-
"""
BM25SearchTool - BM25 Keyword Search Tool

Uses Pyserini's Lucene BM25 to index all files in the GAM workspace and
retrieve the most relevant files for a given query.

This is an OPTIONAL tool (not registered by default). Add it explicitly:

    from gam.tools.bm25_search_tool import BM25SearchTool

    bm25_tool = BM25SearchTool(workspace=workspace, index_dir="/tmp/bm25_index")
    chat_agent.register_tool(bm25_tool)
"""

from __future__ import annotations

import os
import json
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import BaseTool
from .result import ToolResult

if TYPE_CHECKING:
    from ..workspaces.base import BaseWorkspace


def _safe_rmtree(path: str, max_retries: int = 3, delay: float = 0.5) -> None:
    """Safely remove a directory tree with retry mechanism."""
    if not os.path.exists(path):
        return
    for attempt in range(max_retries):
        try:
            shutil.rmtree(path)
            if not os.path.exists(path):
                return
            time.sleep(delay)
        except OSError as e:
            if attempt == max_retries - 1:
                try:
                    subprocess.run(["rm", "-rf", path], check=False, capture_output=True)
                    if not os.path.exists(path):
                        return
                except Exception:
                    pass
                raise OSError(f"Failed to remove directory {path}: {e}")
            time.sleep(delay)


class BM25SearchTool(BaseTool):
    """
    BM25 Keyword Search Tool

    Builds a Lucene/BM25 index over all text files in the GAM workspace and
    retrieves the top-k most relevant files for a given natural-language query.

    This tool is NOT registered by default. You can add it to a TextChatAgent
    via `register_tool()` or by passing it in the `tools` parameter.
    """

    name = "bm25_search"
    description = (
        "Search the knowledge base using BM25 keyword retrieval. "
        "Given a natural-language query, returns the most relevant files ranked by BM25 score. "
        "Use this when you want to find files related to a topic without knowing exact keywords."
    )

    def __init__(
        self,
        workspace: Optional["BaseWorkspace"] = None,
        index_dir: Optional[str] = None,
        top_k: int = 5,
        threads: int = 1,
    ):
        """
        Initialize BM25SearchTool.

        Args:
            workspace: Workspace instance (provides root_path and run())
            index_dir: Directory to store the BM25 index. If None, defaults to
                       <workspace_root>/.bm25_index
            top_k: Default number of top results to return
            threads: Number of threads for index building
        """
        super().__init__(workspace=workspace)
        self.top_k = top_k
        self.threads = threads
        self._index_dir = index_dir
        self._searcher = None          # LuceneSearcher instance
        self._file_paths: List[str] = []   # indexed file relative paths
        self._file_contents: List[str] = []  # indexed file contents
        self._index_built = False

    # ------------------------------------------------------------------
    # Internal directory helpers
    # ------------------------------------------------------------------
    @property
    def index_dir(self) -> str:
        if self._index_dir:
            return self._index_dir
        if self.workspace:
            return os.path.join(str(self.workspace.root_path), ".bm25_index")
        return "/tmp/bm25_index"

    def _lucene_dir(self) -> str:
        return os.path.join(self.index_dir, "lucene")

    def _docs_dir(self) -> str:
        return os.path.join(self.index_dir, "documents")

    # ------------------------------------------------------------------
    # Spec (OpenAI Function Calling)
    # ------------------------------------------------------------------
    def spec(self) -> Dict[str, Any]:
        """Return OpenAI Function Calling compatible tool specification"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural-language search query",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": f"Number of top results to return (default: {self.top_k})",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------
    def _collect_files(self) -> List[Dict[str, str]]:
        """
        Walk the workspace directory and collect all text files.
        Returns a list of {"rel_path": ..., "content": ...}.
        """
        if self.workspace is None:
            return []

        root = str(self.workspace.root_path)
        files = []

        for dirpath, dirnames, filenames in os.walk(root):
            # Skip hidden directories
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fname in filenames:
                if fname.startswith("."):
                    continue
                abs_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(abs_path, root)
                try:
                    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    if content.strip():
                        files.append({"rel_path": rel_path, "content": content})
                except Exception:
                    continue

        return files

    def build_index(self) -> None:
        """
        Build (or rebuild) the BM25 index from all files in the workspace.
        Called automatically on first search if not already built.
        """
        try:
            from pyserini.search.lucene import LuceneSearcher
        except ImportError:
            raise ImportError(
                "BM25SearchTool requires pyserini. "
                "Install it with: pip install pyserini"
            )

        # Collect files
        file_records = self._collect_files()
        if not file_records:
            self._index_built = True
            return

        # Clean old index
        _safe_rmtree(self._lucene_dir())
        _safe_rmtree(self._docs_dir())

        os.makedirs(self._docs_dir(), exist_ok=True)
        os.makedirs(self._lucene_dir(), exist_ok=True)

        # Write documents.jsonl for pyserini
        self._file_paths = []
        self._file_contents = []
        docs_path = os.path.join(self._docs_dir(), "documents.jsonl")
        with open(docs_path, "w", encoding="utf-8") as f:
            for i, rec in enumerate(file_records):
                self._file_paths.append(rec["rel_path"])
                self._file_contents.append(rec["content"])
                json.dump(
                    {"id": str(i), "contents": rec["content"]},
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")

        # Build Lucene index via pyserini CLI
        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", self._docs_dir(),
            "--index", self._lucene_dir(),
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", str(self.threads),
            "--storePositions", "--storeDocvectors", "--storeRaw",
        ]

        max_retries = 2
        for attempt in range(max_retries):
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                break
            except subprocess.CalledProcessError as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to build BM25 index: stdout={e.stdout}, stderr={e.stderr}"
                    )
                _safe_rmtree(self._lucene_dir())
                os.makedirs(self._lucene_dir(), exist_ok=True)
                time.sleep(1)

        self._searcher = LuceneSearcher(self._lucene_dir())
        self._index_built = True

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    def execute(
        self,
        query: str = "",
        top_k: int = None,
        **kwargs: Any,
    ) -> List[ToolResult]:
        """
        Search the BM25 index and return the most relevant files.

        Args:
            query: Natural-language search query
            top_k: Number of top results (defaults to self.top_k)

        Returns:
            List[ToolResult] with matching file contents
        """
        if self.workspace is None:
            return [ToolResult(
                path="",
                content="Error: Workspace not initialized",
                score=0.0,
                rank=0,
                meta={"error": True},
            )]

        if not query or not query.strip():
            return [ToolResult(
                path="",
                content="Error: No query provided",
                score=0.0,
                rank=0,
                meta={"error": True},
            )]

        # Build index on first use
        if not self._index_built:
            try:
                self.build_index()
            except Exception as e:
                return [ToolResult(
                    path="",
                    content=f"Error building BM25 index: {str(e)}",
                    score=0.0,
                    rank=0,
                    meta={"error": True},
                )]

        # If no files were indexed
        if not self._file_paths or self._searcher is None:
            return [ToolResult(
                path="/",
                content="No files found in workspace to search.",
                score=0.0,
                rank=0,
                meta={"type": "bm25_search", "empty": True},
            )]

        if top_k is None:
            top_k = self.top_k

        # Perform BM25 search
        try:
            hits = self._searcher.search(query.strip(), k=top_k)
        except Exception as e:
            return [ToolResult(
                path="",
                content=f"Error during BM25 search: {str(e)}",
                score=0.0,
                rank=0,
                meta={"error": True},
            )]

        if not hits:
            return [ToolResult(
                path="/",
                content="No matching files found.",
                score=0.0,
                rank=0,
                meta={"type": "bm25_search", "empty": True},
            )]

        results: List[ToolResult] = []
        for rank, h in enumerate(hits):
            idx = int(h.docid)
            if idx < 0 or idx >= len(self._file_paths):
                continue

            rel_path = "/" + self._file_paths[idx]
            content = self._file_contents[idx]

            results.append(ToolResult(
                path=rel_path,
                content=content,
                score=float(h.score),
                rank=rank,
                meta={"type": "file", "bm25_score": float(h.score)},
            ))

        return results if results else [ToolResult(
            path="/",
            content="No matching files found.",
            score=0.0,
            rank=0,
            meta={"type": "bm25_search", "empty": True},
        )]
