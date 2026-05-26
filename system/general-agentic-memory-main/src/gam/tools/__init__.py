# -*- coding: utf-8 -*-
"""
GAM Tools Module

Tool module providing various tools for Agent use.

Default tools (auto-registered):
    - LsTool: List directories
    - CatTool: View file contents
    - GrepTool: Keyword search

Optional tools (manually added):
    - BM25SearchTool: BM25-based semantic search over workspace files
    - InspectVideoTool: Analyze video segments using multi-modal LLM

"""

from .base import BaseTool
from .result import ToolResult
from .ls_tool import LsTool
from .cat_tool import CatTool
from .grep_tool import GrepTool
from .bm25_search_tool import BM25SearchTool
from .inspect_video_tool import InspectVideoTool

__all__ = [
    "BaseTool",
    "ToolResult",
    "LsTool",
    "CatTool",
    "GrepTool",
    "BM25SearchTool",
    "InspectVideoTool",
]
