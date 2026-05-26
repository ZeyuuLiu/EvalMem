# -*- coding: utf-8 -*-
"""
ToolResult - Tool execution result data class
"""

from __future__ import annotations

from typing import Any, Dict
from dataclasses import dataclass, field


@dataclass
class ToolResult:
    """Tool execution result"""
    path: str = ""
    content: str = ""
    score: float = 0.0
    rank: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)
