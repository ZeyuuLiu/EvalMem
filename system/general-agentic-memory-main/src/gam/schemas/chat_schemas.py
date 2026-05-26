# -*- coding: utf-8 -*-
"""
Chat Schemas

Data schemas for chat agent operations.
"""

from __future__ import annotations

from typing import List
from dataclasses import dataclass, field


@dataclass
class ChatResult:
    """聊天/研究结果"""
    question: str = ""
    answer: str = ""
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""
    
    # 过程信息
    files_read: List[str] = field(default_factory=list)
    dirs_explored: List[str] = field(default_factory=list)
    trajectory: str = ""  # 完整的探索轨迹
