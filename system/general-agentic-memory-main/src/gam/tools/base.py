# -*- coding: utf-8 -*-
"""
BaseTool - Tool Base Class

Provides OpenAI Function Calling compatible tool interface.
Executes real Linux commands via workspace.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .result import ToolResult

if TYPE_CHECKING:
    from ..workspaces.base import BaseWorkspace


class BaseTool(ABC):
    """
    Tool Base Class
    
    All tools must implement:
    - name: Tool name
    - description: Tool description
    - spec(): Return OpenAI Function Calling compatible tool specification
    - execute(): Execute tool logic
    
    Tools execute real Linux commands via workspace (supports Local/Docker).
    """
    
    name: str
    description: str

    def __init__(self, workspace: Optional["BaseWorkspace"] = None):
        """
        Initialize tool.
        
        Args:
            workspace: Workspace instance for executing Linux commands
        """
        self.workspace = workspace

    def resolve_path(self, path: str) -> str:
        """
        Resolve virtual path used by agent to real absolute path.
        
        Handles the following cases:
        - "/" or "" → workspace root
        - "/gam/subdir" → strips workspace root name prefix to avoid double paths
        - "subdir" → relative path concatenation
        
        Args:
            path: Path from agent (may contain root node name prefix)
            
        Returns:
            Absolute path string
        """
        if self.workspace is None:
            return path
        
        root = str(self.workspace.root_path)
        root_dir_name = self.workspace.root_path.name  # e.g. "gam"
        
        # "/" 或 "" 表示 workspace 根目录
        if path == "/" or path == "":
            return root
        
        # 去掉前导斜杠
        clean = path.strip("/")
        
        # 如果路径以 workspace root 目录名开头，去掉这个前缀
        # 例如 root_dir_name="gam", clean="gam/subdir" → "subdir"
        # 这样 workspace_root/subdir 而不是 workspace_root/gam/subdir
        if clean == root_dir_name:
            return root
        if clean.startswith(root_dir_name + "/"):
            clean = clean[len(root_dir_name) + 1:]
        
        if not clean:
            return root
        
        return f"{root}/{clean}"

    @abstractmethod
    def spec(self) -> Dict[str, Any]:
        """
        返回 OpenAI Function Calling 兼容的工具规范
        
        格式:
        {
            "type": "function",
            "function": {
                "name": "tool_name",
                "description": "tool description",
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }
        """
        pass

    @abstractmethod
    def execute(self, **kwargs: Any) -> List[ToolResult]:
        """执行工具逻辑"""
        pass