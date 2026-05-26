# -*- coding: utf-8 -*-
"""
CatTool - File Viewer Tool

Reads file contents, similar to Linux cat command.
Executes real Linux commands via workspace.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import BaseTool
from .result import ToolResult

if TYPE_CHECKING:
    from ..workspaces.base import BaseWorkspace


class CatTool(BaseTool):
    """
    File Viewer Tool
    
    Reads file contents via workspace using the cat command.
    """
    
    name = "cat"
    description = "View file contents"

    def __init__(self, workspace: Optional["BaseWorkspace"] = None):
        super().__init__(workspace=workspace)

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
                        "paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": 'List of file paths, e.g. ["/docs/readme.md"]',
                        },
                        "path": {
                            "type": "string",
                            "description": 'Single file path, e.g. "/docs/readme.md" (use either path or paths)',
                        },
                    },
                    "required": [],
                },
            },
        }

    def execute(
        self,
        paths: str | List[str] = None,
        path: str = None,
        **kwargs: Any,
    ) -> List[ToolResult]:
        """Execute cat command via workspace"""
        if self.workspace is None:
            return [ToolResult(
                path="",
                content="Error: Workspace not initialized",
                score=0.0,
                rank=0,
                meta={"error": True},
            )]
        
        files_to_read = paths or path
        if files_to_read is None:
            return [ToolResult(
                path="",
                content="Error: No file paths provided",
                score=0.0,
                rank=0,
                meta={"error": True},
            )]

        if isinstance(files_to_read, str):
            files_to_read = [files_to_read]

        results: List[ToolResult] = []
        for rank, file_path in enumerate(files_to_read):
            # 使用 BaseTool.resolve_path 统一解析路径（处理根节点名称前缀）
            abs_path = self.resolve_path(file_path)
            
            # 检查是否是目录
            check_output, _ = self.workspace.run(f'test -d "{abs_path}" && echo "is_dir"')
            if "is_dir" in check_output:
                results.append(ToolResult(
                    path=file_path,
                    content="This is a directory. Use ls to list its contents.",
                    score=0.0,
                    rank=rank,
                    meta={"type": "error", "is_dir": True},
                ))
                continue
            
            # 检查文件是否存在
            check_output, _ = self.workspace.run(f'test -f "{abs_path}" && echo "exists"')
            if "exists" not in check_output:
                results.append(ToolResult(
                    path=file_path,
                    content=f"File not found: {file_path}",
                    score=0.0,
                    rank=rank,
                    meta={"type": "error", "exists": False},
                ))
                continue
            
            # 执行 cat 命令（读取文件内容，增加长度限制防止上下文爆炸）
            # 默认只读取前 30000 字符（约 7500 tokens）
            max_chars = 30000
            output, exit_code = self.workspace.run(f'head -c {max_chars + 1} "{abs_path}"')
            
            if isinstance(exit_code, str) and exit_code.startswith("Error"):
                results.append(ToolResult(
                    path=file_path,
                    content=f"Error reading file: {file_path}",
                    score=0.0,
                    rank=rank,
                    meta={"type": "error"},
                ))
                continue
            
            content = output
            if not content:
                content = "(empty file)"
            
            if len(content) > max_chars:
                content = content[:max_chars] + f"\n\n[... File content truncated because it exceeds {max_chars} characters. Use other tools or specific queries if you need more details. ...]"

            results.append(ToolResult(
                path=file_path,
                content=content,
                score=1.0,
                rank=rank,
                meta={"type": "file"},
            ))

        return results
