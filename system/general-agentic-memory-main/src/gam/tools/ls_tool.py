# -*- coding: utf-8 -*-
"""
LsTool - Directory Listing Tool

Lists files and subdirectories in a directory, similar to Linux ls command.
Executes real Linux commands via workspace.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import BaseTool
from .result import ToolResult

if TYPE_CHECKING:
    from ..workspaces.base import BaseWorkspace


class LsTool(BaseTool):
    """
    Directory Listing Tool
    
    Lists files and subdirectories via workspace using the ls command.
    """
    
    name = "ls"
    description = "List files and subdirectories in a directory"

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
                            "description": 'List of directory paths, e.g. ["/", "/docs"]',
                        },
                        "path": {
                            "type": "string",
                            "description": 'Single directory path, e.g. "/docs" (use either path or paths)',
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
        """Execute ls command via workspace"""
        if self.workspace is None:
            return [ToolResult(
                path="/",
                content="Error: Workspace not initialized",
                score=0.0,
                rank=0,
                meta={"error": True},
            )]
        
        dirs_to_list = paths or path or ["/"]
        if isinstance(dirs_to_list, str):
            dirs_to_list = [dirs_to_list]

        results: List[ToolResult] = []
        for rank, dir_path in enumerate(dirs_to_list):
            # 使用 BaseTool.resolve_path 统一解析路径（处理根节点名称前缀）
            abs_path = self.resolve_path(dir_path)
            
            # 执行 ls 命令
            # -1: 每行一个文件
            # -p: 目录后加 /
            # --group-directories-first: 目录优先
            output, exit_code = self.workspace.run(
                f'ls -1p --group-directories-first "{abs_path}" 2>&1'
            )
            
            # 检查目录是否存在（避免 pipe 吞掉 ls 的错误码）
            check_dir, _ = self.workspace.run(f'test -d "{abs_path}" && echo "is_dir"')
            if "is_dir" not in check_dir:
                # 检查是否是文件而不是目录
                check_output, check_code = self.workspace.run(f'test -f "{abs_path}" && echo "is_file"')
                if "is_file" in check_output:
                    results.append(ToolResult(
                        path=dir_path or "/",
                        content="Not a directory. Use cat to read file content.",
                        score=0.0,
                        rank=rank,
                        meta={"type": "error", "is_dir": False},
                    ))
                else:
                    results.append(ToolResult(
                        path=dir_path or "/",
                        content=f"Directory not found: {dir_path or '/'}",
                        score=0.0,
                        rank=rank,
                        meta={"type": "error", "exists": False},
                    ))
                continue
            
            items = [line.strip() for line in output.strip().split("\n") if line.strip()]
            # 过滤掉隐藏文件
            items = [item for item in items if not item.startswith(".")]
            
            content = "\n".join(items) if items else "(empty directory)"

            results.append(ToolResult(
                path=dir_path or "/",
                content=content,
                score=1.0,
                rank=rank,
                meta={"type": "directory", "count": len(items)},
            ))

        return results
