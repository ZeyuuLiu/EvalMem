# -*- coding: utf-8 -*-
"""
GrepTool - Content Search Tool

Searches for keywords or regular expressions in files, similar to Linux grep.
Executes real Linux commands via workspace.

Supports deduplication: when `exclude_paths` is provided (typically the set of
file paths already returned in earlier calls), the tool will:
1. Skip those files and report them as "already retrieved".
2. Continue searching for NEW matching files up to `max_results`.
This prevents repeated grep calls from returning the same results and ensures
that the agent can always discover new content.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from .base import BaseTool
from .result import ToolResult

if TYPE_CHECKING:
    from ..workspaces.base import BaseWorkspace


class GrepTool(BaseTool):
    """
    Content Search Tool
    
    Searches for keywords or regular expressions via workspace using the grep command.
    
    Supports automatic deduplication via `exclude_paths`: when provided, files that
    were already returned in previous calls are skipped and reported, while the tool
    continues to search for new matching files.
    """
    
    name = "grep"
    description = "Search for keywords or regular expressions in files"

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
                        "pattern": {
                            "type": "string",
                            "description": "Keyword or regular expression to search for",
                        },
                        "path": {
                            "type": "string",
                            "description": 'Starting directory for search, defaults to "/"',
                        },
                        "regex": {
                            "type": "boolean",
                            "description": "Whether to use regex matching, defaults to true",
                        },
                        "case_sensitive": {
                            "type": "boolean",
                            "description": "Whether search is case sensitive, defaults to false",
                        },
                    },
                    "required": ["pattern"],
                },
            },
        }

    def execute(
        self,
        pattern: str = None,
        path: str = "/",
        regex: bool = True,
        case_sensitive: bool = False,
        max_results: int = 3,
        max_line_length: int = -1,
        exclude_paths: Optional[Set[str]] = None,
        **kwargs: Any,
    ) -> List[ToolResult]:
        """Execute grep command via workspace.
        
        Args:
            pattern: Keyword or regex to search for.
            path: Starting directory, defaults to "/".
            regex: Use regex matching (True) or fixed-string (False).
            case_sensitive: Case-sensitive search.
            max_results: Maximum number of NEW file results to return.
            max_line_length: (unused, kept for compat).
            exclude_paths: Set of relative file paths already returned in
                           previous calls. These files will be skipped, and
                           the tool will continue searching for new matches.
            **kwargs: Extra keyword arguments (ignored).
            
        Returns:
            List of ToolResult. If some matches were already seen, a summary
            note is prepended before the new results.
        """
        if self.workspace is None:
            return [ToolResult(
                path="/",
                content="Error: Workspace not initialized",
                score=0.0,
                rank=0,
                meta={"error": True},
            )]
        
        if not pattern:
            return [ToolResult(
                path=path,
                content="Error: No pattern provided",
                score=0.0,
                rank=0,
                meta={"error": True},
            )]

        # 使用 BaseTool.resolve_path 统一解析路径（处理根节点名称前缀）
        abs_path = self.resolve_path(path)
        
        # 构建 grep 命令
        # -r: 递归搜索
        # -l: 只输出匹配的文件名（不输出具体行）
        # -E: 使用扩展正则（如果 regex=True）
        # -F: 固定字符串匹配（如果 regex=False）
        # -i: 忽略大小写（如果 case_sensitive=False）
        grep_flags = "-rl"
        if regex:
            grep_flags += "E"
        else:
            grep_flags += "F"
        if not case_sensitive:
            grep_flags += "i"
        
        exclude_paths = exclude_paths or set()
        
        # 如果有需要排除的路径，多取一些结果以便过滤后仍能返回 max_results 个新文件
        # 取 max_results + len(exclude_paths) + 余量，确保能搜到足够的新结果
        fetch_limit = max_results + len(exclude_paths) + max_results
        
        cmd = f'grep {grep_flags} -- "{pattern}" "{abs_path}" 2>/dev/null | head -n {fetch_limit}'
        output, exit_code = self.workspace.run(cmd)
        
        # grep 返回 1 表示没有匹配，不是错误
        if not output.strip():
            return [ToolResult(
                path=path,
                content="No matches found.",
                score=0.0,
                rank=0,
                meta={"type": "grep", "empty": True},
            )]
        
        # 解析 grep -l 输出：每行是一个匹配到的文件路径
        root_path_str = str(self.workspace.root_path)
        
        # 将所有匹配文件分为"已看过"和"新发现"两组
        already_seen_paths: List[str] = []
        new_results: List[ToolResult] = []
        
        for line in output.strip().split("\n"):
            file_path_abs = line.strip()
            if not file_path_abs:
                continue
            
            # 转换为相对路径
            if file_path_abs.startswith(root_path_str):
                file_path_rel = file_path_abs[len(root_path_str):].lstrip("/")
                file_path_rel = "/" + file_path_rel if file_path_rel else "/"
            else:
                file_path_rel = file_path_abs
            
            # 检查是否已经看过
            if file_path_rel in exclude_paths:
                already_seen_paths.append(file_path_rel)
                continue
            
            # 新文件，读取内容（增加内容限制防止上下文爆炸）
            if len(new_results) >= max_results:
                # 已经收集够 max_results 个新文件了
                continue
            
            # 默认只读取前 10000 字符，防止搜索结果中出现超大文件撑爆上下文
            max_chars = 10000
            cat_cmd = f'head -c {max_chars + 1} "{file_path_abs}" 2>/dev/null'
            file_content, cat_exit_code = self.workspace.run(cat_cmd)
            
            if (isinstance(cat_exit_code, str) and cat_exit_code.startswith("Error")) or not file_content:
                file_content = "(Failed to read file content)"
            
            if len(file_content) > max_chars:
                file_content = file_content[:max_chars] + f"\n\n[... File content truncated because it exceeds {max_chars} characters in GrepTool. Use cat tool for more details. ...]"

            new_results.append(ToolResult(
                path=file_path_rel,
                content=file_content,
                score=1.0,
                rank=len(new_results),
                meta={"type": "file"},
            ))
        
        # 构建最终结果
        results: List[ToolResult] = []
        
        # 如果有已看过的文件，先添加一个汇总提示
        if already_seen_paths:
            seen_list = ", ".join(already_seen_paths)
            seen_note = (
                f"[Note] The following {len(already_seen_paths)} file(s) also match "
                f"but were already retrieved in previous searches: {seen_list}"
            )
            results.append(ToolResult(
                path=path,
                content=seen_note,
                score=0.0,
                rank=-1,
                meta={"type": "grep_dedup_note", "already_seen": already_seen_paths},
            ))
        
        # 添加新发现的文件
        results.extend(new_results)
        
        # 如果既没有已看过的也没有新发现的
        if not results:
            return [ToolResult(
                path=path,
                content="No matches found.",
                score=0.0,
                rank=0,
                meta={"type": "grep", "empty": True},
            )]
        
        # 如果只有已看过的（没有新发现）
        if already_seen_paths and not new_results:
            results.append(ToolResult(
                path=path,
                content=(
                    f"All matching files ({len(already_seen_paths)}) have already been "
                    f"retrieved in previous searches. No new matches found for this pattern. "
                    f"Try a different search pattern or directory to find more information."
                ),
                score=0.0,
                rank=0,
                meta={"type": "grep", "all_seen": True},
            ))
        
        return results
