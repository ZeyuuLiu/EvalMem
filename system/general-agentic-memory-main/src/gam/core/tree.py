# -*- coding: utf-8 -*-
"""
GAMTree - General Agentic Memory Tree

The main file system tree structure that manages the entire GAM.
Only provides READ-ONLY view capabilities. All operations go through Workspace.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from pydantic import BaseModel, Field
from datetime import datetime
import os

from .node import FSNode, NodeType

if TYPE_CHECKING:
    from ..workspaces.base import BaseWorkspace


class BaseTree(ABC):
    """Minimal tree interface for GAM-like structures (READ-ONLY)."""

    @property
    @abstractmethod
    def root_path(self) -> Path:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_disk(cls, root_path: Path, workspace: "BaseWorkspace") -> "BaseTree":
        raise NotImplementedError


class GAMTree(BaseModel, BaseTree):
    """
    General Agentic Memory Tree (READ-ONLY)
    
    管理整个文件系统树的查看功能：
    - 查看树结构
    - 从磁盘加载（通过 workspace 执行命令）
    
    注意：所有操作（读取、写入）都通过 Workspace 执行 Linux 命令来完成
    """
    root_path_: Path = Field(..., alias="root_path", description="磁盘上的根路径")
    root: FSNode = Field(..., description="根节点")
    meta: Dict[str, Any] = Field(default_factory=dict, description="GAM 元信息")
    
    model_config = {"arbitrary_types_allowed": True, "populate_by_name": True}
    
    @property
    def root_path(self) -> Path:
        """实现 BaseTree 抽象属性，返回磁盘上的根路径"""
        return self.root_path_
    
    @property
    def is_empty(self) -> bool:
        """GAM 是否为空（只有根节点，没有子内容）"""
        return len(self.root.children) == 0
    
    def get_node(self, path: str) -> Optional[FSNode]:
        """
        根据路径获取节点
        
        Args:
            path: 路径，如 "/dir1/dir2/file.md" 或 "dir1/dir2"
        """
        # 处理路径
        path = path.strip()
        if path in ("", "/", "."):
            return self.root
        
        # 移除开头的 /
        if path.startswith("/"):
            path = path[1:]
        
        # 分割路径
        parts = [p for p in path.split("/") if p]
        
        return self.root.find_node(parts)
    
    def tree_view(self, path: str = "/", depth: int = -1) -> str:
        """
        获取树形视图字符串
        
        Args:
            path: 起始路径
            depth: 最大深度
        """
        node = self.get_node(path)
        if node is None:
            return f"Path not found: {path}"
        return node.to_tree_str(depth=depth)
    
    def reload(self, workspace: "BaseWorkspace") -> "GAMTree":
        """
        从磁盘重新加载 GAM 树
        
        Args:
            workspace: Workspace 实例，用于执行命令
        
        用于在 workspace 执行写操作后刷新内存中的树结构
        """
        return self.from_disk(self.root_path_, workspace)
    
    @classmethod
    def from_disk(cls, root_path: Path, workspace: "BaseWorkspace") -> "GAMTree":
        """
        从磁盘加载 GAM 树（通过 workspace 执行命令）
        
        Args:
            root_path: GAM 根目录路径
            workspace: Workspace 实例，用于执行命令
        """
        root_path = Path(root_path)
        
        # 检查目录是否存在
        # workspace.run() 返回 Tuple[str, str] = (output, exit_code_or_error)
        output, _ = workspace.run(f"test -d '{root_path}' && echo 'exists' || echo 'not_found'")
        if "not_found" in output:
            raise FileNotFoundError(f"GAM root path not found: {root_path}")
        
        # 读取元数据
        meta_file = root_path / ".gam_meta.json"
        meta = {}
        root_name = root_path.name
        
        output, exit_info = workspace.run(f"cat '{meta_file}' 2>/dev/null || echo ''")
        # exit_info 可能是 "0" 或 "Error: ..." 格式
        is_success = not exit_info.startswith("Error")
        if output.strip() and is_success:
            try:
                meta_data = json.loads(output.strip())
                meta = meta_data.get("meta", {})
                root_name = meta_data.get("root_name", root_path.name)
            except json.JSONDecodeError:
                pass
        
        # 递归读取节点
        root = cls._read_node_from_disk(root_path, root_name, workspace)
        
        return cls(root_path=root_path, root=root, meta=meta)
    
    @classmethod
    def _read_node_from_disk(cls, path: Path, name: str, workspace: "BaseWorkspace") -> FSNode:
        """
        递归从磁盘读取节点（通过 workspace）
        
        Args:
            path: 节点路径
            name: 节点名称
            workspace: Workspace 实例
        """
        # 检查是否为目录
        output, _ = workspace.run(f"test -d '{path}' && echo 'dir' || echo 'file'")
        is_dir = "dir" in output
        
        if is_dir:
            # 读取 README.md 作为 summary
            readme_path = path / "README.md"
            output, exit_info = workspace.run(f"cat '{readme_path}' 2>/dev/null || echo ''")
            is_success = not exit_info.startswith("Error")
            summary = output if is_success and "No such file" not in output else ""
            
            # 创建目录节点
            node = FSNode.create_dir(name, summary=summary)
            
            # 列出子节点（排除隐藏文件和 README.md）
            output, exit_info = workspace.run(f"ls -1 '{path}' 2>/dev/null | grep -v '^\\..*' | grep -v '^README\\.md$' | sort")
            is_success = not exit_info.startswith("Error")
            if output.strip() and is_success:
                for child_name in output.strip().split('\n'):
                    child_name = child_name.strip()
                    if child_name:
                        child_path = path / child_name
                        child = cls._read_node_from_disk(child_path, child_name, workspace)
                        node.children.append(child)
            
            return node
        else:
            # 读取文件内容
            output, exit_info = workspace.run(f"cat '{path}'")
            is_success = not exit_info.startswith("Error")
            content = output if is_success else ""
            return FSNode.create_file(name, content=content)
    
    @classmethod
    def create_empty(cls, root_path: Path, name: str = "gam_root", meta: Dict = None) -> "GAMTree":
        """
        创建空的 GAM 树（仅内存结构）
        
        注意：这只创建内存中的树结构，实际的磁盘目录需要通过 workspace 创建
        """
        root_path = Path(root_path)
        root = FSNode.create_dir(name, summary="# General Agentic Memory\n\nThis is the root of the GAM.")
        return cls(root_path=root_path, root=root, meta=meta or {"created_at": datetime.now().isoformat()})
    
    def get_structure_summary(self) -> str:
        """
        获取 GAM 结构摘要，用于 LLM 上下文
        
        返回格式：
        - 树形结构
        - 各目录的 README 摘要
        """
        lines = ["## GAM Structure\n"]
        lines.append("```")
        lines.append(self.tree_view())
        lines.append("```\n")
        
        # 添加各目录的摘要
        lines.append("## Directory Summaries\n")
        for dir_node in self.root.get_all_dirs():
            if dir_node.summary:
                # 计算相对路径
                lines.append(f"### {dir_node.name}/")
                # 只取摘要的前几行
                summary_preview = dir_node.summary[:500]
                if len(dir_node.summary) > 500:
                    summary_preview += "..."
                lines.append(summary_preview)
                lines.append("")
        
        return "\n".join(lines)



class VideoGAMTree(BaseModel, BaseTree):
    """
    Video General Agentic Memory Tree (READ-ONLY)
    
    管理整个文件系统树的查看功能：
    - 查看树结构
    - 检查 video workspace 是否生成正确
    - 从磁盘加载（通过 workspace 执行命令）
    
    注意：所有操作（读取、写入）都通过 Workspace 执行 Linux 命令来完成
    """
    root_path_: Path = Field(..., alias="root_path", description="磁盘上的根路径")
    root: FSNode = Field(..., description="根节点")
    meta: Dict[str, Any] = Field(default_factory=dict, description="GAM 元信息")
    
    model_config = {"arbitrary_types_allowed": True, "populate_by_name": True}
    
    @property
    def root_path(self) -> Path:
        """实现 BaseTree 抽象属性，返回磁盘上的根路径"""
        return self.root_path_
    
    @property
    def is_empty(self) -> bool:
        """GAM 是否为空（只有根节点，没有子内容）"""
        return len(self.root.children) == 0
    
    def get_node(self, path: str) -> Optional[FSNode]:
        """
        根据路径获取节点
        
        Args:
            path: 路径，如 "/dir1/dir2/file.md" 或 "dir1/dir2"
        """
        # 处理路径
        path = path.strip()
        if path in ("", "/", "."):
            return self.root
        
        # 移除开头的 /
        if path.startswith("/"):
            path = path[1:]
        
        # 分割路径
        parts = [p for p in path.split("/") if p]
        
        return self.root.find_node(parts)
    
    def tree_view(self, path: str = "/", depth: int = -1) -> str:
        """
        获取树形视图字符串
        
        Args:
            path: 起始路径
            depth: 最大深度
        """
        node = self.get_node(path)
        if node is None:
            return f"Path not found: {path}"
        return node.to_tree_str(depth=depth)
    
    def reload(self, workspace: "BaseWorkspace") -> "GAMTree":
        """
        从磁盘重新加载 GAM 树
        
        Args:
            workspace: Workspace 实例，用于执行命令
        
        用于在 workspace 执行写操作后刷新内存中的树结构
        """
        return self.from_disk(self.root_path_, workspace)
    
    @classmethod
    def from_disk(cls, root_path: Path, workspace: "BaseWorkspace") -> "GAMTree":
        """
        从磁盘加载 GAM 树（通过 workspace 执行命令）
        
        Args:
            root_path: GAM 根目录路径
            workspace: Workspace 实例，用于执行命令
        """
        root_path = Path(root_path)
        
        # 检查目录是否存在
        # workspace.run() 返回 Tuple[str, str] = (output, exit_code_or_error)
        output, _ = workspace.run(f"test -d '{root_path}' && echo 'exists' || echo 'not_found'")
        if "not_found" in output:
            raise FileNotFoundError(f"GAM root path not found: {root_path}")
        
        # 读取元数据
        meta_file = root_path / ".gam_meta.json"
        meta = {}
        root_name = root_path.name
        
        output, exit_info = workspace.run(f"cat '{meta_file}' 2>/dev/null || echo ''")
        # exit_info 可能是 "0" 或 "Error: ..." 格式
        is_success = not exit_info.startswith("Error")
        if output.strip() and is_success:
            try:
                meta_data = json.loads(output.strip())
                meta = meta_data.get("meta", {})
                root_name = meta_data.get("root_name", root_path.name)
            except json.JSONDecodeError:
                pass
        
        # 递归读取节点
        root = cls._read_node_from_disk(root_path, root_name, workspace)
        
        return cls(root_path=root_path, root=root, meta=meta)
    
    @classmethod
    def _read_node_from_disk(cls, path: Path, name: str, workspace: "BaseWorkspace") -> FSNode:
        """
        递归从磁盘读取节点（通过 workspace）
        
        Args:
            path: 节点路径
            name: 节点名称
            workspace: Workspace 实例
        """
        # 检查是否为目录
        output, _ = workspace.run(f"test -d '{path}' && echo 'dir' || echo 'file'")
        is_dir = "dir" in output
        
        if is_dir:
            # 读取 README.md 作为 summary
            readme_path = path / "README.md"
            output, exit_info = workspace.run(f"cat '{readme_path}' 2>/dev/null || echo ''")
            is_success = not exit_info.startswith("Error")
            summary = output if is_success and "No such file" not in output else ""
            
            # 创建目录节点
            node = FSNode.create_dir(name, summary=summary)
            
            # 列出子节点（排除隐藏文件和 README.md）
            # output, exit_info = workspace.run(f"ls -1 '{path}' 2>/dev/null | grep -v '^\\..*' | grep -v '^README\\.md$' | sort")
            output, exit_info = workspace.run(f"ls -1 '{path}' 2>/dev/null | grep -v '^\\..*' | sort")
            is_success = not exit_info.startswith("Error")
            if output.strip() and is_success:
                for child_name in output.strip().split('\n'):
                    child_name = child_name.strip()
                    if child_name:
                        child_path = path / child_name
                        child = cls._read_node_from_disk(child_path, child_name, workspace)
                        node.children.append(child)
            
            return node
        else:
            # 读取文件内容
            output, exit_info = workspace.run(f"cat '{path}'")
            is_success = not exit_info.startswith("Error")
            content = output if is_success else ""
            return FSNode.create_file(name, content=content)
    
    @classmethod
    def create_empty(cls, root_path: Path, name: str = "gam_root", meta: Dict = None) -> "GAMTree":
        """
        创建空的 GAM 树（仅内存结构）
        
        注意：这只创建内存中的树结构，实际的磁盘目录需要通过 workspace 创建
        """
        root_path = Path(root_path)
        root = FSNode.create_dir(name, summary="# Video General Agentic Memory\n\nThis is the root of the GAM.")
        return cls(root_path=root_path, root=root, meta=meta or {"created_at": datetime.now().isoformat()})
    
    def check_video_workspace(self, workspace: "BaseWorkspace") -> bool:
        """
        Check if the video workspace contains all necessary files for a single video ID.
        
        Args:
            video_workspace_dir (str): Path to the video workspace directory.
            workspace (BaseWorkspace): Workspace instance for executing commands.
            
        Returns:
            bool: True if qualified. Raises Exception otherwise.
        """
        the_video_path = 'video.mp4'
        global_info_path = 'README.md'
        seg_dir = 'segments/'
        
        try:
            # check
            # assert os.path.exists(the_video_path)
            # _, exit_info = workspace.run(f"test -e '{the_video_path}'")
            # assert not exit_info.startswith("Error"), f"{the_video_path} not exists"
            
            # assert os.path.exists(global_info_path)
            _, exit_info = workspace.run(f"test -e '{global_info_path}'")
            assert not exit_info.startswith("Error"), f"{global_info_path} not exists"
            
            # assert os.path.exists(seg_dir), f'{seg_dir} not exists'
            _, exit_info = workspace.run(f"test -e '{seg_dir}'")
            assert not exit_info.startswith("Error"), f'{seg_dir} not exists'
            
            # assert len(os.listdir(seg_dir)) > 0
            output, exit_info = workspace.run(f"ls -1 '{seg_dir}'")
            assert not exit_info.startswith("Error"), f"Failed to list {seg_dir}"
            seg_ids = [x for x in output.strip().split('\n') if x]
            assert len(seg_ids) > 0, f"{seg_dir} is empty"
            
            # check segments
            # for seg_id in seg_ids:
            #     seg_path = os.path.join(seg_dir, seg_id)
            #     # assert os.path.exists(seg_path)
            #     _, exit_info = workspace.run(f"test -e '{seg_path}'")
            #     assert not exit_info.startswith("Error"), f"{seg_path} not exists"
                
            #     # assert os.path.exists(os.path.join(seg_path, 'README.md'))
            #     readme_path = os.path.join(seg_path, 'README.md')
            #     _, exit_info = workspace.run(f"test -e '{readme_path}'")
            #     assert not exit_info.startswith("Error"), f"{readme_path} not exists"
                
            #     # assert os.path.exists(os.path.join(seg_path, 'video.mp4'))
            #     video_path = os.path.join(seg_path, 'video.mp4')
            #     _, exit_info = workspace.run(f"test -e '{video_path}'")
            #     assert not exit_info.startswith("Error"), f"{video_path} not exists"
                
        except Exception as e:
            return False
            
        return True
    
    def organize_video_workspace(self, workspace: "BaseWorkspace"):
        """
        Organize the video workspace by moving some files to .agentignore directory.
        
        Args:
            video_workspace_dir (str): Path to the video workspace directory.
            workspace (BaseWorkspace): Workspace instance for executing commands.
        """
        workspace.run(f"mkdir -p '.agentignore'")
        
        _, exit_info = workspace.run(f"test -e 'subtitles.srt'")
        if not exit_info.startswith("Error"):
            dest_srt = os.path.join('.agentignore', 'subtitles.srt')
            workspace.run(f"mv subtitles.srt '.agentignore/'")
            
        _, exit_info = workspace.run(f"test -e 'PROBE_RESULT.json'")
        if not exit_info.startswith("Error"):
            dest_probe = os.path.join('.agentignore', 'PROBE_RESULT.json')
            workspace.run(f"mv PROBE_RESULT.json '.agentignore/'")
