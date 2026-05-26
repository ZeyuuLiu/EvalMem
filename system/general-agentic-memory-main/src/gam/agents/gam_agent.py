from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from ..generators.base import BaseGenerator
from ..core.tree import BaseTree
from ..workspaces.base import BaseWorkspace


class BaseGAMAgent(ABC):
    """
    Base GAM abstraction.

    Holds generator (LLM/API client), tree (memory file system view), and workspace (command executor).
    
    Architecture:
    - tree: READ-ONLY view of the file system structure
    - workspace: Executes Linux commands (mkdir, touch, echo, etc.) to modify the file system
    - generator: LLM for intelligent decision making
    
    Usage:
        When the agent wants to create a directory:
            workspace.run("mkdir -p /path/to/dir")
        
        When the agent wants to create a file:
            workspace.run("echo 'content' > /path/to/file.md")
        
        After modifications, reload the tree:
            tree = tree.reload(workspace)
    
    Public interface:
        add() - The single unified entry point.
            - If the GAM is empty (no existing content), performs a full creation.
            - If the GAM already has content, performs an incremental addition.
    """

    def __init__(
        self,
        generator: BaseGenerator,
        tree: BaseTree,
        workspace: Optional[BaseWorkspace] = None,
    ):
        self.generator = generator
        self.tree = tree
        self.workspace = workspace

    def run_command(self, command: str, workdir: str = None) -> tuple[str, str]:
        """
        Execute a Linux command through the workspace.
        
        Args:
            command: Linux command to execute (e.g., "mkdir -p /dir", "echo 'text' > file.md")
            workdir: Working directory for the command
            
        Returns:
            Tuple of (output, exit_code_or_error)
        """
        if self.workspace is None:
            raise RuntimeError("No workspace configured. Cannot execute commands.")
        return self.workspace.run(command, workdir=workdir)
    
    def reload_tree(self) -> None:
        """Reload the tree from disk after workspace modifications."""
        if self.workspace is None:
            raise RuntimeError("No workspace configured. Cannot reload tree.")
        if hasattr(self.tree, 'reload'):
            self.tree = self.tree.reload(self.workspace)

    @abstractmethod
    def add(
        self,
        input_file: Path,
        segmentation_request: str = "",
    ) -> Any:
        """
        Add memory from an input file.
        
        This is the single unified entry point:
        - If the GAM is empty, performs a full creation (generates taxonomy, organizes).
        - If the GAM already has content, performs an incremental addition.
        
        Args:
            input_file: Path to input file
            segmentation_request: Optional context/instruction for segmentation
            
        Returns:
            Result of the operation
        """
        raise NotImplementedError
