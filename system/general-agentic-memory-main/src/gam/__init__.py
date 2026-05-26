# -*- coding: utf-8 -*-
"""
GAM - General Agentic Memory

A hierarchical memory system for AI agents.
"""

__version__ = "0.1.0"

from .core.tree import BaseTree, GAMTree, VideoGAMTree
from .core.node import FSNode, NodeType
from .generators.base import BaseGenerator
from .generators.openai_generator import OpenAIGenerator
from .generators.sglang_generator import SGLangGenerator
from .generators.config import (
    GeneratorConfig,
    OpenAIGeneratorConfig,
    SGLangGeneratorConfig,
)
from .workflows.base import BaseWorkflow, WorkflowType
from .workflows import Workflow, TextWorkflow, VideoWorkflow
from .workspaces.base import BaseWorkspace
from .workspaces.local_workspace import LocalWorkspace
from .workspaces.docker_workspace import DockerWorkspace
from .agents.gam_agent import BaseGAMAgent
from .agents.chat_agent import BaseChatAgent
from .agents.text_gam_agent import TextGAMAgent
from .agents.text_chat_agent import TextChatAgent
from .agents.video_gam_agent import VideoGAMAgent
from .agents.video_chat_agent import VideoChatAgent
from .tools.base import BaseTool
from .tools.result import ToolResult
from .tools.ls_tool import LsTool
from .tools.cat_tool import CatTool
from .tools.grep_tool import GrepTool
from .tools.bm25_search_tool import BM25SearchTool
from .tools.inspect_video_tool import InspectVideoTool
from .readers.base import BaseReader
from .readers.pdf_reader import PdfReader
from .readers.txt_reader import TxtReader
from .schemas.chunk_schemas import (
    MemorizedChunk,
    ChunkResult,
    ChunkWithMemoryResult,
    BatchMemorizedChunk,
    DirectoryNode,
    BatchOrganizationPlan,
    BatchProcessingResult,
    TaxonomyNode,
    TaxonomyTree,
    ChunkAssignmentResult,
    AddChunksResult,
    IncrementalAddState,
)
from .schemas.chat_schemas import ChatResult
from .prompts.skill_prompts import GAM_SKILL_PROMPT, get_skill_prompt

__all__ = [
    # Core
    "BaseTree",
    "GAMTree",
    "VideoGAMTree",
    "FSNode",
    "NodeType",
    # Generators
    "BaseGenerator",
    "OpenAIGenerator",
    "SGLangGenerator",
    "GeneratorConfig",
    "OpenAIGeneratorConfig",
    "SGLangGeneratorConfig",
    # Workflows
    "Workflow",
    "BaseWorkflow",
    "WorkflowType",
    "TextWorkflow",
    "VideoWorkflow",
    # Workspaces
    "BaseWorkspace",
    "LocalWorkspace",
    "DockerWorkspace",
    # Agents
    "BaseGAMAgent",
    "BaseChatAgent",
    "TextGAMAgent",
    "TextChatAgent",
    "VideoGAMAgent",
    "VideoChatAgent",
    # Tools
    "BaseTool",
    "ToolResult",
    "LsTool",
    "CatTool",
    "GrepTool",
    "BM25SearchTool",
    "InspectVideoTool",
    # Readers
    "BaseReader",
    "PdfReader",
    "TxtReader",
    # Schemas
    "MemorizedChunk",
    "ChunkResult",
    "ChunkWithMemoryResult",
    "BatchMemorizedChunk",
    "DirectoryNode",
    "BatchOrganizationPlan",
    "BatchProcessingResult",
    "TaxonomyNode",
    "TaxonomyTree",
    "ChunkAssignmentResult",
    "AddChunksResult",
    "IncrementalAddState",
    "ChatResult",
    # Skill
    "GAM_SKILL_PROMPT",
    "get_skill_prompt",
]
