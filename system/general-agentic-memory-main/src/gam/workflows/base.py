# -*- coding: utf-8 -*-
"""
GAM Workflow Base

Higher-level abstraction that wraps Agents, Generators, Workspaces, and Trees
into a simple unified interface. Users only need one class to do everything.

Usage:
    from gam import Workflow

    wf = Workflow("text", gam_dir="./my_gam", model="gpt-4o-mini", api_key="sk-xxx")
    wf.add("paper.pdf")
    result = wf.request("What is the main conclusion?")
    print(result.answer)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..generators.base import BaseGenerator
    from ..workspaces.base import BaseWorkspace
    from ..core.tree import BaseTree
    from ..schemas.chat_schemas import ChatResult


class WorkflowType(str, Enum):
    """Supported workflow types."""
    TEXT = "text"
    VIDEO = "video"
    AUDIO = "audio"   # reserved
    IMAGE = "image"   # reserved


class BaseWorkflow(ABC):
    """
    Base workflow — higher-level abstraction over GAM components.

    Handles common initialisation logic:
    - Build / reuse an LLM Generator
    - Build a Workspace (local file-system)
    - Load or create a Tree

    Subclasses implement type-specific ``add()`` and ``request()`` methods.
    """

    workflow_type: WorkflowType

    def __init__(
        self,
        gam_dir: str | Path,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        verbose: bool = True,
        generator: Optional["BaseGenerator"] = None,
        chat_model: str | None = None,
        chat_api_key: str | None = None,
        chat_api_base: str | None = None,
        chat_generator: Optional["BaseGenerator"] = None,
        **kwargs: Any,
    ):
        """
        Args:
            gam_dir: GAM root directory path (will be created if absent).
            model: LLM model name for the **GAM Agent** (memory building).
                   Env fallback: ``GAM_MODEL`` → default ``gpt-4o-mini``.
            api_key: API key for the **GAM Agent**.
                     Env fallback: ``GAM_API_KEY`` → ``OPENAI_API_KEY``.
            api_base: API base URL for the **GAM Agent**.
                      Env fallback: ``GAM_API_BASE`` → default ``https://api.openai.com/v1``.
            max_tokens: Max generation tokens.
            temperature: Sampling temperature.
            verbose: Whether to print progress information.
            generator: Pass an **existing** :class:`BaseGenerator` for the GAM Agent.
                       If provided, *model / api_key / api_base / max_tokens /
                       temperature* are ignored.
            chat_model: LLM model name for the **Chat Agent** (Q&A).
                        Env fallback: ``GAM_CHAT_MODEL`` → falls back to *model*.
            chat_api_key: API key for the **Chat Agent**.
                          Env fallback: ``GAM_CHAT_API_KEY`` → falls back to *api_key*.
            chat_api_base: API base URL for the **Chat Agent**.
                           Env fallback: ``GAM_CHAT_API_BASE`` → falls back to *api_base*.
            chat_generator: Pass an **existing** :class:`BaseGenerator` for the Chat Agent.
                            If provided, *chat_model / chat_api_key / chat_api_base* are ignored.
        """
        self.gam_dir = Path(gam_dir).resolve()
        self.verbose = verbose

        # GAM Agent LLM configuration (with env-var fallbacks)
        self.model = model or os.environ.get("GAM_MODEL", "gpt-4o-mini")
        self.api_key = (
            api_key
            or os.environ.get("GAM_API_KEY")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        self.api_base = (
            api_base
            or os.environ.get("GAM_API_BASE", "https://api.openai.com/v1")
        )
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Chat Agent LLM configuration (falls back to GAM Agent config)
        self.chat_model = (
            chat_model
            or os.environ.get("GAM_CHAT_MODEL")
        )
        self.chat_api_key = (
            chat_api_key
            or os.environ.get("GAM_CHAT_API_KEY")
        )
        self.chat_api_base = (
            chat_api_base
            or os.environ.get("GAM_CHAT_API_BASE")
        )

        # Allow injecting pre-built generators (for advanced users)
        self._generator_override = generator
        self._chat_generator_override = chat_generator

        # Lazy-init caches
        self._generator: Optional["BaseGenerator"] = None
        self._chat_generator: Optional["BaseGenerator"] = None
        self._workspace: Optional["BaseWorkspace"] = None
        self._tree: Optional["BaseTree"] = None

    # ------------------------------------------------------------------
    # Lazy properties
    # ------------------------------------------------------------------

    @property
    def generator(self) -> "BaseGenerator":
        """Return (and lazily build) the GAM Agent LLM generator."""
        if self._generator is None:
            self._generator = (
                self._generator_override
                if self._generator_override is not None
                else self._build_generator()
            )
        return self._generator

    @property
    def chat_generator(self) -> "BaseGenerator":
        """Return (and lazily build) the Chat Agent LLM generator.

        Falls back to the GAM Agent generator when no chat-specific
        configuration is provided.
        """
        if self._chat_generator is None:
            if self._chat_generator_override is not None:
                self._chat_generator = self._chat_generator_override
            elif self.chat_model or self.chat_api_key or self.chat_api_base:
                self._chat_generator = self._build_chat_generator()
            else:
                self._chat_generator = self.generator
        return self._chat_generator

    @property
    def workspace(self) -> "BaseWorkspace":
        """Return (and lazily build) the workspace."""
        if self._workspace is None:
            self._workspace = self._build_workspace()
        return self._workspace

    @property
    def tree(self) -> "BaseTree":
        """Return (and lazily load/create) the GAM tree."""
        if self._tree is None:
            self._tree = self._load_or_create_tree()
        return self._tree

    @tree.setter
    def tree(self, value: "BaseTree") -> None:
        self._tree = value

    # ------------------------------------------------------------------
    # Builder helpers
    # ------------------------------------------------------------------

    def _build_generator(self) -> "BaseGenerator":
        """Build an OpenAI-compatible generator for the GAM Agent."""
        from ..generators.openai_generator import OpenAIGenerator
        from ..generators.config import OpenAIGeneratorConfig

        config = OpenAIGeneratorConfig(
            model_name=self.model,
            api_key=self.api_key,
            base_url=self.api_base,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return OpenAIGenerator(config)

    def _build_chat_generator(self) -> "BaseGenerator":
        """Build an OpenAI-compatible generator for the Chat Agent.

        Uses chat-specific config with fallback to GAM Agent config.
        """
        from ..generators.openai_generator import OpenAIGenerator
        from ..generators.config import OpenAIGeneratorConfig

        config = OpenAIGeneratorConfig(
            model_name=self.chat_model or self.model,
            api_key=self.chat_api_key or self.api_key,
            base_url=self.chat_api_base or self.api_base,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return OpenAIGenerator(config)

    def _build_workspace(self) -> "BaseWorkspace":
        """Build a local workspace rooted at *gam_dir*."""
        from ..workspaces.local_workspace import LocalWorkspace

        self.gam_dir.mkdir(parents=True, exist_ok=True)
        return LocalWorkspace(root_path=str(self.gam_dir))

    @abstractmethod
    def _load_or_create_tree(self) -> "BaseTree":
        """Load tree from disk, or create an empty one."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Tree management
    # ------------------------------------------------------------------

    def reload_tree(self) -> "BaseTree":
        """Force-reload the tree from disk and return it."""
        self._tree = self._load_or_create_tree()
        return self._tree

    def get_tree_view(self, path: str = "/", depth: int = -1) -> str:
        """Return a human-readable tree-structure string."""
        if hasattr(self.tree, "tree_view"):
            return self.tree.tree_view(path, depth=depth)
        return "(tree view not available)"

    # ------------------------------------------------------------------
    # Public interface (implemented by subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def add(self, *args: Any, **kwargs: Any) -> Any:
        """Add content to GAM memory."""
        raise NotImplementedError

    @abstractmethod
    def request(self, question: str, **kwargs: Any) -> "ChatResult":
        """Run Q&A over GAM memory."""
        raise NotImplementedError

    def chat(self, question: str, **kwargs: Any) -> "ChatResult":
        """Convenience alias for :meth:`request`."""
        return self.request(question, **kwargs)

    # ------------------------------------------------------------------
    # Skill prompt
    # ------------------------------------------------------------------

    def get_skill_prompt(self) -> str:
        """
        Return the GAM *skill prompt* with this workflow's GAM directory
        baked in. Useful for injecting into an external agent's system prompt.
        """
        from ..prompts.skill_prompts import get_skill_prompt
        return get_skill_prompt(gam_dir=str(self.gam_dir))

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"type={self.workflow_type.value!r}, "
            f"gam_dir={str(self.gam_dir)!r}, "
            f"model={self.model!r})"
        )
