# -*- coding: utf-8 -*-
"""
TextWorkflow — Text-specific workflow.

Wraps TextGAMAgent (chunking → memory → organise) and TextChatAgent (Q&A)
behind a single, easy-to-use object.

Usage::

    from gam import Workflow

    wf = Workflow("text", gam_dir="./my_gam", model="gpt-4o-mini", api_key="sk-xxx")

    # Build memory
    wf.add("paper.pdf")
    wf.add(["paper1.pdf", "paper2.txt"])
    wf.add(content="Some text to memorize")

    # Q&A
    result = wf.request("What is the main conclusion?")
    print(result.answer)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from .base import BaseWorkflow, WorkflowType

# Type alias for convenience
PathLike = Union[str, Path]


class TextWorkflow(BaseWorkflow):
    """
    Text workflow: intelligent chunking → memory generation → directory
    organisation → exploration-based Q&A.

    All heavy imports are deferred so that instantiation is cheap.
    """

    workflow_type = WorkflowType.TEXT

    def __init__(
        self,
        gam_dir: PathLike = "./gam",
        *,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        verbose: bool = True,
        generator: Any | None = None,
        chat_model: str | None = None,
        chat_api_key: str | None = None,
        chat_api_base: str | None = None,
        chat_generator: Any | None = None,
        # ---- text-specific options ----
        use_chunking: bool = True,
        memory_workers: int = 4,
        max_iterations: int = 10,
        **kwargs: Any,
    ):
        """
        Args:
            gam_dir: GAM root directory.
            model / api_key / api_base / max_tokens / temperature:
                LLM configuration for the **GAM Agent** (see :class:`BaseWorkflow`).
            verbose: Print progress information.
            generator: Optional pre-built :class:`BaseGenerator` for the GAM Agent.
            chat_model / chat_api_key / chat_api_base:
                LLM configuration for the **Chat Agent** (Q&A).
                Falls back to the GAM Agent config when not specified.
            chat_generator: Optional pre-built :class:`BaseGenerator` for the Chat Agent.
            use_chunking: Split input text into semantic chunks (default True).
            memory_workers: Parallel workers for memory generation.
            max_iterations: Max exploration rounds for Q&A agent.
        """
        super().__init__(
            gam_dir=gam_dir,
            model=model,
            api_key=api_key,
            api_base=api_base,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=verbose,
            generator=generator,
            chat_model=chat_model,
            chat_api_key=chat_api_key,
            chat_api_base=chat_api_base,
            chat_generator=chat_generator,
            **kwargs,
        )
        self.use_chunking = use_chunking
        self.memory_workers = memory_workers
        self.max_iterations = max_iterations

        # Cached GAM agent (preserves incremental state across add() calls)
        self._gam_agent: Any | None = None

    # ------------------------------------------------------------------
    # Tree
    # ------------------------------------------------------------------

    def _load_or_create_tree(self):
        from ..core.tree import GAMTree

        self.gam_dir.mkdir(parents=True, exist_ok=True)
        try:
            return GAMTree.from_disk(self.gam_dir, self.workspace)
        except Exception:
            return GAMTree.create_empty(self.gam_dir, name=self.gam_dir.name)

    # ------------------------------------------------------------------
    # Lazy agent builders
    # ------------------------------------------------------------------

    @property
    def gam_agent(self):
        """Return (and lazily create) the :class:`TextGAMAgent`.

        The agent is cached so that incremental-add state (reorganisation
        counter, etc.) survives across multiple ``add()`` calls.
        """
        if self._gam_agent is None:
            from ..agents.text_gam_agent import TextGAMAgent

            self._gam_agent = TextGAMAgent(
                generator=self.generator,
                tree=self.tree,
                workspace=self.workspace,
                use_chunking=self.use_chunking,
                auto_save=True,
                verbose=self.verbose,
                memory_workers=self.memory_workers,
            )
        return self._gam_agent

    def _create_chat_agent(self):
        """Build a **fresh** :class:`TextChatAgent` with the latest tree."""
        from ..agents.text_chat_agent import TextChatAgent

        return TextChatAgent(
            generator=self.chat_generator,
            tree=self.reload_tree(),
            workspace=self.workspace,
            max_iterations=self.max_iterations,
            verbose=self.verbose,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        input_file: PathLike | List[PathLike] | None = None,
        content: str | List[str] | None = None,
        *,
        context: str = "",
        use_chunking: bool | None = None,
        output_dir: PathLike | None = None,
        force_reorganize: bool = False,
        callback: Callable | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Add files and / or text to the GAM knowledge base.

        First call creates the GAM from scratch; subsequent calls add
        incrementally.

        Args:
            input_file: File path(s) — ``str``, ``Path``, or a list of them.
                        Supported formats: PDF, TXT, MD, and other plain-text.
            content: Direct text content(s) — ``str`` or a list of ``str``.
            context: Optional contextual note for the LLM.
            use_chunking: Override the default chunking setting for this call.
            output_dir: Save raw chunks to this directory.
            force_reorganize: Force a taxonomy reorganisation.
            callback: Optional progress callback ``fn(event, data)``.

        Returns:
            ``BatchProcessingResult`` (first creation) or
            ``AddChunksResult`` (incremental addition).
        """
        # Normalise file paths
        if input_file is not None:
            if isinstance(input_file, (str, Path)):
                input_file = Path(input_file)
            else:
                input_file = [Path(f) for f in input_file]

        result = self.gam_agent.add(
            input_file=input_file,
            content=content,
            context=context,
            use_chunking=use_chunking,
            output_dir=output_dir,
            force_reorganize=force_reorganize,
            callback=callback,
            **kwargs,
        )

        # Sync the tree reference (the agent reloads it internally)
        self._tree = self.gam_agent.tree
        return result

    def request(
        self,
        question: str,
        *,
        system_prompt: str = "",
        max_iter: int | None = None,
        action_callback: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Ask a question over the GAM knowledge base.

        Internally creates a fresh :class:`TextChatAgent` (with the latest
        tree) and runs a multi-round ReAct exploration loop.

        Args:
            question: The question to answer.
            system_prompt: Optional system prompt for the Q&A agent.
            max_iter: Override the default max exploration rounds.
            action_callback: Optional callback ``fn(event, data)`` for
                             real-time tool-call notifications.

        Returns:
            :class:`ChatResult` with ``answer``, ``sources``, ``confidence``.
        """
        agent = self._create_chat_agent()
        return agent.request(
            system_prompt=system_prompt,
            user_prompt=question,
            max_iter=max_iter or self.max_iterations,
            action_callback=action_callback,
            **kwargs,
        )
