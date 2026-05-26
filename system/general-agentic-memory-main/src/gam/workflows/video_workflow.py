# -*- coding: utf-8 -*-
"""
VideoWorkflow — Video-specific workflow.

Wraps VideoGAMAgent (probe → segment → describe → organise) and
VideoChatAgent (text + visual Q&A) behind a single object.

Usage::

    from gam import Workflow

    wf = Workflow(
        "video",
        gam_dir="./my_video_gam",
        model="gpt-4o",
        api_key="sk-xxx",
    )

    # Build video memory
    wf.add("./video_dir")

    # Q&A (text + optional visual inspection)
    result = wf.request("What happens in the video?")
    print(result.answer)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

from .base import BaseWorkflow, WorkflowType

PathLike = Union[str, Path]


class VideoWorkflow(BaseWorkflow):
    """
    Video workflow: probe → segment → caption → global summary → Q&A.

    Supports dual-LLM architecture:
    - *planner / main* generator: video probing & global planning + Q&A agent.
    - *segmentor* generator: segment-level description (defaults to main).
    - *video_generator*: multimodal LLM for ``inspect_video`` tool in Q&A
      (defaults to main).
    """

    workflow_type = WorkflowType.VIDEO

    def __init__(
        self,
        gam_dir: PathLike = "./video_gam",
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
        # ---- video-specific options ----
        segmentor_model: str | None = None,
        segmentor_api_base: str | None = None,
        segmentor_api_key: str | None = None,
        video_model: str | None = None,
        video_api_base: str | None = None,
        video_api_key: str | None = None,
        video_fps: float = 1.0,
        video_max_resolution: int = 480,
        max_iterations: int = 10,
        **kwargs: Any,
    ):
        """
        Args:
            gam_dir: GAM root directory.
            model / api_key / api_base / max_tokens / temperature:
                LLM configuration for the *planner* / main model
                (see :class:`BaseWorkflow`).
            verbose: Print progress information.
            generator: Optional pre-built :class:`BaseGenerator` for the main model.
            chat_model / chat_api_key / chat_api_base:
                LLM configuration for the **Chat Agent** (Q&A).
                Falls back to the GAM Agent config when not specified.
            chat_generator: Optional pre-built :class:`BaseGenerator` for the Chat Agent.
            segmentor_model: Model for segment-level description (defaults to *model*).
            segmentor_api_base: API base for segmentor (defaults to *api_base*).
            segmentor_api_key: API key for segmentor (defaults to *api_key*).
            video_model: Multimodal model for ``inspect_video`` tool in Q&A
                         (defaults to *model*).
            video_api_base: API base for video model.
            video_api_key: API key for video model.
            video_fps: Frame sampling rate for ``inspect_video`` (default 1.0).
            video_max_resolution: Max video resolution for analysis (default 480).
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

        # Segmentor LLM config (fallback to main)
        self.segmentor_model = segmentor_model
        self.segmentor_api_base = segmentor_api_base
        self.segmentor_api_key = segmentor_api_key

        # Video / multimodal LLM config (fallback to main)
        self.video_model = video_model
        self.video_api_base = video_api_base
        self.video_api_key = video_api_key

        self.video_fps = video_fps
        self.video_max_resolution = video_max_resolution
        self.max_iterations = max_iterations

        # Lazy caches
        self._segmentor: Any | None = None
        self._video_generator: Any | None = None

    # ------------------------------------------------------------------
    # Tree
    # ------------------------------------------------------------------

    def _load_or_create_tree(self):
        from ..core.tree import VideoGAMTree

        self.gam_dir.mkdir(parents=True, exist_ok=True)
        try:
            return VideoGAMTree.from_disk(self.gam_dir, self.workspace)
        except Exception:
            return VideoGAMTree.create_empty(self.gam_dir, name=self.gam_dir.name)

    # ------------------------------------------------------------------
    # Extra generators
    # ------------------------------------------------------------------

    @property
    def segmentor(self):
        """Segmentor generator (lazy, defaults to main generator)."""
        if self._segmentor is None:
            if self.segmentor_model:
                self._segmentor = self._build_extra_generator(
                    model=self.segmentor_model,
                    api_base=self.segmentor_api_base,
                    api_key=self.segmentor_api_key,
                )
            else:
                self._segmentor = self.generator
        return self._segmentor

    @property
    def video_generator(self):
        """Multimodal generator for ``inspect_video`` (lazy, defaults to main)."""
        if self._video_generator is None:
            if self.video_model:
                self._video_generator = self._build_extra_generator(
                    model=self.video_model,
                    api_base=self.video_api_base,
                    api_key=self.video_api_key,
                )
            else:
                self._video_generator = None  # VideoChatAgent will reuse main
        return self._video_generator

    def _build_extra_generator(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
    ):
        """Build an auxiliary OpenAI-compatible generator."""
        from ..generators.openai_generator import OpenAIGenerator
        from ..generators.config import OpenAIGeneratorConfig

        config = OpenAIGeneratorConfig(
            model_name=model,
            base_url=api_base or self.api_base,
            api_key=api_key or self.api_key,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return OpenAIGenerator(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        input_path: PathLike | None = None,
        *,
        caption_with_subtitles: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Build Video GAM from a video directory.

        The input directory should contain:
        - ``video.mp4`` (required)
        - ``subtitles.srt`` (optional)
        - ``metadata.json`` (optional)

        Args:
            input_path: Path to the video directory.
            caption_with_subtitles: Include subtitles when generating
                                    segment descriptions (default True).

        Returns:
            ``CreateVideoGAMResult`` with segment information.
        """
        from ..agents.video_gam_agent import VideoGAMAgent

        agent = VideoGAMAgent(
            planner=self.generator,
            segmentor=self.segmentor,
            workspace=self.workspace,
            tree=self.tree,
        )

        result = agent.add(
            input_path=Path(input_path) if input_path else None,
            verbose=self.verbose,
            caption_with_subtitles=caption_with_subtitles,
            **kwargs,
        )

        # Sync tree
        self._tree = agent.tree
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
        Ask a question over the Video GAM knowledge base.

        Supports text tools (ls, cat, grep) and visual inspection
        (``inspect_video``) via a multimodal LLM.

        Args:
            question: The question to answer.
            system_prompt: Optional system prompt.
            max_iter: Override max exploration rounds.
            action_callback: Optional callback ``fn(event, data)``.

        Returns:
            :class:`ChatResult` with ``answer``, ``sources``, ``confidence``.
        """
        from ..agents.video_chat_agent import VideoChatAgent

        agent = VideoChatAgent(
            generator=self.chat_generator,
            tree=self.reload_tree(),
            workspace=self.workspace,
            video_generator=self.video_generator,
            max_iterations=max_iter or self.max_iterations,
            verbose=self.verbose,
            video_fps=self.video_fps,
            video_max_resolution=self.video_max_resolution,
        )

        return agent.request(
            system_prompt=system_prompt,
            user_prompt=question,
            max_iter=max_iter or self.max_iterations,
            action_callback=action_callback,
            **kwargs,
        )
