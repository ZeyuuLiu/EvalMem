# -*- coding: utf-8 -*-
"""
GAM REST API route handlers.
"""

from __future__ import annotations

import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from ..core.tree import GAMTree, VideoGAMTree
from ..generators.config import OpenAIGeneratorConfig
from ..generators.openai_generator import OpenAIGenerator
from ..workspaces.local_workspace import LocalWorkspace
from .models import AddRequest, AddResponse, ErrorResponse, QueryRequest, QueryResponse

router = APIRouter(prefix="/api/v1", tags=["GAM"])

_default_generator: Optional[OpenAIGenerator] = None
_default_video_generator: Optional[OpenAIGenerator] = None
_default_output_base: str = ""


def configure(
    generator: Optional[OpenAIGenerator] = None,
    video_generator: Optional[OpenAIGenerator] = None,
    output_base: str = "",
) -> None:
    """Set module-level defaults (called once at startup)."""
    global _default_generator, _default_video_generator, _default_output_base
    _default_generator = generator
    _default_video_generator = video_generator
    _default_output_base = output_base


def _get_timestamp_dir() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def _build_generator(
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> OpenAIGenerator:
    """Build a generator, merging per-request overrides with defaults."""
    base = _default_generator

    config = OpenAIGeneratorConfig(
        model_name=model or (base.config.model_name if base else "gpt-4o-mini"),
        base_url=api_base or (base.config.base_url if base else "https://api.openai.com/v1"),
        api_key=api_key or (base.config.api_key if base else os.environ.get("OPENAI_API_KEY", "")),
        max_tokens=max_tokens or (base.config.max_tokens if base else 4096),
        temperature=temperature if temperature is not None else (base.config.temperature if base else 0.3),
    )
    return OpenAIGenerator(config)


# ---------------------------------------------------------------------------
# POST /api/v1/add
# ---------------------------------------------------------------------------

@router.post(
    "/add",
    response_model=AddResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Add content to a GAM",
    description="Add text or video content to a GAM knowledge base. "
                "If `gam_dir` is omitted a new GAM is created automatically.",
)
async def add_content(body: AddRequest) -> AddResponse:
    try:
        gam_type = body.type

        if body.gam_dir:
            gam_dir = Path(body.gam_dir)
        else:
            timestamp = _get_timestamp_dir()
            gam_dir = Path(_default_output_base or "/tmp/gam_output") / timestamp / "gam"

        gam_dir.mkdir(parents=True, exist_ok=True)
        workspace = LocalWorkspace(root_path=str(gam_dir))
        generator = _build_generator(
            model=body.model,
            api_base=body.api_base,
            api_key=body.api_key,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
        )

        if gam_type == "text":
            return await _add_text(body, gam_dir, workspace, generator)
        elif gam_type == "video":
            return await _add_video(body, gam_dir, workspace, generator)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported GAM type: {gam_type}")

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


async def _add_text(
    body: AddRequest,
    gam_dir: Path,
    workspace: LocalWorkspace,
    generator: OpenAIGenerator,
) -> AddResponse:
    from ..agents.text_gam_agent import TextGAMAgent

    try:
        tree = GAMTree.from_disk(gam_dir, workspace)
    except Exception:
        tree = GAMTree.create(gam_dir, name=gam_dir.name)

    agent = TextGAMAgent(
        generator=generator,
        tree=tree,
        workspace=workspace,
        use_chunking=body.use_chunking,
        window_size=body.window_size,
        overlap_size=body.overlap_size,
        auto_save=True,
        verbose=body.verbose,
        memory_workers=body.memory_workers,
    )

    inputs = None
    if body.input:
        if isinstance(body.input, str):
            inputs = [Path(body.input)]
        else:
            inputs = [Path(f) for f in body.input]

    contents = None
    if body.content:
        if isinstance(body.content, str):
            contents = [body.content]
        else:
            contents = list(body.content)

    if body.output_dir:
        output_dir = Path(body.output_dir)
    else:
        output_dir = gam_dir.parent / f"chunks_{_get_timestamp_dir()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = agent.add(
        input_file=inputs,
        content=contents,
        context=body.context,
        memorize_instruction=body.memorize_instruction,
        max_splits=body.max_splits,
        taxonomy_batch_size=body.taxonomy_batch_size,
        output_dir=output_dir,
        force_reorganize=body.force_reorganize,
    )

    return AddResponse(
        success=True,
        type="text",
        gam_dir=str(gam_dir),
        output_dir=str(output_dir),
        created_files=len(getattr(result, "created_files", [])),
        new_directories=len(getattr(result, "new_directories", [])),
    )


async def _add_video(
    body: AddRequest,
    gam_dir: Path,
    workspace: LocalWorkspace,
    generator: OpenAIGenerator,
) -> AddResponse:
    from ..agents.video_gam_agent import VideoGAMAgent

    try:
        tree = VideoGAMTree.from_disk(gam_dir, workspace)
    except Exception:
        tree = VideoGAMTree.create_empty(gam_dir, name=gam_dir.name)

    if body.segmentor_model:
        segmentor = _build_generator(
            model=body.segmentor_model,
            api_base=body.segmentor_api_base,
            api_key=body.segmentor_api_key,
            max_tokens=body.segmentor_max_tokens,
            temperature=body.segmentor_temperature,
        )
    else:
        segmentor = generator

    agent = VideoGAMAgent(
        planner=generator,
        segmentor=segmentor,
        workspace=workspace,
        tree=tree,
    )

    if not body.input:
        raise HTTPException(status_code=400, detail="Video type requires 'input' (video directory path)")

    if isinstance(body.input, list):
        input_path = Path(body.input[0]).resolve()
    else:
        input_path = Path(body.input).resolve()

    video_path = Path(body.video_path).resolve() if body.video_path else None
    subtitle_path = Path(body.subtitle_path).resolve() if body.subtitle_path else None

    result = agent.add(
        input_path=input_path,
        video_path=video_path,
        subtitle_path=subtitle_path,
        verbose=body.verbose,
        caption_with_subtitles=body.caption_with_subtitles,
    )

    return AddResponse(
        success=True,
        type="video",
        gam_dir=str(gam_dir),
        segment_num=getattr(result, "segment_num", 0),
    )


# ---------------------------------------------------------------------------
# POST /api/v1/query
# ---------------------------------------------------------------------------

@router.post(
    "/query",
    response_model=QueryResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Query a GAM knowledge base",
    description="Ask a question to an existing GAM. "
                "The agent explores the GAM file system and returns an answer.",
)
async def query_gam(body: QueryRequest) -> QueryResponse:
    try:
        gam_dir = Path(body.gam_dir)
        if not gam_dir.exists():
            raise HTTPException(status_code=400, detail=f"GAM directory not found: {body.gam_dir}")

        workspace = LocalWorkspace(root_path=str(gam_dir))
        generator = _build_generator(
            model=body.model,
            api_base=body.api_base,
            api_key=body.api_key,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
        )

        gam_type = body.type

        if gam_type == "text":
            result = await _query_text(body, gam_dir, workspace, generator)
        elif gam_type == "video":
            result = await _query_video(body, gam_dir, workspace, generator)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported GAM type: {gam_type}")

        return QueryResponse(
            success=True,
            question=result.question,
            answer=result.answer,
            sources=result.sources,
            confidence=result.confidence,
            notes=result.notes,
            files_read=result.files_read,
            dirs_explored=result.dirs_explored,
        )

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


async def _query_text(body, gam_dir, workspace, generator):
    from ..agents.text_chat_agent import TextChatAgent

    tree = GAMTree.from_disk(gam_dir, workspace)
    agent = TextChatAgent(
        generator=generator,
        tree=tree,
        workspace=workspace,
        max_iterations=body.max_iter,
        verbose=body.verbose,
    )
    return agent.request(
        system_prompt=body.system_prompt,
        user_prompt=body.question,
        max_iter=body.max_iter,
    )


async def _query_video(body, gam_dir, workspace, generator):
    from ..agents.video_chat_agent import VideoChatAgent

    tree = VideoGAMTree.from_disk(gam_dir, workspace)

    if body.video_model:
        video_generator = _build_generator(
            model=body.video_model,
            api_base=body.video_api_base,
            api_key=body.video_api_key,
            max_tokens=body.video_max_tokens,
            temperature=body.video_temperature,
        )
    else:
        video_generator = _default_video_generator

    agent = VideoChatAgent(
        generator=generator,
        tree=tree,
        workspace=workspace,
        video_generator=video_generator,
        max_iterations=body.max_iter,
        verbose=body.verbose,
        video_fps=body.video_fps,
        video_max_resolution=body.video_max_resolution,
    )
    return agent.request(
        system_prompt=body.system_prompt,
        user_prompt=body.question,
        max_iter=body.max_iter,
    )
