# -*- coding: utf-8 -*-
"""
Pydantic models for GAM REST API request/response validation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared generator config (overrides per-request)
# ---------------------------------------------------------------------------

class GeneratorParams(BaseModel):
    """LLM generator parameters that can be overridden per-request."""
    model: Optional[str] = Field(None, description="LLM model name")
    api_base: Optional[str] = Field(None, description="API base URL")
    api_key: Optional[str] = Field(None, description="API key")
    max_tokens: Optional[int] = Field(None, description="Max tokens to generate")
    temperature: Optional[float] = Field(None, description="Sampling temperature")


# ---------------------------------------------------------------------------
# Add requests
# ---------------------------------------------------------------------------

class TextAddRequest(GeneratorParams):
    """Request body for adding text content to a GAM."""
    type: str = Field("text", description="GAM type (fixed to 'text')")
    gam_dir: Optional[str] = Field(None, description="Path to GAM directory (auto-created if omitted)")
    input: Optional[Union[str, List[str]]] = Field(None, description="Input file path(s)")
    content: Optional[Union[str, List[str]]] = Field(None, description="Raw text content to add")
    context: str = Field("", description="Additional context for memorization")
    memorize_instruction: Optional[str] = Field(None, description="Custom memorization instruction")
    use_chunking: bool = Field(True, description="Whether to use intelligent chunking")
    memory_workers: int = Field(4, ge=1, description="Number of parallel workers for memorization")
    window_size: int = Field(30000, ge=1000, description="Chunking window size (tokens)")
    overlap_size: int = Field(10000, ge=0, description="Chunking overlap size (tokens)")
    max_splits: int = Field(120, ge=1, description="Maximum number of chunks")
    taxonomy_batch_size: int = Field(50, ge=1, description="Batch size for taxonomy assignment")
    force_reorganize: bool = Field(False, description="Force hierarchical reorganization")
    output_dir: Optional[str] = Field(None, description="Output directory for chunks")
    verbose: bool = Field(True, description="Enable verbose logging")


class VideoAddRequest(GeneratorParams):
    """Request body for adding video content to a GAM."""
    type: str = Field("video", description="GAM type (fixed to 'video')")
    gam_dir: Optional[str] = Field(None, description="Path to GAM directory (auto-created if omitted)")
    input: Union[str, List[str]] = Field(..., description="Video directory path(s)")
    video_path: Optional[str] = Field(None, description="Specific video file path")
    subtitle_path: Optional[str] = Field(None, description="Subtitle file path")
    caption_with_subtitles: bool = Field(True, description="Use subtitles for captioning")
    verbose: bool = Field(True, description="Enable verbose logging")
    segmentor_model: Optional[str] = Field(None, description="Segmentor model name (overrides main model)")
    segmentor_api_base: Optional[str] = Field(None, description="Segmentor API base URL")
    segmentor_api_key: Optional[str] = Field(None, description="Segmentor API key")
    segmentor_max_tokens: Optional[int] = Field(None, description="Segmentor max tokens")
    segmentor_temperature: Optional[float] = Field(None, description="Segmentor temperature")


class AddRequest(BaseModel):
    """Unified add request that dispatches by type."""
    type: str = Field("text", description="GAM type: 'text' or 'video'")
    gam_dir: Optional[str] = Field(None, description="Path to GAM directory")

    # Text-specific fields
    input: Optional[Union[str, List[str]]] = Field(None, description="Input file/video path(s)")
    content: Optional[Union[str, List[str]]] = Field(None, description="[Text] Raw text content")
    context: str = Field("", description="[Text] Additional context")
    memorize_instruction: Optional[str] = Field(None, description="[Text] Custom instruction")
    use_chunking: bool = Field(True, description="[Text] Intelligent chunking")
    memory_workers: int = Field(4, ge=1, description="[Text] Parallel workers")
    window_size: int = Field(30000, ge=1000, description="[Text] Window size")
    overlap_size: int = Field(10000, ge=0, description="[Text] Overlap size")
    max_splits: int = Field(120, ge=1, description="[Text] Max chunks")
    taxonomy_batch_size: int = Field(50, ge=1, description="[Text] Batch size")
    force_reorganize: bool = Field(False, description="[Text] Force reorganize")
    output_dir: Optional[str] = Field(None, description="[Text] Chunks output dir")

    # Video-specific fields
    video_path: Optional[str] = Field(None, description="[Video] Video file path")
    subtitle_path: Optional[str] = Field(None, description="[Video] Subtitle file path")
    caption_with_subtitles: bool = Field(True, description="[Video] Use subtitles")
    segmentor_model: Optional[str] = Field(None, description="[Video] Segmentor model")
    segmentor_api_base: Optional[str] = Field(None, description="[Video] Segmentor API base")
    segmentor_api_key: Optional[str] = Field(None, description="[Video] Segmentor API key")
    segmentor_max_tokens: Optional[int] = Field(None, description="[Video] Segmentor max tokens")
    segmentor_temperature: Optional[float] = Field(None, description="[Video] Segmentor temperature")

    # Generator override fields
    model: Optional[str] = Field(None, description="LLM model name")
    api_base: Optional[str] = Field(None, description="API base URL")
    api_key: Optional[str] = Field(None, description="API key")
    max_tokens: Optional[int] = Field(None, description="Max tokens")
    temperature: Optional[float] = Field(None, description="Temperature")
    verbose: bool = Field(True, description="Verbose logging")


# ---------------------------------------------------------------------------
# Request (query) requests
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Request body for querying a GAM knowledge base."""
    type: str = Field("text", description="GAM type: 'text' or 'video'")
    gam_dir: str = Field(..., description="Path to the GAM directory (required)")
    question: str = Field(..., description="User question (required)")
    system_prompt: str = Field("", description="Optional system prompt override")
    max_iter: int = Field(10, ge=1, le=100, description="Maximum exploration iterations")
    verbose: bool = Field(True, description="Enable verbose logging")

    # Generator override fields
    model: Optional[str] = Field(None, description="LLM model name")
    api_base: Optional[str] = Field(None, description="API base URL")
    api_key: Optional[str] = Field(None, description="API key")
    max_tokens: Optional[int] = Field(None, description="Max tokens")
    temperature: Optional[float] = Field(None, description="Temperature")

    # Video-specific fields
    video_model: Optional[str] = Field(None, description="[Video] Multimodal model for video inspection")
    video_api_base: Optional[str] = Field(None, description="[Video] Video model API base")
    video_api_key: Optional[str] = Field(None, description="[Video] Video model API key")
    video_max_tokens: Optional[int] = Field(None, description="[Video] Video model max tokens")
    video_temperature: Optional[float] = Field(None, description="[Video] Video model temperature")
    video_fps: float = Field(1.0, gt=0, description="[Video] Frames per second for inspection")
    video_max_resolution: int = Field(480, ge=144, description="[Video] Max video resolution")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class AddResponse(BaseModel):
    """Response for the add endpoint."""
    success: bool
    type: str
    gam_dir: str
    output_dir: Optional[str] = None
    created_files: Optional[int] = None
    new_directories: Optional[int] = None
    segment_num: Optional[int] = None


class QueryResponse(BaseModel):
    """Response for the query endpoint."""
    success: bool
    question: str
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""
    files_read: List[str] = Field(default_factory=list)
    dirs_explored: List[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Standard error response."""
    success: bool = False
    error: str
    detail: Optional[str] = None
