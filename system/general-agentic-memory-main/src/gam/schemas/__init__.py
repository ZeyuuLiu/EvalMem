# -*- coding: utf-8 -*-
"""
GAM Schemas Module

Data schemas for chunking, organization, and chat operations.
"""

from .chunk_schemas import (
    # Chunking
    MemorizedChunk,
    ChunkResult,
    ChunkWithMemoryResult,
    # Organization
    BatchMemorizedChunk,
    DirectoryNode,
    BatchOrganizationPlan,
    BatchProcessingResult,
    # Taxonomy
    TaxonomyNode,
    TaxonomyTree,
    TaxonomyOrganizationPlan,
    ChunkAssignmentResult,
    # Incremental add
    ChunkAddAssignment,
    NewDirectoryInfo,
    AddChunksResult,
    ReorganizeOperation,
    ReorganizeResult,
    IncrementalAddState,
    # JSON Schemas
    MEMORIZE_CHUNK_SCHEMA,
    CHUNK_FORMAT_ANALYSIS_SCHEMA,
    MEMORY_FORMAT_ANALYSIS_SCHEMA,
    BATCH_ORGANIZATION_SCHEMA,
    TAXONOMY_NODE_SCHEMA,
    GENERATE_TAXONOMY_SCHEMA,
    ADJUST_TAXONOMY_SCHEMA,
    ASSIGN_CHUNK_SCHEMA,
    SHOULD_ADD_TO_EXISTING_SCHEMA,
    ADD_CHUNKS_SCHEMA,
    REORGANIZE_TAXONOMY_SCHEMA,
    GAM_GENERATE_README_SCHEMA,
    MERGE_ROOT_README_SCHEMA,
)
from .chat_schemas import (
    ChatResult,
)

from .video_schemas import (
    VideoGlobal,
    VideoSeg,
    SamplingConfig,
    VideoProcessSpec,
    SegmentSpec,
    CaptionSpec,
    DEFAULT_STRATEGY_PACKAGE,
    ALLOWED_GENRES,
    ALLOWED_STRUCTURE_MODES,
    ALLOWED_GRANULARITY,
    ALLOWED_EVIDENCE,
    DESCRIPTION_SLOTS,
    CreateVideoGAMResult,
)

__all__ = [
    # Chunking
    "MemorizedChunk",
    "ChunkResult",
    "ChunkWithMemoryResult",
    # Organization
    "BatchMemorizedChunk",
    "DirectoryNode",
    "BatchOrganizationPlan",
    "BatchProcessingResult",
    # Taxonomy
    "TaxonomyNode",
    "TaxonomyTree",
    "TaxonomyOrganizationPlan",
    "ChunkAssignmentResult",
    # Incremental add
    "ChunkAddAssignment",
    "NewDirectoryInfo",
    "AddChunksResult",
    "ReorganizeOperation",
    "ReorganizeResult",
    "IncrementalAddState",
    # JSON Schemas
    "MEMORIZE_CHUNK_SCHEMA",
    "CHUNK_FORMAT_ANALYSIS_SCHEMA",
    "MEMORY_FORMAT_ANALYSIS_SCHEMA",
    "BATCH_ORGANIZATION_SCHEMA",
    "TAXONOMY_NODE_SCHEMA",
    "GENERATE_TAXONOMY_SCHEMA",
    "ADJUST_TAXONOMY_SCHEMA",
    "ASSIGN_CHUNK_SCHEMA",
    "SHOULD_ADD_TO_EXISTING_SCHEMA",
    "ADD_CHUNKS_SCHEMA",
    "REORGANIZE_TAXONOMY_SCHEMA",
    "GAM_GENERATE_README_SCHEMA",
    "MERGE_ROOT_README_SCHEMA",
    # Chat
    "ChatResult",
    
    # Video
    "VideoGlobal",
    "VideoSeg",
    "SamplingConfig",
    "VideoProcessSpec",
    "SegmentSpec",
    "CaptionSpec",
    "DEFAULT_STRATEGY_PACKAGE",
    "ALLOWED_GENRES",
    "ALLOWED_STRUCTURE_MODES",
    "ALLOWED_GRANULARITY",
    "ALLOWED_EVIDENCE",
    "DESCRIPTION_SLOTS",
    "CreateVideoGAMResult"
]
