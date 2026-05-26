# -*- coding: utf-8 -*-
"""
Chunk Schemas

Data schemas for chunking and organization operations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


# ============== Context Generation Schemas ==============

SEG_TEMPLATE="""# Segment Context

**SegID**: {seg_id}

**Start Time**: {start_time}

**End Time**: {end_time}

**Duration**: {duration}

**Title**: {seg_title}

**Summary**: {summary}

**Detail Description**: {detail}

# Additional Files
- Raw video for this segment: `./video_clip.mp4`
- Subtitles for this segment: `./SUBTITLES.md`
""".strip()

@dataclass
class VideoSeg:
    """记忆化的 chunk，包含 title、memory、tldr 和原始内容"""
    seg_id: str  # 分段 ID
    start_time: float  # 开始时间
    end_time: float  # 结束时间
    duration: float  # 持续时间
    seg_title: str  # 分段标题
    summary: str  # 生成的标题
    detail: str  # 记忆/摘要

    def to_markdown(self, with_subtitles: bool = True) -> str:
        """转换为 markdown 格式"""
        
        mk_str = SEG_TEMPLATE.format(
            seg_id=self.seg_id,
            seg_title=self.seg_title,
            summary=self.summary,
            start_time=self.start_time,
            end_time=self.end_time,
            duration=self.duration,
            detail=self.detail
        )
        if not with_subtitles:
            mk_str = mk_str.replace(
                '- Subtitles for this segment: `./SUBTITLES.md`',
                ''
            )
        
        return mk_str

# GLOBAL_TEMPLATE="""# Video Global Context

# **Title**: {title}

# **Abstract**: {abstract}

# **Duration**: {duration} seconds

# **Segments**: {num_segments}

# # Segmentation Context

# ## Structure and Content
# - Each segment is saved in `./segments/seg_xxxx/`.
# - Each segment includes:
# - A `README.md` file containing the title, description, start time, and end time.
# - A `video.mp4` file with the corresponding video clip.
# - A `SUBTITLES.md` file with the corresponding subtitles.

# ## Segments Quickview
# Total: {num_segments} segments:
# ```
# {segments_quickview}
# ```

# # Additional Files
# - Raw video: `./video.mp4`
# - Detail information of segments: `./segments/`
# - Full subtitles for this video: `./SUBTITLES.md`
# """.strip()

GLOBAL_TEMPLATE="""
**Title**: {title}

**Duration**: {duration} seconds

**Abstract**: {abstract}

# Segmentation Context
There are {num_segments} segments are extracted from raw video.
- Each segment is saved in `./segments`.
- Each segment includes:
- A `README.md` file containing the title, description, start time, and end time.
- A `video.mp4` file with the corresponding video clip.
- A `SUBTITLES.md` file with the corresponding subtitles.

# Additional Files
- Raw video: `./video.mp4`
- Detail information of segments: `./segments/`
- Full subtitles for this video: `./SUBTITLES.md`
""".strip()

@dataclass
class VideoGlobal:
    """记忆化的 chunk，包含 title、memory、tldr 和原始内容"""
    title: str  # 视频标题
    abstract: str  # 视频描述
    num_segments: int  # 分段数量
    segments_quickview: str  # 分段快速查看`
    duration: float  # 持续时间

    def to_markdown(self, with_subtitles: bool = True) -> str:
        """转换为 markdown 格式"""
        
        mk_str = GLOBAL_TEMPLATE.format(
            title=self.title,
            duration=self.duration,
            abstract=self.abstract,
            num_segments=self.num_segments,
            # segments_quickview=self.segments_quickview
        )
        if not with_subtitles:
            mk_str = mk_str.replace(
                '- Full subtitles for this video: `./SUBTITLES.md`',
                ''
            )
            mk_str = mk_str.replace(
                '- A `SUBTITLES.md` file with the corresponding subtitles.',
                ''
            )
        
        return mk_str
        
        
@dataclass
class SamplingConfig:
    fps: float
    max_resolution: int
    use_subtitles: bool


@dataclass
class SegmentSpec:
    genre_str: str
    mode_str: str
    signal_audio_priority: str
    signal_visual_priority: str
    target_segment_length_sec: str
    boundary_evidence_primary: str
    boundary_evidence_secondary: str


@dataclass
class CaptionSpec:
    genre_str: str
    mode_str: str
    signal_audio_priority: str
    signal_visual_priority: str
    slots_weight: str
    notes: str


@dataclass
class VideoProcessSpec:
    segment_spec: SegmentSpec
    caption_spec: CaptionSpec
    segmentation_sampling: SamplingConfig
    description_sampling: SamplingConfig
    normalized_strategy: Dict[str, Any]
    
# ----------------------------
# Default strategy (fallback)
# ----------------------------
DEFAULT_STRATEGY_PACKAGE: Dict[str, Any] = {
    "planner_confidence": 0.25,
    "genre_distribution": {"other": 0.6, "vlog_lifestyle": 0.2, "podcast_interview": 0.2},
    "structure_mode": {"primary": "other", "secondary": []},
    "signal_priority": {
        "audio_text": 0.6,
        "visual": 0.4,
        "rationale": "Probe unavailable; use a conservative hybrid strategy relying slightly more on subtitles/audio-text.",
    },
    "segmentation": {
        "granularity": "hybrid",
        "target_segment_length_sec": [90, 480],
        "boundary_evidence_primary": ["topic_shift_in_subtitles", "scene_location_change"],
        "boundary_evidence_secondary": ["speaker_change", "on_screen_text_title_change"],
        "sampling": {"fps": 0.5, "max_resolution": 384, "use_subtitles": True},
        "notes": (
            "Conservative segmentation: prefer fewer, self-contained segments. "
            "Avoid cutting on minor shot changes or filler. Only cut when there is clear topic/scene change."
        ),
    },
    "description": {
        "slots_weight": {
            "cast_speaker": 0.18,
            "setting": 0.12,
            "core_events": 0.22,
            "topic_claims": 0.22,
            "outcome_progress": 0.18,
            "notable_cues": 0.08,
        },
        "sampling": {"fps": 0.2, "max_resolution": 384, "use_subtitles": True},
        "notes": (
            "Use a stable slot-based description. Prioritize who/where/what + main topic or key events. "
            "Do not narrate frame-by-frame; produce concise, segment-level summaries."
        ),
    },
}


# ----------------------------
# Enums / Allowed values
# ----------------------------
ALLOWED_GENRES = {
    "narrative_film",
    "animation",
    "vlog_lifestyle",
    "podcast_interview",
    "lecture_talk",
    "tutorial_howto",
    "news_report",
    "documentary",
    "gameplay",
    "compilation_montage",
    "sports_event",
    "other",
}

ALLOWED_STRUCTURE_MODES = {
    "turn_taking_qa",
    "lecture_slide_driven",
    "narrative_scene_based",
    "chronological_vlog",
    "step_by_step_procedure",
    "news_segmented",
    "compilation_blocks",
    "sports_play_by_play",
    "other",
}

ALLOWED_GRANULARITY = {"scene", "topic_block", "step_block", "hybrid"}

ALLOWED_EVIDENCE = {
    "topic_shift_in_subtitles",
    "speaker_change",
    "scene_location_change",
    "shot_style_change",
    "on_screen_text_title_change",
    "music_or_audio_pattern_change",
    "step_transition",
    "time_jump_or_recap",
    "other",
}

DESCRIPTION_SLOTS = [
    "cast_speaker",
    "setting",
    "core_events",
    "topic_claims",
    "outcome_progress",
    "notable_cues",
]


@dataclass
class CreateVideoGAMResult:
    success: bool = True
    segment_num: int = 0
    segmentation_sampling: SamplingConfig = field(default_factory=SamplingConfig)
    description_sampling: SamplingConfig = field(default_factory=SamplingConfig)
    segment_spec: SegmentSpec = field(default_factory=SegmentSpec)
    caption_spec: CaptionSpec = field(default_factory=CaptionSpec)
    