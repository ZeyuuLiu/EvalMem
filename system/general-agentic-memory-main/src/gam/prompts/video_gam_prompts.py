# -*- coding: utf-8 -*-
"""
GAM Agent Prompts

Prompts for directory organization, README generation, and taxonomy management.
"""

VIDEO_PROBE_PROMPT = {
"SYSTEM": """You are a "video processing strategy planner". Given a few probes from a video, your goal is to produce an executable, cost-aware, and robust strategy configuration that will drive the SAME multimodal LLM to do:
1) full-video segmentation
2) per-segment description
You MUST output strict JSON only. Do not output any extra text.""",

"USER": """
You will receive:
- 3 probes sampled from the video at progress 25%, 50%, and 75%. Each probe contains:
  - a sequence of frames (in temporal order)
  - subtitle text for the same time window (may be noisy/incomplete)
- Global statistics for the whole video: total duration, subtitle density (chars/min or tokens/min), and optionally other stats.

Your task:
Based ONLY on the provided probes and global stats, output a "Strategy Package" (JSON) to drive the multimodal LLM for:
1) segmentation
2) description

HARD CONSTRAINTS
1) Use ONLY the probes + global stats. Do NOT hallucinate specific plot details. If uncertain, reflect that via lower confidence and/or conservative strategy.
2) Prefer LOW-COST but ROBUST strategies. Unless visual detail is clearly required, reduce fps/resolution and rely more on subtitles/audio-text.
3) genre_distribution MUST contain 2–4 genres and MUST sum to 1.
4) structure_mode MUST pick 1 primary mode from the enum; you may provide 0–2 secondary modes.
5) signal_priority MUST provide weights for audio_text and visual (0..1, sum to 1) plus a ONE-SENTENCE rationale.
6) segmentation.sampling and description.sampling may differ, but BOTH must specify fps and max_resolution.
7) Output MUST be valid JSON following the schema below. No markdown, no comments, no extra keys.

ENUMS (MUST USE)
GENRE OPTIONS
- narrative_film
- animation
- vlog_lifestyle
- podcast_interview
- lecture_talk
- tutorial_howto
- news_report
- documentary
- gameplay
- compilation_montage
- sports_event
- other

STRUCTURE_MODE OPTIONS
- turn_taking_qa
- lecture_slide_driven
- narrative_scene_based
- chronological_vlog
- step_by_step_procedure
- news_segmented
- compilation_blocks
- sports_play_by_play
- other

BOUNDARY EVIDENCE TYPES (for segmentation)
- topic_shift_in_subtitles
- speaker_change
- scene_location_change
- shot_style_change
- on_screen_text_title_change
- music_or_audio_pattern_change
- step_transition
- time_jump_or_recap
- other

FPS:
- 0.25
- 0.5
- 1

Resolution:
- 360
- 480
- 720

STRICT OUTPUT JSON SCHEMA (MUST FOLLOW EXACTLY)
{
  "planner_confidence": 0.0,
  "genre_distribution": { "<genre>": 0.0, "<genre>": 0.0 },
  "structure_mode": {
    "primary": "<structure_mode>",
    "secondary": ["<structure_mode>"]
  },
  "signal_priority": {
    "audio_text": 0.0,
    "visual": 0.0,
    "rationale": "<one sentence>"
  },
  "segmentation": {
    "target_segment_length_sec": [0, 0],
    "boundary_evidence_primary": ["<evidence_type>", "<evidence_type>"],
    "boundary_evidence_secondary": ["<evidence_type>"],
    "sampling": {
      "fps": 0.0,
      "max_resolution": 0,
      "use_subtitles": true
    }
  },
  "description": {
    "slots_weight": {
      "cast_speaker": 0.0,
      "setting": 0.0,
      "core_events": 0.0,
      "topic_claims": 0.0,
      "outcome_progress": 0.0,
      "notable_cues": 0.0
    },
    "sampling": {
      "fps": 0.0,
      "max_resolution": 0,
      "use_subtitles": true
    },
    "notes": "<one short paragraph: how to prioritize content in descriptions>"
  }
}

YOU WILL RECEIVE THE DATA IN THIS FORMAT
[GLOBAL_STATS]
... (duration, subtitle_density, etc.)

[PROBE_25%]
Frames: ...
Subtitles: ...

[PROBE_50%]
Frames: ...
Subtitles: ...

[PROBE_75%]
Frames: ...
Subtitles: ...

NOW produce JSON that strictly matches the schema. Output JSON ONLY.
"""
}

VIDEO_SEGMENT_PROMPT = {
"SYSTEM": r"""
You are a segmentation boundary detector and semantic labeler for video chunks.

Goal:
Given a segmentation specification and ONE contiguous chunk [T_start, T_end), detect valid semantic boundaries strictly inside the chunk AND generate a concise semantic title for the segment ending at each boundary.

Hard rules (must follow):
1) Validity:
   - A boundary is valid ONLY if at least one PRIMARY evidence type is clearly supported by the inputs.
2) Signal priority:
   - If audio_text priority is higher: favor boundaries that align with completed thoughts, even if visuals change.
   - If visual priority is higher: favor strong visual transitions.
3) Chunk edges are NOT boundaries:
   - Do NOT output T_start or T_end as boundaries.
4) Soft pacing:
   - target_segment_length_sec is a guideline only. Never force a cut without PRIMARY evidence.
5) Title Generation:
   - For each detected boundary, generate a `segment_title` describing the content of the segment immediately PRECEDING this boundary (i.e., from the previous boundary or T_start up to this timestamp).
   - Titles must be: 
     a) Concise (3-6 words ideally).
     b) Action/Topic oriented.
     c) Objective and descriptive (avoid subjective terms like "interesting part").
     d) Based on the dominant semantic event in that specific interval.
6) Output hygiene:
   - Output timestamps MUST be strictly within (T_start, T_end).
   - Sort boundaries by timestamp ascending.
   - Remove duplicates (timestamps within 0.5s are duplicates; keep the higher-confidence one).
   - Enforce a minimum spacing of 2.0s between boundaries (if closer, keep the stronger one).
   - If no valid boundary exists in (T_start, T_end), return [].

Output format:
Return ONLY a strict JSON array. Each item represents a boundary and the segment ending there:
{
  "timestamp": <number in seconds, the end of the segment>,
  "segment_title": "<concise semantic title for the segment ending at this timestamp>",
  "boundary_rationale": "<brief evidence-based reason for the cut, mention primary evidence>",
  "evidence": ["<evidence_type>", ...],
  "confidence": <0..1>
}
No extra text.
""".strip(),

"USER": r"""
Given the above frames and the following:

Chunk time range:
- T_start: {t_start}
- T_end: {t_end}

Segmentation specification:
- genre_distribution: {genre_str}
- structure_mode: {mode_str}
- signal_priority: audio_text={signal_audio_priority}, visual={signal_visual_priority}
- target_segment_length_sec: {target_segment_length_sec}
- primary_boundary_evidence: {boundary_evidence_primary}
- secondary_boundary_evidence: {boundary_evidence_secondary}

Chunk subtitles (each line lies strictly within this chunk):
{subtitles}

Now output the JSON list of boundaries and their corresponding segment titles within (T_start, T_end).
Remember: The `segment_title` should describe the content from the start of the segment (previous boundary or T_start) up to the `timestamp`.
""".strip()
}


CONTEXT_GENERATION_PROMPT = {
"SYSTEM": r"""
You are a segment captioning model (multimodal LLM).

Goal:
Given ONE video segment (frames/video + optional subtitles) AND a provided `segment_title`, produce:
1) a concise summary (for human or agent quick viewing).
2) a structured slot-based description (for downstream parsing, QA, retrieval),
3) a fluent final_caption paragraph (for human reading) synthesized from the slots.

How to use the inputs (follow strictly):

1) Integration of `segment_title` (CRITICAL):
   - The `segment_title` provides the high-level semantic theme or main activity of this segment.
   - USE IT AS A GUIDE: Align your `summary` and `final_caption` with the theme defined in the title.
   - DO NOT COPY: Never simply repeat the title. You must expand it with specific visual details (objects, actions, colors, spatial relations) and audio context found in the frames/subtitles.
   - VERIFY & CORRECT: If the visual/audio evidence clearly contradicts the `segment_title` (e.g., title says "cooking" but video shows "driving"), trust the sensory evidence (frames/audio) and describe what is actually happening, optionally noting the discrepancy in `notable_cues`.
   - EXPAND DETAILS: Use the title as a hypothesis to focus your attention. If the title is "Preparing ingredients", actively look for specific ingredients, tools, and preparation methods to describe.

2) Genre & Structure Priors:
   - genre_distribution + structure_mode: decide WHAT the segment is mainly about and HOW to organize the summary.
     * turn_taking_qa / podcast_interview: emphasize speakers, key questions, claims, stance, and conclusions.
     * lecture_slide_driven: emphasize topic structure, key points, definitions, and slide/on-screen text.
     * narrative_scene_based: emphasize plot-relevant events, characters, setting, and causal progression.
     * step_by_step_procedure / tutorial_howto: emphasize steps, actions, tools, and outcomes.

3) Signal Priority:
   - signal_priority: decide WHICH modality is authoritative when uncertain or conflicting.
     * audio_text higher: prefer subtitles/speech meaning; avoid over-interpreting visuals beyond what's supported.
     * visual higher: include salient visual details; use subtitles as support when available.

4) Slot Weighting:
   - slots_weight: allocate detail proportional to weights (higher weight => more detail). Use the same priorities when writing final_caption.

Hard rules:
- Output MUST be strict JSON only. No extra text.
- Fill ALL slots. If uncertain, write "unknown" or a brief uncertainty note rather than guessing.
- Do NOT narrate frame-by-frame. Summarize at the segment level.
- Prefer concrete, verifiable statements grounded in the provided inputs.
- summary must be concise (1 sentence), reflecting genre_distribution, structure_mode, AND aligning with the segment_title.
- final_caption must be coherent and self-contained (4–8 sentences), expanding on the segment_title with rich details based on slots_weight priorities.

Output JSON schema (exact keys only):
{
  "summary": "<1 sentence summary>",
  "slots": {
    "cast_speaker": "<text>",
    "setting": "<text>",
    "core_events": "<text>",
    "topic_claims": "<text>",
    "outcome_progress": "<text>",
    "notable_cues": "<text>"
  },
  "final_caption": "<4-8 sentence paragraph>",
  "confidence": <number between 0 and 1>
}
""".strip(),

"USER": r"""
Given the above frames and the following:

Segment Context:
- segment_title: "{segment_title}" 
  (Use this as the semantic anchor for the segment's main activity/topic)

Captioning priors:
- genre_distribution: {genre_str}
- structure_mode: {mode_str}
- signal_priority: audio_text={signal_audio_priority}, visual={signal_visual_priority}
- slots_weight: {slots_weight}
- notes: {notes}       

Segment subtitles (if provided; may be noisy/incomplete):
{subtitles}

Now generate the JSON output, ensuring the description expands upon the `segment_title` with concrete visual and audio details.
""".strip()
}


VIDEO_GLOBAL_PROMPT = {
"SYSTEM": """
You are an expert in video content analysis and summarization. Your task is to generate a concise, informative, and engaging **title** and a clear, coherent **abstract** for a video based solely on its structured segment descriptions.

- The title should capture the core theme or main event of the video in a natural and compelling way.
- The abstract should summarize the key points, narrative flow, or semantic content across all segments, avoiding redundancy and maintaining logical coherence.
- Do not invent details not supported by the segment descriptions.
- Use neutral, objective language appropriate for descriptive metadata.

**Output Format**
```json
{
  "title": "<string>",
  "abstract": "<string>"
}
```

Do not include any additional text or markdown formatting.
""",

"USER": """
Given the following video segments description:

**video segments description**
```
{segments_description}
```

Now, generate the video title and abstract.
"""
}

