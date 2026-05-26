from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any, Optional, List, Dict, Tuple
from dataclasses import asdict
import re
import os
import json
import json_repair
import concurrent.futures
import time
import queue
import threading
import random
import copy

from ..generators.base import BaseGenerator
from ..core.tree import BaseTree
from ..workspaces.base import BaseWorkspace
from ..prompts import VIDEO_SEGMENT_PROMPT, CONTEXT_GENERATION_PROMPT, VIDEO_GLOBAL_PROMPT, VIDEO_PROBE_PROMPT
from ..schemas import (
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
    CreateVideoGAMResult
)
# from ..utils import get_video_input_for_api
from ..utils import get_frame_indices, prepare_video_input, parse_srt, get_subtitle_in_segment, get_video_property, read_json
from ..agents.gam_agent import BaseGAMAgent


class VideoGAMAgent(BaseGAMAgent):
    """
    VideoGAMAgent - Video GAM Agent
    
    Handles video probing, segmentation, context generation, and organizing into GAM.
    
    Architecture:
    - tree: READ-ONLY view of the file system structure (VideoGAMTree)
    - workspace: Executes Linux commands (mkdir, ffmpeg, echo, etc.) to modify the file system
    - planner: LLM for video probing and global planning
    - segmentor: LLM for segment-level processing (segmentation + description)
    
    Public interface:
        add() - The single unified entry point for creating Video GAM.
    """

    def __init__(
        self,
        planner: BaseGenerator,
        segmentor: BaseGenerator,
        workspace: Optional[BaseWorkspace] = None,
        tree: BaseTree = None,
    ):
        # planner 作为主 generator 传递给 BaseGAMAgent
        super().__init__(generator=planner, tree=tree, workspace=workspace)
        self.planner = planner
        self.segmentor = segmentor
            
    def parse_response(self, generated_text: str) -> dict | list:
        """
        Parse the JSON response from the LLM.
        
        Handles various response formats including:
        - Plain JSON
        - Markdown code blocks (```json ... ```)
        - <think> tags (removes them)
        - Loose JSON structures (using json_repair)
        
        Args:
            generated_text: Raw text response from LLM
            
        Returns:
            Parsed JSON object (dict or list), or empty dict if parsing fails
        """
        if generated_text is None:
            return {}
        
        text = generated_text.strip()
        
        # 1. Remove <think> blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        
        # 2. Remove Markdown code block markers (```json ... ```)
        # Try matching list or dictionary
        pattern = r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```"
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            text = match.group(1)
        else:
            # Fallback: Try to find the outermost [] or {}
            first_brace = text.find('{')
            first_bracket = text.find('[')
            
            if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
                # Likely a list
                last_bracket = text.rfind(']')
                if last_bracket != -1:
                    text = text[first_bracket:last_bracket+1]
            elif first_brace != -1:
                # Likely an object
                last_brace = text.rfind('}')
                if last_brace != -1:
                    text = text[first_brace:last_brace+1]
            
        # 3. Try parsing
        try:
            context = json.loads(text)
        except Exception: 
            try:
                context = json_repair.loads(text)
            except Exception:
                context = {}
                
        return context
    
    def _process_segment_task(
        self, 
        video_path: str, 
        seg_start_time: float, 
        seg_end_time: float, 
        seg_title: str,
        seg_id: int, 
        subtitle_items: list,
        video_process_spec: VideoProcessSpec
    ) -> dict:
        """
        Process a single video segment: generate summary and details.
        
        Steps:
        1. Extract subtitles for the segment
        2. Prepare prompts (System + User)
        3. Call LLM to generate description
        4. Parse response
        5. Save results to markdown files
        6. Extract video clip using ffmpeg if needed
        
        Args:
            video_path: Path to the source video
            seg_start_time: Start time in seconds
            seg_end_time: End time in seconds
            seg_title: title generated for this seg
            seg_id: Segment identifier (integer)
            subtitle_items: List of subtitle items
            video_process_spec: Configuration for video processing
            
        Returns:
            Dictionary containing segment metadata and generated content
        """
        try:
            description_sampling = asdict(video_process_spec.description_sampling)
            caption_with_subtitles = description_sampling.get('use_subtitles', True)
            caption_spec = video_process_spec.caption_spec

            # # 1. Get frames (commented out in original code)
            # frame_indices = get_frame_indices(video_path, seg_start_time, seg_end_time, fps=description_sampling.get('fps', 1))
            # frame_base64_list, timestamps = prepare_video_input(video_path, frame_indices, description_sampling.get('max_resolution', 480), max_workers=8)
            
            # 2. Get subtitles
            if caption_with_subtitles:
                _, subtitles_str_in_seg = get_subtitle_in_segment(subtitle_items, seg_start_time, seg_end_time)
            else:
                subtitles_str_in_seg = ''
            
            # 3. Prepare prompt
            system_prompt = CONTEXT_GENERATION_PROMPT['SYSTEM']
            format_args = {**asdict(caption_spec), 'segment_title': seg_title}
            user_prompt = CONTEXT_GENERATION_PROMPT['USER'].format(subtitles=subtitles_str_in_seg, **format_args)

            # 4. Generate
            out = self._generate_single_w_video(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                video_path=video_path,
                start_time=seg_start_time,
                end_time=seg_end_time,
                video_sampling=description_sampling,
                generator=self.segmentor,
            )
            
            generated_text = out['text']
            response = out['response']
            # 5. Parse
            context = self.parse_response(generated_text)
            
            if 'summary' not in context or context['summary'] == '':
                context['summary'] = f'No summary for segment {seg_start_time} to {seg_end_time}'
            if 'final_caption' not in context or context['final_caption'] == '':
                context['detail'] = f'No detail description for segment {seg_start_time} to {seg_end_time}'
            else:
                # context['detail'] = context['final_caption'] + '\n' + '\n'.join([f'{k}: {v}' for k, v in context['slots'].items()])
                context['detail'] = context['final_caption']
                
            # 6. Save results
            video_seg = VideoSeg(
                seg_id=f'seg_{seg_id:04d}',
                seg_title=seg_title,
                summary=context.get('summary', ''),
                start_time=seg_start_time,
                end_time=seg_end_time,
                duration=seg_end_time - seg_start_time,
                detail=context.get('detail', ''),
            )
            context_md = video_seg.to_markdown(with_subtitles=caption_with_subtitles)
            
            seg_title_str = '_'.join(seg_title.split(' '))
            save_name = f'seg{seg_id:04d}-{seg_title_str}-{seg_start_time:.2f}-{seg_end_time:.2f}s'
            self.workspace.run(f'mkdir -p segments/{save_name}')
            self.workspace.run(f'echo "{context_md}" > segments/{save_name}/README.md')
                
            if subtitles_str_in_seg:
                self.workspace.run(f'echo "{subtitles_str_in_seg}" > segments/{save_name}/SUBTITLES.md')
            
            output, exit_code = self.workspace.run(f'test -e segments/{save_name}/video_clip.mp4')

            if not output and exit_code.startswith("Error"):
                # Ensure segments/seg_{seg_id:04d} directory exists
                # Call ffmpeg
                output, exit_code = self.workspace.run(f'ffmpeg -y -loglevel quiet -ss {seg_start_time} -to {seg_end_time} -i {os.path.basename(video_path)} -c copy segments/{save_name}/video_clip.mp4')

                if not exit_code.endswith('0'):
                    raise Exception(f'ffmpeg failed with exit code {exit_code}: {output}')
            
            return {'start_time': seg_start_time, 'end_time': seg_end_time, 'seg_title': seg_title, **context, 'token_usage': response['usage']['total_tokens']}
        except Exception as e:
            print(f"❌ Error processing segment {seg_id} ({seg_start_time}-{seg_end_time}): {e}")
            return {
                'start_time': seg_start_time,
                'end_time': seg_end_time,
                'seg_title': f'Error: {e}',
                'summary': f'Error: {e}',
                'detail': f'Error: {e}',
                'token_usage': 0,
            }


    
    def _prepare_messages_w_video(self, system_prompt, user_prompt, frame_base64_list, timestamps):
        """
        Prepare messages for LLM with video frames.
        
        Args:
            system_prompt: System instruction
            user_prompt: User instruction
            frame_base64_list: List of base64 encoded video frames
            timestamps: List of timestamps corresponding to frames
            
        Returns:
            List of message dictionaries for the LLM
        """
        
        user_content = []
        for frame_base64, timestamp in zip(frame_base64_list, timestamps):
            user_content.extend([
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_base64}",
                    }
                },
                {
                    "type": "text",
                    "text": f"<{timestamp:.1f} seconds>"
                },  
            ])
        user_content.append(
            {
                "type": "text",
                "text": user_prompt
            }
        )
        
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_content
            }
        ]

        return messages
    
    def _prepare_messages(self, system_prompt, user_prompt):
        """
        Prepare text-only messages for LLM.
        
        Args:
            system_prompt: System instruction
            user_prompt: User instruction
            
        Returns:
            List of message dictionaries for the LLM
        """
    
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        return messages
    
    def check_segmentation_info(self, segmentation_info, chunk_start_time=0, chunk_end_time=float('inf')):
        """
        Validate the segmentation info structure and logic.
        
        Args:
            segmentation_info: List of segment dictionaries
            chunk_start_time: Start time constraint
            chunk_end_time: End time constraint
            
        Raises:
            AssertionError: If validation fails
        """
        # check structed_output, if failed, continue
        assert type(segmentation_info) == list and len(segmentation_info) > 0, f'empty segmentation_info: {segmentation_info}'
        the_last_break_timepoint = chunk_start_time
        for item in segmentation_info:
            assert type(item) == dict, f'segment item is not dict: {item}'
            break_timepoint = item.get('timestamp', 0)
            assert break_timepoint >= chunk_start_time, f'break_timepoint {break_timepoint} is not in chunk [{chunk_start_time}, {chunk_end_time}]'
            assert break_timepoint < chunk_end_time, f'break_timepoint {break_timepoint} is not in chunk [{chunk_start_time}, {chunk_end_time}]'
            assert break_timepoint > the_last_break_timepoint, f'break_timepoint {break_timepoint} is not after the last timepoint {the_last_break_timepoint}'
            the_last_break_timepoint = break_timepoint
    
    def revise_segmentation_info(self, segmentation_info, chunk_start_time=0, chunk_end_time=float('inf')):
        """
        Revise segmentation info to fit the chunk time range.
        Filters out invalid timestamps that are out of bounds or not strictly increasing.
        
        Args:
            segmentation_info: List of segment dictionaries
            chunk_start_time: Start time constraint
            chunk_end_time: End time constraint
            
        Returns:
            Filtered list of segment dictionaries
        """
        revised_info = []
        the_last_break_timepoint = chunk_start_time
        for item in segmentation_info:
            break_timepoint = item.get('timestamp', 0)
            if break_timepoint >= chunk_end_time:
                continue
            if break_timepoint <= chunk_start_time:
                continue
            if break_timepoint <= the_last_break_timepoint:
                continue
            revised_info.append(item)
            the_last_break_timepoint = break_timepoint
        return revised_info
    
    def check_probe_result(self, probe_result):
        """
        Check if the probe result has all required fields.
        
        Args:
            probe_result: Dictionary containing probe results
            
        Raises:
            AssertionError: If validation fails
        """
        assert type(probe_result) == dict, f'probe_result is not dict: {probe_result}'
        assert 'genre_distribution' in probe_result and type(probe_result['genre_distribution']) == dict, f'correct genre_distribution not in probe_result: {probe_result}'
        assert 'structure_mode' in probe_result, f'structure_mode not in probe_result: {probe_result}'
        assert 'signal_priority' in probe_result, f'signal_priority not in probe_result: {probe_result}'
        assert 'segmentation' in probe_result, f'segmentation not in probe_result: {probe_result}'
        assert 'description' in probe_result, f'description not in probe_result: {probe_result}'
    
    def _clamp(self, x: float, lo: float, hi: float) -> float:
        """Clamp value x between lo and hi."""
        try:
            x = float(x)
        except Exception:
            return lo
        return max(lo, min(hi, x))


    def _as_bool(self, x: Any, default: bool = True) -> bool:
        """Convert x to boolean with flexible string interpretation."""
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            return bool(x)
        if isinstance(x, str):
            v = x.strip().lower()
            if v in {"true", "1", "yes", "y"}:
                return True
            if v in {"false", "0", "no", "n"}:
                return False
        return default


    def _normalize_dist(self, d: Dict[str, Any], allowed: set[str], fallback_key: str = "other", topk: int = 4) -> Dict[str, float]:
        """Keep allowed keys, drop invalid, keep topk by weight, then renormalize. If empty, fallback."""
        if not isinstance(d, dict):
            return {fallback_key: 1.0}

        items: List[Tuple[str, float]] = []
        for k, v in d.items():
            if k in allowed:
                w = float(v) if isinstance(v, (int, float)) else 0.0
                if w > 0:
                    items.append((k, w))

        if not items:
            return {fallback_key: 1.0}

        items.sort(key=lambda kv: kv[1], reverse=True)
        items = items[: max(1, min(topk, len(items)))]

        s = sum(w for _, w in items)
        if s <= 0:
            return {fallback_key: 1.0}

        return {k: w / s for k, w in items}


    def _normalize_weights(self, d: Dict[str, Any], keys: List[str]) -> Dict[str, float]:
        """Ensure all keys exist; clamp >=0; renormalize; fallback to uniform if all zero."""
        out = {}
        total = 0.0
        for k in keys:
            v = d.get(k, 0.0) if isinstance(d, dict) else 0.0
            w = float(v) if isinstance(v, (int, float)) else 0.0
            w = max(0.0, w)
            out[k] = w
            total += w

        if total <= 0:
            # uniform
            u = 1.0 / len(keys)
            return {k: u for k in keys}

        return {k: out[k] / total for k in keys}


    def _merge_defaults(self, user_strategy: Dict[str, Any], default_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge with defaults: missing fields filled from default."""
        merged = copy.deepcopy(default_strategy)

        def rec(dst: Dict[str, Any], src: Dict[str, Any]):
            for k, v in (src or {}).items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    rec(dst[k], v)
                else:
                    dst[k] = v

        if isinstance(user_strategy, dict):
            rec(merged, user_strategy)
        return merged
        
    def _build_spec_from_strategy_pkg(self, strategy_pkg: Dict[str, Any]) -> VideoProcessSpec:
        """
        Input: planner-produced segmentation/captioning specification (possibly incomplete/noisy)
        Output:
        - segment_request: instruction text for segmentation MLLM
        - caption_request: instruction text for caption/description MLLM
        - segmentation_sampling: fps/res/use_subtitles for segmentation
        - description_sampling: fps/res/use_subtitles for description
        - normalized_strategy: cleaned + normalized strategy dict
        """
        # 1) Merge with defaults
        s = self._merge_defaults(strategy_pkg, DEFAULT_STRATEGY_PACKAGE)

        # 2) Normalize key fields
        s["planner_confidence"] = self._clamp(s.get("planner_confidence", 0.25), 0.0, 1.0)

        s["genre_distribution"] = self._normalize_dist(
            s.get("genre_distribution", {}),
            allowed=ALLOWED_GENRES,
            fallback_key="other",
            topk=4,
        )

        # structure_mode
        sm = s.get("structure_mode", {})
        primary = sm.get("primary", "other")
        if primary not in ALLOWED_STRUCTURE_MODES:
            primary = "other"
        secondary = sm.get("secondary", [])
        if not isinstance(secondary, list):
            secondary = []
        secondary = [m for m in secondary if m in ALLOWED_STRUCTURE_MODES and m != primary][:2]
        s["structure_mode"] = {"primary": primary, "secondary": secondary}

        # signal_priority
        sp = s.get("signal_priority", {})
        audio_w = self._clamp(sp.get("audio_text", 0.6), 0.0, 1.0)
        visual_w = self._clamp(sp.get("visual", 0.4), 0.0, 1.0)
        tot = audio_w + visual_w
        if tot <= 0:
            audio_w, visual_w = 0.6, 0.4
            tot = 1.0
        audio_w /= tot
        visual_w /= tot
        rationale = sp.get("rationale", "")
        if not isinstance(rationale, str) or not rationale.strip():
            rationale = "Use a conservative hybrid strategy based on available evidence."
        s["signal_priority"] = {"audio_text": audio_w, "visual": visual_w, "rationale": rationale.strip()}

        # segmentation
        seg = s.get("segmentation", {})
        gran = seg.get("granularity", "hybrid")
        if gran not in ALLOWED_GRANULARITY:
            gran = "hybrid"
        tlen = seg.get("target_segment_length_sec", [90, 480])
        if not (isinstance(tlen, list) and len(tlen) == 2):
            tlen = [90, 480]
        t0 = int(max(10, min(3600, int(tlen[0]))))
        t1 = int(max(t0 + 10, min(7200, int(tlen[1]))))
        primary_evd = seg.get("boundary_evidence_primary", [])
        secondary_evd = seg.get("boundary_evidence_secondary", [])
        if not isinstance(primary_evd, list):
            primary_evd = []
        if not isinstance(secondary_evd, list):
            secondary_evd = []
        primary_evd = [e for e in primary_evd if e in ALLOWED_EVIDENCE][:3]
        secondary_evd = [e for e in secondary_evd if e in ALLOWED_EVIDENCE and e not in primary_evd][:3]
        if len(primary_evd) == 0:
            primary_evd = ["topic_shift_in_subtitles", "scene_location_change"]

        seg_sampling = seg.get("sampling", {})
        seg_fps = self._clamp(seg_sampling.get("fps", 0.5), 0.05, 4.0)
        seg_res = int(max(128, min(1536, int(seg_sampling.get("max_resolution", 384)))))
        seg_use_sub = self._as_bool(seg_sampling.get("use_subtitles", True), default=True)

        seg_notes = seg.get("notes", "")
        if not isinstance(seg_notes, str) or not seg_notes.strip():
            seg_notes = "Prefer self-contained segments; avoid over-segmentation on weak boundaries."
        s["segmentation"] = {
            "granularity": gran,
            "target_segment_length_sec": [t0, t1],
            "boundary_evidence_primary": primary_evd,
            "boundary_evidence_secondary": secondary_evd,
            "sampling": {"fps": seg_fps, "max_resolution": seg_res, "use_subtitles": seg_use_sub},
            "notes": seg_notes.strip(),
        }

        # description
        desc = s.get("description", {})
        slot_w = self._normalize_weights(desc.get("slots_weight", {}), DESCRIPTION_SLOTS)
        desc_sampling = desc.get("sampling", {})
        desc_fps = self._clamp(desc_sampling.get("fps", 0.2), 0.05, 4.0)
        desc_res = int(max(128, min(1536, int(desc_sampling.get("max_resolution", 384)))))
        desc_use_sub = self._as_bool(desc_sampling.get("use_subtitles", True), default=True)

        desc_notes = desc.get("notes", "")
        if not isinstance(desc_notes, str) or not desc_notes.strip():
            desc_notes = "Fill all slots; focus on high-level segment summary rather than frame-by-frame narration."
        s["description"] = {
            "slots_weight": slot_w,
            "sampling": {"fps": desc_fps, "max_resolution": desc_res, "use_subtitles": desc_use_sub},
            "notes": desc_notes.strip(),
        }
        
        # 3) Prepare spec
        genre_top = sorted(s["genre_distribution"].items(), key=lambda kv: kv[1], reverse=True)
        genre_str = ", ".join([f"{k}:{v:.2f}" for k, v in genre_top])
        modes = [s["structure_mode"]["primary"]] + s["structure_mode"]["secondary"]
        mode_str = ", ".join(modes)

        seg_cfg = s["segmentation"]
        seg_sampling_cfg = seg_cfg["sampling"]
        
        segment_spec = SegmentSpec(
            genre_str=genre_str,
            mode_str=mode_str,
            signal_audio_priority=f'{s["signal_priority"]["audio_text"]:.2f}',
            signal_visual_priority=f'{s["signal_priority"]["visual"]:.2f}',
            target_segment_length_sec=f'[{seg_cfg["target_segment_length_sec"][0]}, {seg_cfg["target_segment_length_sec"][1]}]',
            boundary_evidence_primary=", ".join(seg_cfg["boundary_evidence_primary"]),
            boundary_evidence_secondary=", ".join(seg_cfg["boundary_evidence_secondary"]),
        )
        
        desc_cfg = s["description"]
        weights = desc_cfg["slots_weight"]
        weight_str = ", ".join([f"{k}:{weights[k]:.2f}" for k in DESCRIPTION_SLOTS])
        caption_spec = CaptionSpec(
            genre_str=genre_str,
            mode_str=mode_str,
            signal_audio_priority=f'{s["signal_priority"]["audio_text"]:.2f}',
            signal_visual_priority=f'{s["signal_priority"]["visual"]:.2f}',
            slots_weight=weight_str,
            notes=desc_cfg["notes"],
        )

        # 4) Return sampling configs (you asked explicitly for segmentation sampling)
        segmentation_sampling = SamplingConfig(
            fps=float(seg_sampling_cfg["fps"]),
            max_resolution=int(seg_sampling_cfg["max_resolution"]),
            use_subtitles=bool(seg_sampling_cfg["use_subtitles"]),
        )
        description_sampling = SamplingConfig(
            fps=float(desc_cfg["sampling"]["fps"]),
            max_resolution=int(desc_cfg["sampling"]["max_resolution"]),
            use_subtitles=bool(desc_cfg["sampling"]["use_subtitles"]),
        )

        return VideoProcessSpec(
            segment_spec=segment_spec,
            caption_spec=caption_spec,
            segmentation_sampling=segmentation_sampling,
            description_sampling=description_sampling,
            normalized_strategy=s,
        )
    
    def _probe_video_content(
        self, 
        video_path: str, 
        duration_int: int, 
        subtitle_items: list = None, 
        video_sampling: dict = {}
    ) -> dict:
        """
        Probe video content by sampling segments and merging results.
        
        Args:
            video_path: Path to the video file
            duration_int: Total duration of video in seconds
            subtitle_items: List of parsed subtitle items
            video_sampling: Sampling configuration for the probe
            
        Returns:
            VideoProcessSpec object containing the strategy
        """

        # Define time segments to process: take 30s clips at 25%, 50%, 75% marks
        segments = []
        clip_duration = 30
        
        if duration_int <= clip_duration:
            segments.append((0, 0, duration_int))   # percentile, start, end
        else:
            for ratio in [0.25, 0.50, 0.75]:
                start = int(duration_int * ratio)
                # Ensure segment doesn't exceed video duration
                if start + clip_duration > duration_int:
                    start = duration_int - clip_duration
                segments.append((int(ratio*100), start, start + clip_duration))
            
            # Deduplicate and sort
            segments = sorted(list(set(segments)))
        
        def _process_segment(segment):
            ratio, start, end = segment
            
            _, subtitles_str_in_seg = get_subtitle_in_segment(subtitle_items, start, end)

            fps = video_sampling.get('fps', 1)
            max_resolution = video_sampling.get('max_resolution', 480)
            frame_indices = get_frame_indices(video_path, start, end, fps=fps)
            frame_base64_list, timestamps = prepare_video_input(video_path, frame_indices, max_resolution, max_workers=4)
        
            return ratio, frame_base64_list, timestamps, subtitles_str_in_seg

        # Parallel processing
        prepared_inputs = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(segments), 3)) as executor:
            future_to_segment = {executor.submit(_process_segment, seg): seg for seg in segments}
            for future in concurrent.futures.as_completed(future_to_segment):
                try:
                    res = future.result()
                    if res:
                        prepared_inputs.append(res)
                except Exception as e:
                    print(f"⚠️ Warning: Probe failed for segment {future_to_segment[future]}: {e}")
        
        prepared_inputs = sorted(prepared_inputs, key=lambda x: x[0])
        
        system_prompt = VIDEO_PROBE_PROMPT['SYSTEM']
        user_prompt = VIDEO_PROBE_PROMPT['USER']
        
        all_subtitle_str = ' '.join([item['text'] for item in subtitle_items])
        subtitle_word_count = len(all_subtitle_str.split())
        global_stats = f"- Duration: {duration_int} seconds.\n" # 视频总时长
        global_stats += f"- Subtitle word count: {subtitle_word_count}.\n" # 字幕总字数
        
        user_content = [
            {
                'type': 'text',
                'text': user_prompt
            },
            {
            'type': 'text',
            'text': '[GLOBAL_STATS]'
            },
            {
            'type': 'text',
            'text': global_stats
            },
        ]
        
        for ratio, frame_base64_list, timestamps, subtitles_str_in_seg in prepared_inputs:
            user_content.extend([
                {
                'type': 'text',
                'text': f'[PROBE_{ratio}%]'
                },
                {
                'type': 'text',
                'text': 'Frames:\n'
                }
            ])
            for frame_base64, timestamp in zip(frame_base64_list, timestamps):
                user_content.extend([
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_base64}",
                        }
                    },
                    {
                        "type": "text",
                        "text": f"<{timestamp:.1f} seconds>"
                    },  
                ])
            user_content.extend([
                {
                'type': 'text',
                'text': f'Subtitles: \n{subtitles_str_in_seg}'
                },
            ])
        
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        
        out = self._generate_single_w_video(messages=messages, generator=self.planner)
        
        strategy_pkg = self.parse_response(out['text'])
        video_process_spec = self._build_spec_from_strategy_pkg(strategy_pkg)
        
        return video_process_spec

    def _generate_segments_and_context(self, video_path: str, duration_int: int, subtitle_items: list, verbose: bool = False, chunk_size: int = 600, video_process_spec: VideoProcessSpec = None):
        """
        Generate video segments and context by iterating through the video in chunks.
        
        Args:
            video_path: Path to video
            duration_int: Duration in seconds
            subtitle_items: Subtitles
            verbose: Enable verbose logging
            chunk_size: Processing chunk size in seconds
            video_process_spec: Strategy specification
            
        Returns:
            List of all segment contexts
        """
        
        segment_spec = video_process_spec.segment_spec
        segmentation_sampling = asdict(video_process_spec.segmentation_sampling)

        all_contexts = []
        futures = []
        
        chunk_start_time = 0
        total_segments_count = 0
        c_idx = 0
        
        # Use a global ThreadPoolExecutor for segments processing
        # This avoids creating a new executor for each chunk and allows better load balancing
        max_workers = 12
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            try:
                while True:
                    chunk_end_time = min(chunk_start_time + chunk_size, duration_int)
                    st = time.time()
                    
                    # 1. Generate Segmentation for this Chunk (Producer)
                    try:
                        _, subtitles_str_in_seg = get_subtitle_in_segment(subtitle_items, chunk_start_time, chunk_end_time)
                        system_prompt = VIDEO_SEGMENT_PROMPT['SYSTEM']
                        user_prompt = VIDEO_SEGMENT_PROMPT['USER'].format(t_start=chunk_start_time, t_end=chunk_end_time, subtitles=subtitles_str_in_seg, **asdict(segment_spec))
                        
                        out = self._generate_single_w_video(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            video_path=video_path,
                            start_time=chunk_start_time,
                            end_time=chunk_end_time,
                            video_sampling=segmentation_sampling,
                            generator=self.segmentor,
                        )
                        
                        generated_text = out['text']
                        response = out['response']
                        segmentation_info = self.parse_response(generated_text)
                        segmentation_info = self.revise_segmentation_info(segmentation_info, chunk_start_time=chunk_start_time, chunk_end_time=chunk_end_time)
                        
                    except (AssertionError, Exception) as e:
                        print(f"❌ [Chunk {c_idx:03d}] Failed to generate valid segmentation info: {e}")
                        # If segmentation fails, we might want to skip or stop. 
                        # For now, let's stop to avoid infinite loops or bad state
                        break
                    
                    if verbose:
                        print(f"[Chunk {c_idx:03d}] Segmentation generated in {time.time() - st:.2f}s | Segments found: {len(segmentation_info)} | Token usage: {response['usage']['total_tokens']}")

                    if chunk_end_time >= duration_int:
                        segmentation_info.append({
                            'timestamp': chunk_end_time,
                            'segment_title': 'ending'  # TODO: require additional process logic for the last segment
                        })
                    # print(f'chunk {c_idx} finished. {time.time() - st:.2f}s')
                    # 2. Submit Segment Tasks to Executor (Consumer Pool)
                    seg_start_time = chunk_start_time
                    for i, item in enumerate(segmentation_info):
                        seg_end_time = item['timestamp']
                        seg_title = item['segment_title']
                        seg_id = total_segments_count + 1 + i
                        
                        future = executor.submit(
                            self._process_segment_task,
                            video_path=video_path,
                            seg_start_time=seg_start_time,
                            seg_end_time=seg_end_time,
                            seg_title=seg_title,
                            seg_id=seg_id,
                            subtitle_items=subtitle_items,
                            video_process_spec=video_process_spec
                        )
                        futures.append(future)
                        seg_start_time = seg_end_time

                    total_segments_count += len(segmentation_info)
                    chunk_start_time = segmentation_info[-1]['timestamp']
                    c_idx += 1
                    # print(f'Async {c_idx} chunk submitted. {time.time() - st:.2f}s')
                    if chunk_end_time >= duration_int:
                        break
            
            finally:
                pass
                
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    all_contexts.append(res)
                except Exception as e:
                    print(f"❌ Segment processing failed: {e}")

        # Sort by start time to ensure order
        all_contexts.sort(key=lambda x: x['start_time'])
            
        return all_contexts
    
    def _generate_global_context(self, all_contexts, duration_int, verbose, caption_with_subtitles: bool = True):
        """
        Generate global context (title, abstract) for the entire video.
        
        Args:
            all_contexts: List of segment contexts
            duration_int: Total video duration
            verbose: Enable verbose logging
        """

        st = time.time()
        system_prompt = VIDEO_GLOBAL_PROMPT['SYSTEM']
        segments_description = '\n'.join([
            f'- segments{seg_id+1:04d}: {seg["start_time"]:.1f} - {seg["end_time"]:.1f} seconds\nTitle: {seg["seg_title"]}\nDetail Description: {seg["detail"]}\n'
            for seg_id, seg in enumerate(all_contexts)
        ])
        user_prompt = VIDEO_GLOBAL_PROMPT['USER'].format(segments_description=segments_description)
        messages = self._prepare_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        out = self.planner.generate_single(
            messages=messages
        )   
        
        generated_text = out['text']
        response = out['response']
        global_context = self.parse_response(generated_text)
        
        if verbose:
            print(f"[Global] Context generation completed in {time.time() - st:.2f}s | Token usage: {response['usage']['total_tokens']}")
        
        segments_quickview = '\n'.join([
            f'- segments{seg_id+1:04d}: {seg["start_time"]:.1f} - {seg["end_time"]:.1f} seconds: {seg["summary"]}'
            for seg_id, seg in enumerate(all_contexts)
        ])
    
        video_global = VideoGlobal(
            title=global_context.get('title', ''),
            abstract=global_context.get('abstract', ''),
            duration=duration_int,
            num_segments=len(all_contexts),
            segments_quickview=segments_quickview,
        )
        with open(os.path.join(self.workspace.root_path, 'README.md'), 'w') as f:
            f.write(video_global.to_markdown(with_subtitles=caption_with_subtitles))
        
        
    def _generate_single_w_video(
        self, 
        system_prompt: str = '', 
        user_prompt: str = '', 
        video_path: str = '', 
        start_time: float = 0.0, 
        end_time: float = 0.0, 
        video_sampling: dict = {}, 
        messages: list[dict] = [], 
        generator = None
    ):
        """
        Generate a single LLM response, optionally with video context.
        
        Args:
            system_prompt: System instruction
            user_prompt: User instruction
            video_path: Path to video
            start_time: Start time for video clip
            end_time: End time for video clip
            video_sampling: Sampling config (fps, resolution)
            messages: Pre-constructed messages (optional)
            generator: LLM generator instance (defaults to self.planner)
            
        Returns:
            Generator output dictionary
        """
        
        if not generator:
            generator = self.planner
        
        if not messages:
            fps = video_sampling.get('fps', 1)
            max_resolution = video_sampling.get('max_resolution', 480)
            frame_indices = get_frame_indices(video_path, start_time, end_time, fps=fps)
            frame_base64_list, timestamps = prepare_video_input(video_path, frame_indices, max_resolution, max_workers=4)
            
            messages = self._prepare_messages_w_video(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                frame_base64_list=frame_base64_list,
                timestamps=timestamps,
            )
        
        out = generator.generate_single(messages=messages)
        return out
  
  
    def _create(
        self,
        verbose: bool = False,
        caption_with_subtitles: bool = True,
    ) -> CreateVideoGAMResult:
        """
        Create GAM for Video.
        
        Orchestrates the entire video processing pipeline:
        1. Initialize workspace
        2. Probe video content
        3. Generate segments and contexts
        4. Generate global context
        5. Organize workspace
        
        Args:
            verbose: Enable verbose logging
            caption_with_subtitles: Use subtitles in captioning
            
        Returns:
            Result object containing statistics and success status
        """
        # Initilize workspace: Prepare information for segmentor
        workspace_dir = self.workspace.root_path # workspace_dir: 是一个字符串
        # 获取这个目录下的 mp4 文件
        video_path = str(list(Path(workspace_dir).glob('*.mp4'))[0])
        srt_files = list(Path(workspace_dir).glob('*.srt'))
        srt_path = str(srt_files[0]) if srt_files else ''

        # Prepare text content related to video
        ## Subtitles
        subtitle_items, subtitles_str = parse_srt(srt_path)
        if caption_with_subtitles:
            with open(os.path.join(workspace_dir, 'SUBTITLES.md'), 'w') as f:
                f.write(subtitles_str)

        ## Video property
        video_info = get_video_property(video_path)
        duration_int = int(video_info['duration'])
        resolution = video_info['resolution']
        
        # Probe Video: First complete basic information judgment of the video
        video_sampling = {'fps': 1, 'max_resolution': 480}
        st = time.time()
        video_process_spec = self._probe_video_content(video_path, duration_int, subtitle_items, video_sampling)
        
        probe_result = video_process_spec.normalized_strategy
        
        # Override use_subtitles in caption_spec (mapped to description_sampling here)
        video_process_spec.description_sampling.use_subtitles = caption_with_subtitles

        if verbose:
            print(f"[Probe] Video analysis completed in {time.time() - st:.2f}s")
            print(f"[Probe] Strategy determined:\n{json.dumps(probe_result, indent=2)}")
            
        save_path = os.path.join(workspace_dir, 'PROBE_RESULT.json')
        with open(save_path, 'w') as f:
            json.dump(probe_result, f, indent=4)
        
        
        # Segmentation & Context Generation: Generate video segments and contexts based on previous results
        all_contexts = self._generate_segments_and_context(
            video_path=video_path,
            duration_int=duration_int,
            subtitle_items=subtitle_items,
            verbose=verbose,
            video_process_spec=video_process_spec
        )
        
        # TODO adjust context: Continue to divide or merge
        
        # Global Context Generation: Finally generate global context of video based on all segment contexts
        self._generate_global_context(all_contexts, duration_int, verbose, caption_with_subtitles)
        
        # Final Check: Check if GAM directory format meets requirements
        ## Check video workspace
        if not self.tree.check_video_workspace(self.workspace):
            raise Exception(f'workspace {self.workspace} is not a valid video workspace')
        ## Organize video workspace, move some files to .agentignore directory
        self.tree.organize_video_workspace(self.workspace)
        
        # Construct result
        result = CreateVideoGAMResult(
            success=True,
            segment_num=len(all_contexts),
            segmentation_sampling=video_process_spec.segmentation_sampling,
            description_sampling=video_process_spec.description_sampling,
            segment_spec=video_process_spec.segment_spec,
            caption_spec=video_process_spec.caption_spec,
        )
        
        # End
        if verbose:
            print(f"✅ Video GAM construction completed successfully!")
            
        return result

    def add(
        self,
        input_path: str | Path | None = None,
        video_path: str | Path | None = None,
        subtitle_path: str | Path | None = None,
        verbose: bool = False,
        caption_with_subtitles: bool = True,
    ) -> CreateVideoGAMResult:
        """
        Video GAM only support full creation now.
        
        Args:
            input_path: Input directory containing video.mp4, subtitles.srt (optional).
            verbose: Whether to print verbose output.
            caption_with_subtitles: Whether to include subtitles in the caption generation.
        """
        assert input_path or video_path, "Either input_path or video_path must be provided."
        
        if input_path:
            input_path = Path(input_path)
            if not input_path.exists():
                raise FileNotFoundError(f"Input path does not exist: {input_path}")
            
            if verbose:
                print(f"📂 Processing input video from: {input_path}")
                
            # Check for required files
            mp4_files = list(input_path.glob('*.mp4'))
            srt_files = list(input_path.glob('*.srt'))

            if len(mp4_files) != 1:
                raise ValueError(f"Expected exactly one .mp4 file in {input_path}, found {len(mp4_files)}")
            
            if len(srt_files) > 1 and verbose:
                print(f"⚠️ Warning: Multiple .srt files found in {input_path}. Using the first one.")
            
            # Copy files to workspace
            # Ensure workspace root is a Path object for joining, but convert to str for copy_to_workspace
            workspace_root = Path(self.workspace.root_path)
            
            self.workspace.copy_to_workspace(str(mp4_files[0]), str(workspace_root / mp4_files[0].name))
            
            if srt_files:
                self.workspace.copy_to_workspace(str(srt_files[0]), str(workspace_root / srt_files[0].name))
            
            if verbose:
                print(f"✅ Files copied to workspace: {workspace_root}")
        elif video_path:
            video_path = Path(video_path)
            if not video_path.exists():
                raise FileNotFoundError(f"Video path does not exist: {video_path}")
            
            if verbose:
                print(f"📂 Processing video from: {video_path}")
            
            # Copy video to workspace
            workspace_root = Path(self.workspace.root_path)
            
            self.workspace.copy_to_workspace(str(video_path), str(workspace_root / video_path.name))
            
            if subtitle_path:
                subtitle_path = Path(subtitle_path)
                if not subtitle_path.exists():
                    raise FileNotFoundError(f"Subtitle path does not exist: {subtitle_path}")
                
                if verbose:
                    print(f"📂 Processing subtitle from: {subtitle_path}")
                
                # Copy subtitle to workspace
                self.workspace.copy_to_workspace(str(subtitle_path), str(workspace_root / subtitle_path.name))
                
            if verbose:
                print(f"✅ Files copied to workspace: {workspace_root}")

        return self._create(
            verbose=verbose,
            caption_with_subtitles=caption_with_subtitles,
        )
