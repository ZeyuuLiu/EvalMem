# -*- coding: utf-8 -*-
"""
Custom API routes for GAM – supports flexible parameters for add and request.
"""

from __future__ import annotations

import os
import json
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from flask import Blueprint, current_app, jsonify, request, Response

from ...core.tree import GAMTree, VideoGAMTree
from ...workspaces.local_workspace import LocalWorkspace
from ...generators.openai_generator import OpenAIGenerator
from ...generators.config import OpenAIGeneratorConfig
from ..helpers import get_timestamp_dir, next_chunks_dir, task_results

custom_api_bp = Blueprint("custom_api", __name__)

@dataclass
class MockArgs:
    """Mock object to mimic argparse.Namespace for helper functions."""
    type: str
    gam_dir: str
    model: str
    api_base: str
    api_key: str
    max_tokens: int
    temperature: float
    verbose: bool = True
    # Add other fields as needed

def _get_param(data: Dict, key: str, default: Any = None) -> Any:
    """Helper to get parameter from JSON or form data."""
    return data.get(key, default)

def _build_custom_generator(data: Dict, prefix: str = "") -> OpenAIGenerator:
    """Build a generator from custom parameters in request data."""
    # Use app config as base defaults
    default_gen = current_app.config.get("GENERATOR")
    
    model = _get_param(data, f"{prefix}model") or (default_gen.config.model_name if default_gen else "gpt-4o-mini")
    api_base = _get_param(data, f"{prefix}api_base") or (default_gen.config.base_url if default_gen else "https://api.openai.com/v1")
    api_key = _get_param(data, f"{prefix}api_key") or (default_gen.config.api_key if default_gen else os.environ.get("OPENAI_API_KEY", ""))
    max_tokens = int(_get_param(data, f"{prefix}max_tokens") or (default_gen.config.max_tokens if default_gen else 4096))
    temperature = float(_get_param(data, f"{prefix}temperature") or (default_gen.config.temperature if default_gen else 0.3))

    config = OpenAIGeneratorConfig(
        model_name=model,
        base_url=api_base,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return OpenAIGenerator(config)

@custom_api_bp.route("/api/custom/add", methods=["POST"])
def api_custom_add():
    """
    Custom API for adding content to GAM.
    Supports all CLI parameters via JSON body.
    """
    try:
        data = request.get_json() or request.form.to_dict()
        gam_type = _get_param(data, "type", "text")
        gam_dir_str = _get_param(data, "gam_dir")
        
        if not gam_dir_str:
            # If gam_dir is not provided, create a new one in OUTPUT_BASE
            timestamp = get_timestamp_dir()
            gam_dir = Path(current_app.config["OUTPUT_BASE"]) / timestamp / "gam"
        else:
            gam_dir = Path(gam_dir_str)

        gam_dir.mkdir(parents=True, exist_ok=True)
        workspace = LocalWorkspace(root_path=str(gam_dir))

        generator = _build_custom_generator(data)
        verbose = _get_param(data, "verbose", True)

        if gam_type == "text":
            from ...agents.text_gam_agent import TextGAMAgent
            
            try:
                tree = GAMTree.from_disk(gam_dir, workspace)
            except Exception:
                tree = GAMTree.create(gam_dir, name=gam_dir.name)

            use_chunking = _get_param(data, "use_chunking", True)
            if isinstance(use_chunking, str):
                use_chunking = use_chunking.lower() == "true"
                
            memory_workers = int(_get_param(data, "memory_workers", 4))
            window_size = int(_get_param(data, "window_size", 30000))
            overlap_size = int(_get_param(data, "overlap_size", 10000))
            
            agent = TextGAMAgent(
                generator=generator,
                tree=tree,
                workspace=workspace,
                use_chunking=use_chunking,
                window_size=window_size,
                overlap_size=overlap_size,
                auto_save=True,
                verbose=verbose,
                memory_workers=memory_workers,
            )

            inputs = _get_param(data, "input")
            if inputs:
                if isinstance(inputs, str):
                    inputs = [Path(inputs)]
                else:
                    inputs = [Path(f) for f in inputs]
            
            contents = _get_param(data, "content")
            if contents and isinstance(contents, str):
                contents = [contents]

            context = _get_param(data, "context", "")
            memorize_instruction = _get_param(data, "memorize_instruction")
            max_splits = int(_get_param(data, "max_splits", 120))
            taxonomy_batch_size = int(_get_param(data, "taxonomy_batch_size", 50))
            force_reorganize = _get_param(data, "force_reorganize", False)
            if isinstance(force_reorganize, str):
                force_reorganize = force_reorganize.lower() == "true"

            # Prepare output_dir for chunks if needed
            output_dir_str = _get_param(data, "output_dir")
            if output_dir_str:
                output_dir = Path(output_dir_str)
            else:
                output_dir = gam_dir.parent / f"chunks_{get_timestamp_dir()}"
            output_dir.mkdir(parents=True, exist_ok=True)

            result = agent.add(
                input_file=inputs,
                content=contents,
                context=context,
                memorize_instruction=memorize_instruction,
                max_splits=max_splits,
                taxonomy_batch_size=taxonomy_batch_size,
                output_dir=output_dir,
                force_reorganize=force_reorganize,
            )

            return jsonify({
                "success": True,
                "type": "text",
                "gam_dir": str(gam_dir),
                "output_dir": str(output_dir),
                "created_files": len(getattr(result, "created_files", [])),
                "new_directories": len(getattr(result, "new_directories", [])),
            })

        elif gam_type == "video":
            from ...agents.video_gam_agent import VideoGAMAgent
            
            try:
                tree = VideoGAMTree.from_disk(gam_dir, workspace)
            except Exception:
                tree = VideoGAMTree.create_empty(gam_dir, name=gam_dir.name)

            # Segmentor can be different
            if _get_param(data, "segmentor_model"):
                segmentor = _build_custom_generator(data, prefix="segmentor_")
            else:
                segmentor = generator

            agent = VideoGAMAgent(
                planner=generator,
                segmentor=segmentor,
                workspace=workspace,
                tree=tree,
            )

            inputs = _get_param(data, "input")
            if not inputs:
                return jsonify({"success": False, "error": "video type requires input (video directory path)"})
            
            if isinstance(inputs, list):
                input_path = Path(inputs[0]).resolve()
            elif inputs:
                input_path = Path(inputs).resolve()
            else:
                input_path = None

            video_path = _get_param(data, "video_path")
            if video_path:
                video_path = Path(video_path).resolve()
            
            subtitle_path = _get_param(data, "subtitle_path")
            if subtitle_path:
                subtitle_path = Path(subtitle_path).resolve()

            caption_with_subtitles = _get_param(data, "caption_with_subtitles", True)
            if isinstance(caption_with_subtitles, str):
                caption_with_subtitles = caption_with_subtitles.lower() == "true"

            result = agent.add(
                input_path=input_path,
                video_path=video_path,
                subtitle_path=subtitle_path,
                verbose=verbose,
                caption_with_subtitles=caption_with_subtitles,
            )

            return jsonify({
                "success": True,
                "type": "video",
                "gam_dir": str(gam_dir),
                "segment_num": getattr(result, "segment_num", 0),
            })

        else:
            return jsonify({"success": False, "error": f"Unsupported gam type: {gam_type}"})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})

@custom_api_bp.route("/api/custom/request", methods=["POST"])
def api_custom_request():
    """
    Custom API for requesting information from GAM.
    Supports all CLI parameters via JSON body.
    """
    try:
        data = request.get_json() or request.form.to_dict()
        gam_type = _get_param(data, "type", "text")
        gam_dir_str = _get_param(data, "gam_dir")
        question = _get_param(data, "question")

        if not gam_dir_str:
            return jsonify({"success": False, "error": "gam_dir is required"})
        if not question:
            return jsonify({"success": False, "error": "question is required"})

        gam_dir = Path(gam_dir_str)
        if not gam_dir.exists():
            return jsonify({"success": False, "error": f"GAM directory not found: {gam_dir_str}"})

        workspace = LocalWorkspace(root_path=str(gam_dir))
        generator = _build_custom_generator(data)
        
        system_prompt = _get_param(data, "system_prompt", "")
        max_iter = int(_get_param(data, "max_iter", 10))
        verbose = _get_param(data, "verbose", True)

        if gam_type == "text":
            from ...agents.text_chat_agent import TextChatAgent
            
            tree = GAMTree.from_disk(gam_dir, workspace)
            agent = TextChatAgent(
                generator=generator,
                tree=tree,
                workspace=workspace,
                max_iterations=max_iter,
                verbose=verbose,
            )

            result = agent.request(
                system_prompt=system_prompt,
                user_prompt=question,
                max_iter=max_iter,
            )

        elif gam_type == "video":
            from ...agents.video_chat_agent import VideoChatAgent
            
            tree = VideoGAMTree.from_disk(gam_dir, workspace)
            
            # Video model can be different
            if _get_param(data, "video_model"):
                video_generator = _build_custom_generator(data, prefix="video_")
            else:
                # Try to use current_app's video_generator if not provided in request
                video_generator = current_app.config.get("VIDEO_GENERATOR")

            video_fps = float(_get_param(data, "video_fps", 1.0))
            video_max_resolution = int(_get_param(data, "video_max_resolution", 480))

            agent = VideoChatAgent(
                generator=generator,
                tree=tree,
                workspace=workspace,
                video_generator=video_generator,
                max_iterations=max_iter,
                verbose=verbose,
                video_fps=video_fps,
                video_max_resolution=video_max_resolution,
            )

            result = agent.request(
                system_prompt=system_prompt,
                user_prompt=question,
                max_iter=max_iter,
            )
        else:
            return jsonify({"success": False, "error": f"Unsupported gam type: {gam_type}"})

        return jsonify({
            "success": True,
            "question": result.question,
            "answer": result.answer,
            "sources": result.sources,
            "confidence": result.confidence,
            "notes": result.notes,
            "files_read": result.files_read,
            "dirs_explored": result.dirs_explored,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})
