# -*- coding: utf-8 -*-
"""
Video Pipeline API routes – video upload, GAM building, and status polling.
"""

from __future__ import annotations

import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Dict

from flask import Blueprint, Response, current_app, jsonify, request, send_file

from ...core.tree import VideoGAMTree
from ...workspaces.local_workspace import LocalWorkspace
from ..helpers import get_timestamp_dir, task_results

video_pipeline_bp = Blueprint("video_pipeline", __name__)


# ---------------------------------------------------------------------------
# /api/video/pipeline_start  (asynchronous)
# ---------------------------------------------------------------------------

@video_pipeline_bp.route("/api/video/pipeline_start", methods=["POST"])
def api_video_pipeline_start():
    """
    异步启动 Video GAM pipeline。

    上传文件：
        - video: 视频文件 (mp4, avi, mkv, mov, webm)
        - subtitle: 字幕文件 (srt, vtt) [可选]

    Returns:
        - task_id, gam_dir
    """
    try:
        generator = current_app.config.get("GENERATOR")
        if not generator:
            return jsonify({
                "success": False,
                "error": "LLM generator not configured. Video pipeline requires an LLM.",
            })

        # ---- Receive uploaded files ----
        video_file = request.files.get("video")
        subtitle_file = request.files.get("subtitle")

        if not video_file or not video_file.filename:
            return jsonify({"success": False, "error": "No video file uploaded."})

        # ---- Create output directories ----
        timestamp = get_timestamp_dir()
        base_dir = Path(current_app.config["OUTPUT_BASE"]) / f"video_{timestamp}"
        input_dir = base_dir / "input"
        gam_dir = base_dir / "gam"

        input_dir.mkdir(parents=True, exist_ok=True)
        gam_dir.mkdir(parents=True, exist_ok=True)

        # ---- Save uploaded files to input_dir ----
        # VideoGAMAgent expects exactly one .mp4 in the input directory
        video_ext = Path(video_file.filename).suffix.lower()
        video_save_name = f"video{video_ext}"
        video_save_path = input_dir / video_save_name
        video_file.save(str(video_save_path))

        # If the video is not .mp4, convert it or just rename for the agent
        # VideoGAMAgent globs for *.mp4, so we need to ensure it's .mp4
        if video_ext != ".mp4":
            mp4_path = input_dir / "video.mp4"
            # Try converting with ffmpeg
            import subprocess
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "quiet", "-i", str(video_save_path), "-c", "copy", str(mp4_path)],
                    check=True, timeout=300,
                )
                os.unlink(str(video_save_path))
            except Exception:
                # If conversion fails, just rename
                video_save_path.rename(mp4_path)

        if subtitle_file and subtitle_file.filename:
            srt_ext = Path(subtitle_file.filename).suffix.lower()
            srt_save_path = input_dir / f"subtitles{srt_ext}"
            subtitle_file.save(str(srt_save_path))

        # ---- Create async task ----
        task_id = str(uuid.uuid4())

        task_results[task_id] = {
            "status": "running",
            "stage": "uploading",
            "gam_dir": str(gam_dir),
            "input_dir": str(input_dir),
            "segment_count": 0,
            "error": None,
            "message": "Video uploaded, starting pipeline...",
        }

        # Capture app-level references
        gam_cache: Dict = current_app.config["GAM_CACHE"]
        video_generator = current_app.config.get("VIDEO_GENERATOR")

        def run_video_pipeline():
            try:
                from ...agents.video_gam_agent import VideoGAMAgent

                task_results[task_id]["stage"] = "probing"
                task_results[task_id]["message"] = "Analyzing video content..."

                workspace = LocalWorkspace(
                    root_path=str(gam_dir),
                    name="video_gam",
                    description=f"Video GAM built from uploaded video",
                )

                tree = VideoGAMTree.create_empty(gam_dir, name="video_gam")

                # VideoGAMAgent needs planner + segmentor
                # Use the same generator for both if no separate segmentor configured
                segmentor = current_app.config.get("SEGMENTOR") if hasattr(current_app, 'config') else None
                if segmentor is None:
                    segmentor = generator

                agent = VideoGAMAgent(
                    planner=generator,
                    segmentor=segmentor,
                    workspace=workspace,
                    tree=tree,
                )

                task_results[task_id]["stage"] = "segmenting"
                task_results[task_id]["message"] = "Segmenting video..."

                result = agent.add(
                    input_path=str(input_dir),
                    verbose=True,
                    caption_with_subtitles=True,
                )

                # Count segments
                seg_count = 0
                if hasattr(result, "segment_count"):
                    seg_count = result.segment_count
                elif hasattr(result, "segments"):
                    seg_count = len(result.segments) if result.segments else 0

                # Reload tree
                tree = VideoGAMTree.from_disk(gam_dir, workspace)
                gam_cache[str(gam_dir)] = tree

                task_results[task_id]["status"] = "completed"
                task_results[task_id]["segment_count"] = seg_count
                task_results[task_id]["message"] = "Video GAM built successfully!"

            except Exception as e:
                import traceback
                traceback.print_exc()
                task_results[task_id]["status"] = "error"
                task_results[task_id]["error"] = str(e)

        thread = threading.Thread(target=run_video_pipeline, daemon=True)
        thread.start()

        return jsonify({
            "success": True,
            "task_id": task_id,
            "gam_dir": str(gam_dir),
            "input_dir": str(input_dir),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# /api/video/pipeline_status
# ---------------------------------------------------------------------------

@video_pipeline_bp.route("/api/video/pipeline_status")
def api_video_pipeline_status():
    """获取 video pipeline 任务状态"""
    task_id = request.args.get("task_id", "").strip()

    if not task_id:
        return jsonify({"error": "task_id is required"})

    if task_id not in task_results:
        return jsonify({"error": "Task not found"})

    result = task_results[task_id]
    return jsonify(result)


# ---------------------------------------------------------------------------
# /api/video/serve  (serve video files for the HTML5 player)
# ---------------------------------------------------------------------------

@video_pipeline_bp.route("/api/video/serve")
def api_video_serve():
    """
    Serve a video file from the filesystem for the HTML5 video player.
    Supports Range requests for seeking.
    """
    file_path = request.args.get("path", "").strip()

    if not file_path:
        return jsonify({"error": "path is required"}), 400

    file_path = Path(file_path)
    if not file_path.exists() or not file_path.is_file():
        return jsonify({"error": "File not found"}), 404

    # Determine MIME type
    ext = file_path.suffix.lower()
    mime_map = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
    }
    mimetype = mime_map.get(ext, "application/octet-stream")

    # Support Range requests for video seeking
    file_size = file_path.stat().st_size
    range_header = request.headers.get("Range")

    if range_header:
        # Parse Range header
        byte_range = range_header.replace("bytes=", "")
        parts = byte_range.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1

        if start >= file_size:
            return Response(status=416)

        end = min(end, file_size - 1)
        length = end - start + 1

        def generate_range():
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = length
                chunk_size = 8192
                while remaining > 0:
                    read_size = min(chunk_size, remaining)
                    data = f.read(read_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        response = Response(
            generate_range(),
            status=206,
            mimetype=mimetype,
        )
        response.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        response.headers["Accept-Ranges"] = "bytes"
        response.headers["Content-Length"] = str(length)
        return response
    else:
        return send_file(
            str(file_path),
            mimetype=mimetype,
            conditional=True,
        )
