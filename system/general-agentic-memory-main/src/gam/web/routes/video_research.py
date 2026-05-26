# -*- coding: utf-8 -*-
"""
Video Research API routes – Q&A over Video GAM knowledge base (SSE streaming).
"""

from __future__ import annotations

import json
from pathlib import Path
from queue import Queue
from typing import Dict
import threading

from flask import Blueprint, Response, current_app, jsonify, request

from ...core.tree import VideoGAMTree
from ...workspaces.local_workspace import LocalWorkspace

video_research_bp = Blueprint("video_research", __name__)


# ---------------------------------------------------------------------------
# /api/video/research_stream  (SSE streaming)
# ---------------------------------------------------------------------------

@video_research_bp.route("/api/video/research_stream", methods=["POST"])
def api_video_research_stream():
    """
    使用 VideoChatAgent 对 Video GAM 进行问答（SSE 流式推送 agent 动作）

    Event types:
        - tool_call / thinking / round / complete / error
    """
    try:
        generator = current_app.config.get("GENERATOR")
        if not generator:
            def error_gen():
                yield f"data: {json.dumps({'type': 'error', 'error': 'LLM generator not configured.'})}\n\n"
            return Response(error_gen(), mimetype="text/event-stream")

        question = request.form.get("question", "").strip()
        gam_path = request.form.get("gam_path", "").strip()

        if not question:
            def error_gen():
                yield f"data: {json.dumps({'type': 'error', 'error': 'Question is required'})}\n\n"
            return Response(error_gen(), mimetype="text/event-stream")

        if not gam_path:
            def error_gen():
                yield f"data: {json.dumps({'type': 'error', 'error': 'GAM path is required'})}\n\n"
            return Response(error_gen(), mimetype="text/event-stream")

        gam_path_obj = Path(gam_path)
        if not gam_path_obj.exists():
            def error_gen():
                yield f"data: {json.dumps({'type': 'error', 'error': 'GAM path does not exist'})}\n\n"
            return Response(error_gen(), mimetype="text/event-stream")

        workspace = LocalWorkspace(
            root_path=str(gam_path_obj),
            name="video_research_workspace",
            description=f"Video research workspace for {gam_path_obj.name}",
        )
        tree = VideoGAMTree.from_disk(gam_path_obj, workspace)
        gam_cache: Dict = current_app.config["GAM_CACHE"]
        gam_cache[str(gam_path_obj)] = tree

        from ...agents.video_chat_agent import VideoChatAgent

        # Use video_generator if available, otherwise fallback to generator
        video_generator = current_app.config.get("VIDEO_GENERATOR", generator)

        chat_agent = VideoChatAgent(
            generator=generator,
            tree=tree,
            workspace=workspace,
            video_generator=video_generator,
            max_iterations=20,
            verbose=True,
        )

        event_queue: Queue = Queue()

        def action_callback(event_type, data):
            event_queue.put({"type": event_type, **data})

        def request_agent():
            try:
                result = chat_agent.chat(
                    question=question,
                    max_rounds=20,
                    action_callback=action_callback,
                )
                event_queue.put({
                    "type": "complete",
                    "answer": result.answer,
                    "sources": result.sources if result.sources else result.files_read,
                    "confidence": result.confidence,
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                event_queue.put({"type": "error", "error": str(e)})

        thread = threading.Thread(target=request_agent, daemon=True)
        thread.start()

        def generate():
            while True:
                try:
                    event = event_queue.get(timeout=180)  # Longer timeout for video
                except Exception:
                    yield f"data: {json.dumps({'type': 'error', 'error': 'Timeout waiting for agent'})}\n\n"
                    break

                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                if event.get("type") in ("complete", "error"):
                    break

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    except Exception as e:
        import traceback
        traceback.print_exc()

        def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        return Response(error_gen(), mimetype="text/event-stream")
