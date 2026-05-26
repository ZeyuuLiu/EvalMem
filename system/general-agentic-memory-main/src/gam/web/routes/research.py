# -*- coding: utf-8 -*-
"""
Research API routes – Q&A over GAM knowledge base.
"""

from __future__ import annotations

import json
from pathlib import Path
from queue import Queue
from typing import Dict
import threading

from flask import Blueprint, Response, current_app, jsonify, request

from ...core.tree import GAMTree
from ...workspaces.local_workspace import LocalWorkspace

research_bp = Blueprint("research", __name__)


# ---------------------------------------------------------------------------
# /api/research  (synchronous)
# ---------------------------------------------------------------------------

@research_bp.route("/api/research", methods=["POST"])
def api_research():
    """
    使用 TextChatAgent 对 GAM 进行问答

    Request:
        - question: 用户的问题
        - gam_path: GAM 目录路径

    Response:
        - success / answer / sources / confidence
    """
    try:
        generator = current_app.config.get("GENERATOR")
        if not generator:
            return jsonify({
                "success": False,
                "error": "LLM generator not configured. Research requires an LLM.",
            })

        question = request.form.get("question", "").strip()
        gam_path = request.form.get("gam_path", "").strip()

        if not question:
            return jsonify({"success": False, "error": "Question is required"})
        if not gam_path:
            return jsonify({"success": False, "error": "GAM path is required"})

        gam_path = Path(gam_path)
        if not gam_path.exists():
            return jsonify({"success": False, "error": "GAM path does not exist"})

        workspace = LocalWorkspace(
            root_path=str(gam_path),
            name="research_workspace",
            description=f"Research workspace for {gam_path.name}",
        )

        tree = GAMTree.from_disk(gam_path, workspace)
        gam_cache: Dict = current_app.config["GAM_CACHE"]
        gam_cache[str(gam_path)] = tree

        from ...agents.text_chat_agent import TextChatAgent

        chat_agent = TextChatAgent(
            generator=generator,
            tree=tree,
            workspace=workspace,
            max_iterations=20,
            verbose=True,
        )

        result = chat_agent.chat(question=question, max_rounds=20)

        return jsonify({
            "success": True,
            "answer": result.answer,
            "sources": result.sources if result.sources else result.files_read,
            "confidence": result.confidence,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# /api/research_stream  (SSE streaming)
# ---------------------------------------------------------------------------

@research_bp.route("/api/research_stream", methods=["POST"])
def api_research_stream():
    """
    使用 TextChatAgent 对 GAM 进行问答（SSE 流式推送 agent 动作）

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
            name="research_workspace",
            description=f"Research workspace for {gam_path_obj.name}",
        )
        tree = GAMTree.from_disk(gam_path_obj, workspace)
        gam_cache: Dict = current_app.config["GAM_CACHE"]
        gam_cache[str(gam_path_obj)] = tree

        from ...agents.text_chat_agent import TextChatAgent

        chat_agent = TextChatAgent(
            generator=generator,
            tree=tree,
            workspace=workspace,
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
                    event = event_queue.get(timeout=120)
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
