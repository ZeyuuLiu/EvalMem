# -*- coding: utf-8 -*-
"""
Long Horizontal API routes – agent loop with GAM memory visualization.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from queue import Queue
from typing import Dict, List

import openai
from flask import Blueprint, Response, current_app, jsonify, request
from rank_bm25 import BM25Okapi

from ..helpers import read_uploaded_files, get_timestamp_dir, DEFAULT_OUTPUT_BASE


def strip_think_tag(text: str) -> str:
    """Remove <think> ... </think> tags and their content from text."""
    if not text:
        return text
    return text.split("</think>")[-1].strip()


def _chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks of chunk_size characters."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size
    return chunks

long_horizontal_bp = Blueprint("long_horizontal", __name__)

_sessions: Dict[str, dict] = {}


class WebBM25Searcher:
    """BM25 searcher built from uploaded documents."""

    def __init__(self, documents: list[dict]):
        self.corpus_data = []
        for i, doc in enumerate(documents):
            self.corpus_data.append({
                "id": f"doc_{i}",
                "filename": doc["filename"],
                "text": doc["content"],
            })
        self.documents = [d["text"] for d in self.corpus_data]
        self.tokenized_corpus = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, k: int = 3) -> list[dict]:
        tokenized_query = query.lower().split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True
        )[:k]
        results = []
        for i in top_indices:
            if doc_scores[i] > 0:
                doc = self.corpus_data[i]
                text = doc["text"]
                if len(text) > 3000:
                    text = text[:3000] + "\n... [truncated]"
                results.append({
                    "docid": doc["id"],
                    "filename": doc["filename"],
                    "score": round(float(doc_scores[i]), 4),
                    "text": text,
                })
        return results

    def search_description(self, k: int = 3) -> str:
        return (
            f"Search the uploaded document collection using BM25. "
            f"Returns top-{k} results with docid, filename, score, and text."
        )


# ── Upload docs ──────────────────────────────────────────────────────────────

@long_horizontal_bp.route("/api/long_horizontal/upload_docs", methods=["POST"])
def api_upload_docs():
    try:
        files = request.files.getlist("files")
        if not files or all(not f.filename for f in files):
            return jsonify({"success": False, "error": "No files uploaded"})

        entries = read_uploaded_files(files)
        if not entries:
            return jsonify({"success": False, "error": "No valid files found"})

        # Split each document into 5000-char chunks to build the doc collection
        chunk_size = 5000
        doc_chunks: List[dict] = []
        file_info = []
        for entry in entries:
            text = entry["content"]
            chunks = _chunk_text(text, chunk_size)
            n = len(chunks)
            file_info.append({
                "filename": entry["filename"],
                "length": len(text),
                "chunk_count": n,
            })
            if n <= 1:
                doc_chunks.append({
                    "filename": entry["filename"],
                    "content": text,
                })
            else:
                for i, chunk in enumerate(chunks):
                    doc_chunks.append({
                        "filename": f"{entry['filename']} [chunk {i + 1}/{n}]",
                        "content": chunk,
                    })

        session_id = get_timestamp_dir()
        output_base = current_app.config.get("OUTPUT_BASE", DEFAULT_OUTPUT_BASE)
        gam_dir = os.path.join(output_base, f"long_horizontal_{session_id}", "gam")

        searcher = WebBM25Searcher(doc_chunks)

        _sessions[session_id] = {
            "documents": entries,
            "doc_chunks": doc_chunks,
            "searcher": searcher,
            "gam_dir": gam_dir,
            "status": "ready",
        }

        return jsonify({
            "success": True,
            "session_id": session_id,
            "file_count": len(entries),
            "doc_count": len(doc_chunks),
            "documents": file_info,
            "gam_dir": gam_dir,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


# ── Start agent request (SSE) ──────────────────────────────────────────────────

@long_horizontal_bp.route("/api/long_horizontal/start_request", methods=["POST"])
def api_start_request():
    try:
        session_id = request.form.get("session_id", "").strip()
        question = request.form.get("question", "").strip()

        if not session_id or session_id not in _sessions:
            return _sse_error("Invalid session. Please upload documents first.")
        if not question:
            return _sse_error("Question is required.")

        session = _sessions[session_id]
        searcher = session["searcher"]
        gam_dir = session["gam_dir"]

        generator = current_app.config.get("GENERATOR")
        if not generator:
            return _sse_error("LLM not configured.")

        gen_cfg = generator.config
        event_queue: Queue = Queue()

        def _worker():
            try:
                _request_agent_loop(
                    searcher=searcher,
                    question=question,
                    gam_dir=gam_dir,
                    model=gen_cfg.model_name,
                    api_key=gen_cfg.api_key,
                    api_base=gen_cfg.base_url,
                    event_queue=event_queue,
                )
            except Exception as exc:
                import traceback
                traceback.print_exc()
                event_queue.put({"type": "error", "error": str(exc)})

        threading.Thread(target=_worker, daemon=True).start()

        def generate():
            while True:
                try:
                    ev = event_queue.get(timeout=300)
                except Exception:
                    yield _sse_line({"type": "error", "error": "Timeout"})
                    break
                yield _sse_line(ev)
                if ev.get("type") in ("complete", "error"):
                    break

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return _sse_error(str(e))


# ── Chat with GAM (reuse research stream pattern) ───────────────────────────

@long_horizontal_bp.route("/api/long_horizontal/chat", methods=["POST"])
def api_lh_chat():
    try:
        generator = current_app.config.get("GENERATOR")
        if not generator:
            return _sse_error("LLM not configured.")

        question = request.form.get("question", "").strip()
        gam_path = request.form.get("gam_path", "").strip()

        if not question:
            return _sse_error("Question is required.")
        if not gam_path or not Path(gam_path).exists():
            return _sse_error("GAM path is required or does not exist.")

        from ...core.tree import GAMTree
        from ...workspaces.local_workspace import LocalWorkspace
        from ...agents.text_chat_agent import TextChatAgent

        gam_path_obj = Path(gam_path)
        workspace = LocalWorkspace(root_path=str(gam_path_obj))
        tree = GAMTree.from_disk(gam_path_obj, workspace)

        chat_agent = TextChatAgent(
            generator=generator, tree=tree, workspace=workspace,
            max_iterations=20, verbose=True,
        )

        event_queue: Queue = Queue()

        def action_callback(event_type, data):
            event_queue.put({"type": event_type, **data})

        def _worker():
            try:
                result = chat_agent.chat(
                    question=question, max_rounds=20,
                    action_callback=action_callback,
                )
                event_queue.put({
                    "type": "complete",
                    "answer": result.answer,
                    "sources": result.sources if result.sources else result.files_read,
                    "confidence": result.confidence,
                })
            except Exception as exc:
                import traceback
                traceback.print_exc()
                event_queue.put({"type": "error", "error": str(exc)})

        threading.Thread(target=_worker, daemon=True).start()

        def generate():
            while True:
                try:
                    ev = event_queue.get(timeout=120)
                except Exception:
                    yield _sse_line({"type": "error", "error": "Timeout"})
                    break
                yield _sse_line(ev)
                if ev.get("type") in ("complete", "error"):
                    break

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return _sse_error(str(e))


# ── Core agent loop ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an advanced research agent with access to a document search engine "
    "and a memory system (GAM – General Agentic Memory).\n\n"
    "TOOLS:\n"
    "1. **search** – Search the uploaded documents using BM25.\n"
    "2. **memorize** – Compress search results into GAM memory. "
    "Call with search indices to replace verbose raw results with concise summaries, "
    "freeing context window for more searches.\n"
    "3. **recall** – Retrieve previously memorized information from GAM.\n\n"
    "STRATEGY:\n"
    "- Search for different aspects of the question using multiple search calls.\n"
    "- After every 2-3 searches, call memorize to compress and offload information.\n"
    "- Use recall to retrieve memorized info when needed for your final synthesis.\n"
    "- Finally provide a comprehensive answer based on all gathered information."
)


def _request_agent_loop(
    searcher: WebBM25Searcher,
    question: str,
    gam_dir: str,
    model: str,
    api_key: str,
    api_base: str,
    event_queue: Queue,
    max_tokens: int = 32768,
):
    from ...core.tree import GAMTree
    from ...workspaces.local_workspace import LocalWorkspace
    from ...agents.text_gam_agent import TextGAMAgent
    from ...agents.text_chat_agent import TextChatAgent
    from ...generators.openai_generator import OpenAIGenerator
    from ...generators.config import OpenAIGeneratorConfig

    client = openai.OpenAI(api_key=api_key, base_url=api_base)

    search_history: list[dict] = []
    gam_components: dict = {}

    def _ensure_gam():
        if gam_components.get("agent"):
            return
        cfg = OpenAIGeneratorConfig(
            model_name=model, api_key=api_key, base_url=api_base,
        )
        gen = OpenAIGenerator(cfg)
        gam_path = Path(gam_dir).resolve()
        gam_path.mkdir(parents=True, exist_ok=True)
        ws = LocalWorkspace(root_path=str(gam_path))
        try:
            tree = GAMTree.from_disk(gam_path, ws)
        except Exception:
            tree = GAMTree.create_empty(gam_path, name=gam_path.name)
        gam_components.update({
            "gam_dir": str(gam_path),
            "generator": gen, "workspace": ws, "tree": tree,
            "agent": TextGAMAgent(gen, tree, ws, use_chunking=False, verbose=True),
        })

    tool_defs = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": searcher.search_description(),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memorize",
                "description": (
                    "Compress and save previous search results into GAM memory. "
                    "This replaces the raw verbose search results in conversation with concise summaries. "
                    "Each search is identified by its 0-based index [Search #N]."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_indices": {
                            "type": "array", "items": {"type": "integer"},
                            "description": "Indices of searches to memorize (e.g. [0, 1], not doc indices)",
                        },
                        "question": {
                            "type": "string",
                            "description": "Optional guiding question for summarization",
                        },
                    },
                    "required": ["search_indices"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recall",
                "description": (
                    "Recall information from GAM memory. "
                    "Search through previously memorized knowledge."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Question to search in GAM memory",
                        }
                    },
                    "required": ["question"],
                },
            },
        },
    ]

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    event_queue.put({
        "type": "init",
        "system_prompt": SYSTEM_PROMPT,
        "user_query": question,
    })

    max_rounds = 15
    used_no_search_nudge = False
    used_memorize_nudge = False

    for round_num in range(1, max_rounds + 1):
        event_queue.put({"type": "round", "round": round_num, "max_rounds": max_rounds})

        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                tools=tool_defs, tool_choice="auto",
                max_tokens=min(max_tokens, 65536),
            )
        except Exception as exc:
            event_queue.put({"type": "error", "error": f"LLM error: {exc}"})
            return

        msg = resp.choices[0].message
        raw_content = msg.content or ""
        clean_content = strip_think_tag(raw_content)

        msg_dict: dict = {"role": "assistant"}
        if clean_content:
            msg_dict["content"] = clean_content
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(msg_dict)

        if clean_content:
            event_queue.put({
                "type": "assistant",
                "content": clean_content,
                "has_tool_calls": bool(msg.tool_calls),
            })

        if not msg.tool_calls:
            # Nudge if model answered without ever searching
            if not search_history and not used_no_search_nudge:
                used_no_search_nudge = True
                nudge = (
                    "Before giving a final answer, call the `search` tool at least once "
                    "and use the retrieved evidence from the uploaded documents."
                )
                messages.append({"role": "user", "content": nudge})
                event_queue.put({"type": "nudge", "content": nudge})
                continue

            # Nudge if searches have never been memorized
            if (search_history
                    and not any(s.get("memorized") for s in search_history)
                    and not used_memorize_nudge):
                used_memorize_nudge = True
                unmemorized = [s["index"] for s in search_history if not s.get("memorized")]
                nudge = (
                    f"You have {len(unmemorized)} search result(s) not yet saved to memory "
                    f"(indices: {unmemorized}). Please memorize them into GAM before giving "
                    "the final answer."
                )
                messages.append({"role": "user", "content": nudge})
                event_queue.put({"type": "nudge", "content": nudge})
                continue
            break

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}

            event_queue.put({
                "type": "tool_call",
                "tool_call_id": tc.id,
                "name": name,
                "args": args,
            })

            tool_result = _execute_tool(
                name, args, tc.id,
                searcher, search_history, messages,
                gam_components, _ensure_gam, event_queue,
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": tool_result,
            })

    final_answer = ""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("content"):
            final_answer = m["content"]
            break

    event_queue.put({
        "type": "complete",
        "answer": final_answer,
        "gam_dir": gam_dir,
        "search_count": len(search_history),
        "memorized_count": sum(1 for s in search_history if s.get("memorized")),
    })


def _execute_tool(
    name, args, tool_call_id,
    searcher, search_history, messages,
    gam_components, ensure_gam_fn, event_queue,
):
    if name == "search":
        query = args.get("query", "")
        hits = searcher.search(query)
        result_json = json.dumps(hits, indent=2, ensure_ascii=False)
        idx = len(search_history)
        search_history.append({
            "index": idx, "query": query,
            "result": result_json, "tool_call_id": tool_call_id,
            "memorized": False,
        })
        tool_result = f'[Search #{idx}] Query: "{query}"\n\n{result_json}'
        event_queue.put({
            "type": "tool_result",
            "tool_call_id": tool_call_id,
            "name": name,
            "search_index": idx,
            "query": query,
            "content": tool_result,
        })
        return tool_result

    elif name == "memorize":
        indices = args.get("search_indices", [])
        ensure_gam_fn()
        agent = gam_components["agent"]
        gen = gam_components["generator"]
        ws = gam_components["workspace"]
        stored_gam_dir = gam_components["gam_dir"]

        content_parts = []
        for idx in indices:
            if idx >= len(search_history):
                continue
            entry = search_history[idx]
            content_parts.append(
                f"## Search #{idx}\n\nQuery: {entry['query']}\n\nResults:\n{entry['result']}"
            )

        if not content_parts:
            r = "Error: No valid indices provided for memorize."
            event_queue.put({"type": "tool_result", "tool_call_id": tool_call_id, "name": name, "content": r})
            return r

        try:
            agent.add(content=content_parts)
        except Exception as e:
            r = f"Error: Failed to add content to GAM: {e}"
            event_queue.put({"type": "tool_result", "tool_call_id": tool_call_id, "name": name, "content": r})
            return r

        # Reload GAMTree from disk after adding content so ChatAgent sees the updated tree
        from ...core.tree import GAMTree
        gam_path = Path(stored_gam_dir).resolve()
        try:
            updated_tree = GAMTree.from_disk(gam_path, ws)
        except Exception:
            updated_tree = gam_components["tree"]
        gam_components["tree"] = updated_tree

        results_arr = []
        for idx in indices:
            if idx >= len(search_history):
                continue
            entry = search_history[idx]
            search_query = entry["query"]
            chat_question = args.get("question") or search_query

            try:
                from ...agents.text_chat_agent import TextChatAgent
                chat_agent = TextChatAgent(gen, updated_tree, workspace=ws)
                chat_res = chat_agent.chat(chat_question)
                gam_answer = strip_think_tag(chat_res.answer) or ""
            except Exception as e:
                gam_answer = f"(Content memorized but failed to generate refined answer: {e})"

            gam_tagged = (
                f"[GAM Memory Result] (refined from Search #{idx}, query: \"{search_query}\")\n\n"
                f"{gam_answer}"
            )

            for m in messages:
                if (isinstance(m, dict) and m.get("role") == "tool"
                        and m.get("tool_call_id") == entry["tool_call_id"]):
                    m["content"] = gam_tagged

            event_queue.put({
                "type": "memorize_update",
                "search_index": idx,
                "old_query": search_query,
                "new_content": gam_tagged,
                "gam_answer": gam_answer,
            })
            entry["memorized"] = True

            preview = gam_answer[:300] + "..." if len(gam_answer) > 300 else gam_answer
            results_arr.append({
                "index": idx,
                "query": search_query,
                "status": "success",
                "gam_answer_preview": preview,
            })

        r = (
            f"Memorized {len(indices)} search(es) into GAM. "
            "Original search results have been replaced with GAM refined answers."
        )
        event_queue.put({"type": "tool_result", "tool_call_id": tool_call_id, "name": name, "content": r})
        return r

    elif name == "recall":
        ensure_gam_fn()
        gen = gam_components["generator"]
        ws = gam_components["workspace"]
        stored_gam_dir = gam_components["gam_dir"]

        # Reload the latest tree from disk to reflect any recent memorize calls
        from ...core.tree import GAMTree
        from ...agents.text_chat_agent import TextChatAgent
        gam_path = Path(stored_gam_dir).resolve()
        try:
            latest_tree = GAMTree.from_disk(gam_path, ws)
            gam_components["tree"] = latest_tree
        except Exception:
            latest_tree = gam_components["tree"]

        chat_agent = TextChatAgent(gen, latest_tree, workspace=ws)
        res = chat_agent.chat(args.get("question", ""))
        answer = strip_think_tag(res.answer)
        r = answer
        event_queue.put({"type": "tool_result", "tool_call_id": tool_call_id, "name": name, "content": r})
        return r

    r = f"Error: Unknown tool: {name}"
    event_queue.put({"type": "tool_result", "tool_call_id": tool_call_id, "name": name, "content": r})
    return r


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sse_line(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse_error(msg: str) -> Response:
    def gen():
        yield _sse_line({"type": "error", "error": msg})
    return Response(gen(), mimetype="text/event-stream")
