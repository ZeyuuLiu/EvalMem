#!/usr/bin/env python3
"""CLI: ingest documents, or chat with the agentic RAG agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python main.py` from agentic-rag/ without installing package
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agentic_rag.config import get_settings
from agentic_rag.graph import build_agent
from agentic_rag.ingest import ingest_paths
from agentic_rag.qdrant_store import collection_exists
from agentic_rag.rag_pipeline import run_rag_turn


def _print_assistant_reply(msgs: list) -> None:
    """Prefer the last AIMessage without tool_calls (final answer after tools)."""
    last_ai: AIMessage | None = None
    for m in reversed(msgs):
        if isinstance(m, AIMessage):
            last_ai = m
            tc = getattr(m, "tool_calls", None) or []
            if not tc and (m.content or "").strip():
                print(f"Assistant: {m.content}\n")
                return
    if last_ai is not None and (last_ai.content or "").strip():
        print(f"Assistant: {last_ai.content}\n")
        return
    n_tools = sum(1 for m in msgs if isinstance(m, ToolMessage))
    if n_tools:
        print(
            "Assistant: （模型未返回最终文本；若仅见 <tool_call> 而无检索结果，"
            "请确认 vLLM 已开启 --enable-auto-tool-choice 且 graph 中已关闭 Qwen3 enable_thinking）\n"
        )
    else:
        print(f"Assistant: {msgs}\n")


def cmd_ingest(args: argparse.Namespace) -> None:
    settings = get_settings()
    n = ingest_paths(
        settings,
        Path(args.source),
        recreate=args.recreate,
    )
    print(f"Indexed {n} chunks into collection '{settings.qdrant_collection_name}'.")


def cmd_chat(args: argparse.Namespace) -> None:
    settings = get_settings()
    if not collection_exists(settings):
        print(
            "Qdrant collection not found. Run first:\n"
            f"  python main.py ingest <path_to_txt_or_md_dir> --recreate",
            file=sys.stderr,
        )
        sys.exit(1)
    use_react = settings.rag_mode == "react"
    if use_react:
        agent = build_agent(settings)
        print("Agentic RAG（ReAct 工具模式）ready. Ctrl+D / empty line to exit.\n")
    else:
        print(
            "Agentic RAG（先检索再回答，RAG_MODE=retrieve）ready. "
            "Ctrl+D / empty line to exit.\n"
        )
    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        if use_react:
            result = agent.invoke({"messages": [HumanMessage(content=line)]})
            msgs = result.get("messages", [])
            _print_assistant_reply(msgs)
        else:
            reply = run_rag_turn(settings, line)
            print(f"Assistant: {reply}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG (LangGraph + Qdrant + vLLM)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ing = sub.add_parser("ingest", help="Index .txt/.md files into Qdrant")
    p_ing.add_argument("source", help="File or directory")
    p_ing.add_argument(
        "--recreate",
        action="store_true",
        help="Delete existing collection and rebuild",
    )
    p_ing.set_defaults(func=cmd_ingest)

    p_chat = sub.add_parser("chat", help="Interactive chat (needs vLLM + Qdrant + ingest)")
    p_chat.set_defaults(func=cmd_chat)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
