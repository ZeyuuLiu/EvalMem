"""Retriever exposed as a tool for the ReAct agent."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.tools import create_retriever_tool

from agentic_rag.config import Settings
from agentic_rag.qdrant_store import get_vector_store


def build_retriever_tool(settings: Settings, embeddings: Embeddings):
    store = get_vector_store(settings, embeddings)
    retriever = store.as_retriever(
        search_kwargs={"k": settings.retrieve_k},
    )
    return create_retriever_tool(
        retriever,
        "search_knowledge_base",
        (
            "Search the user's uploaded documents (novels, files indexed in Qdrant). "
            "Use for 人物、剧情、设定、台词、关系等问题. "
            "Input: a short retrieval query in the user's language; include character/place names without extra spaces."
        ),
    )
