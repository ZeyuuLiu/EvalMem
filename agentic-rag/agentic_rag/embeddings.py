"""Embedding provider for Qdrant (OpenAI-compatible API or local model)."""

from __future__ import annotations

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from agentic_rag.config import Settings


def build_embeddings(settings: Settings) -> OpenAIEmbeddings | HuggingFaceEmbeddings:
    if settings.embedding_backend == "openai":
        # Use deployed vLLM embedding service (OpenAI-compatible /v1/embeddings).
        return OpenAIEmbeddings(
            model=settings.embedding_model_name,
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
        )

    mk: dict = {"trust_remote_code": True}
    dev = settings.embedding_device
    if dev == "cpu":
        mk["device"] = "cpu"
    else:
        mk["device"] = dev
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model_path,
        model_kwargs=mk,
        encode_kwargs={"normalize_embeddings": True},
    )
