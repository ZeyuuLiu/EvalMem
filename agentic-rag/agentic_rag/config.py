"""Load settings from environment (.env in project root)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_env() -> None:
    env_path = _project_root() / ".env"
    if env_path.is_file():
        load_dotenv(env_path)
    load_dotenv(override=False)


@dataclass(frozen=True)
class Settings:
    vllm_base_url: str
    vllm_api_key: str
    vllm_model_name: str
    embedding_model_path: str
    embedding_dim: int
    embedding_device: str
    embedding_backend: str
    embedding_base_url: str
    embedding_api_key: str
    embedding_model_name: str
    qdrant_url: str
    qdrant_api_key: str | None
    qdrant_collection_name: str
    retrieve_k: int
    # retrieve：先向量检索再 LLM（默认，兼容 Qwen3+vLLM 工具解析问题）；react：LangGraph ReAct（需可靠 tool_calls）
    rag_mode: str


def get_settings() -> Settings:
    load_env()
    return Settings(
        vllm_base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        vllm_api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
        vllm_model_name=os.getenv("VLLM_MODEL_NAME", "Qwen3-8B"),
        embedding_model_path=os.getenv(
            "EMBEDDING_MODEL_PATH",
            "/path/to/Qwen3-Embedding-0.6B",
        ),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
        embedding_device=os.getenv("EMBEDDING_DEVICE", "cuda"),
        embedding_backend=_normalize_embedding_backend(
            os.getenv("EMBEDDING_BACKEND", "openai")
        ),
        embedding_base_url=os.getenv("EMBEDDING_BASE_URL", "http://127.0.0.1:8001/v1"),
        embedding_api_key=os.getenv("EMBEDDING_API_KEY", "EMPTY"),
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "Qwen3-Embedding-0.6B"),
        qdrant_url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
        qdrant_collection_name=os.getenv("QDRANT_COLLECTION_NAME", "agentic_rag_docs"),
        retrieve_k=int(os.getenv("RETRIEVE_K", "8")),
        rag_mode=_normalize_rag_mode(os.getenv("RAG_MODE", "retrieve")),
    )


def _normalize_rag_mode(raw: str | None) -> str:
    m = (raw or "retrieve").strip().lower()
    return m if m in ("react", "retrieve") else "retrieve"


def _normalize_embedding_backend(raw: str | None) -> str:
    b = (raw or "openai").strip().lower()
    return b if b in ("openai", "local") else "openai"
