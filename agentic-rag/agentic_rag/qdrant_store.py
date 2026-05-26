"""Qdrant vector store helpers."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from agentic_rag.config import Settings


def qdrant_client(settings: Settings) -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )


def get_vector_store(
    settings: Settings,
    embeddings: Embeddings,
) -> QdrantVectorStore:
    """Open an existing collection. Run ingest first if missing."""
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=settings.qdrant_collection_name,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )


def collection_exists(settings: Settings) -> bool:
    client = qdrant_client(settings)
    cols = client.get_collections().collections
    names = {c.name for c in cols}
    return settings.qdrant_collection_name in names
