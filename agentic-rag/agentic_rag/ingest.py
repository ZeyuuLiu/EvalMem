"""Load .txt / .md files into Qdrant."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agentic_rag.config import Settings
from agentic_rag.embeddings import build_embeddings
from agentic_rag.qdrant_store import collection_exists, qdrant_client


def _read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _gather_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in {".txt", ".md", ".markdown"} else []
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".md", ".markdown"}:
            out.append(p)
    return sorted(out)


def ingest_paths(
    settings: Settings,
    source: Path,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    recreate: bool = False,
) -> int:
    """Index files under `source` (file or directory). Returns number of chunks stored."""
    paths = _gather_paths(Path(source))
    if not paths:
        raise FileNotFoundError(f"No .txt/.md files under {source}")

    docs: list[Document] = []
    for p in paths:
        text = _read_file(p)
        docs.append(Document(page_content=text, metadata={"source": str(p)}))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = splitter.split_documents(docs)
    embeddings = build_embeddings(settings)

    client = qdrant_client(settings)
    exists = collection_exists(settings)

    if exists and recreate:
        client.delete_collection(settings.qdrant_collection_name)
        exists = False

    if not exists:
        QdrantVectorStore.from_documents(
            splits,
            embeddings,
            url=settings.qdrant_url,
            collection_name=settings.qdrant_collection_name,
            api_key=settings.qdrant_api_key,
        )
    else:
        store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=settings.qdrant_collection_name,
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        store.add_documents(splits)
    return len(splits)
