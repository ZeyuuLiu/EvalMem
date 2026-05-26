"""
Shared lightweight utilities for native-bridge adapters.

Provides:
- Embedding service (sentence_transformers if available, else lexical fallback)
- In-memory vector store (cosine similarity)
- Lightweight LLM call (OpenAI-compatible chat)
- Time-bucket helpers (for hierarchical adapters like TiMem / MemoryOS)

All native adapters (TiMem / MemoryOS / EverMemOS) share these primitives so they
can produce DIFFERENT numbers from each other (each implements its own design
philosophy on top of this shared substrate), without depending on Qdrant or
heavy framework setup.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =====================================================================
# Embedding service
# =====================================================================

class EmbeddingService:
    """
    Lightweight embedding service.

    Tries sentence_transformers first; falls back to TF-IDF-style lexical signature
    if the dependency is unavailable. Exposes `embed(texts) -> List[List[float]]`.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._mode = "uninitialized"
        self._init_model()

    def _init_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._mode = "sentence_transformer"
            logger.info(f"EmbeddingService: loaded {self.model_name} ({self.device})")
        except Exception as exc:  # pragma: no cover - depends on env
            logger.warning(
                f"EmbeddingService: sentence_transformers unavailable ({exc}); "
                f"falling back to lexical mode"
            )
            self._model = None
            self._mode = "lexical_fallback"

    @property
    def mode(self) -> str:
        return self._mode

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self._mode == "sentence_transformer" and self._model is not None:
            try:
                vecs = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
                return [list(map(float, v)) for v in vecs]
            except Exception as exc:
                logger.warning(f"EmbeddingService: encode failed, using lexical fallback: {exc}")
        return [self._lexical_signature(t) for t in texts]

    @staticmethod
    def _lexical_signature(text: str, dim: int = 256) -> List[float]:
        """
        Very simple bag-of-character-trigram hashed projection for fallback.
        Not a real embedding, but supports cosine ranking when ST is unavailable.
        """
        vec = [0.0] * dim
        text_norm = re.sub(r"\s+", " ", text.lower().strip())
        if not text_norm:
            return vec
        # char trigrams
        for i in range(len(text_norm) - 2):
            tri = text_norm[i: i + 3]
            h = hash(tri) % dim
            vec[h] += 1.0
        # word tokens
        for tok in text_norm.split():
            h = hash(tok) % dim
            vec[h] += 2.0
        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


# =====================================================================
# In-memory vector store
# =====================================================================

@dataclass
class VectorStoreItem:
    item_id: str
    text: str
    vector: List[float]
    meta: Dict[str, Any] = field(default_factory=dict)


class InMemoryVectorStore:
    """Cosine-similarity vector store; replaces Qdrant for lightweight adapters."""

    def __init__(self) -> None:
        self.items: List[VectorStoreItem] = []
        self._id_index: Dict[str, int] = {}

    def upsert(self, item_id: str, text: str, vector: List[float], meta: Optional[Dict[str, Any]] = None) -> None:
        meta = dict(meta or {})
        if item_id in self._id_index:
            idx = self._id_index[item_id]
            self.items[idx] = VectorStoreItem(item_id=item_id, text=text, vector=vector, meta=meta)
        else:
            self._id_index[item_id] = len(self.items)
            self.items.append(VectorStoreItem(item_id=item_id, text=text, vector=vector, meta=meta))

    def search(self, query_vector: List[float], top_k: int = 10, *, filter_fn: Optional[Callable[[VectorStoreItem], bool]] = None) -> List[Tuple[float, VectorStoreItem]]:
        if not self.items or not query_vector:
            return []
        results: List[Tuple[float, VectorStoreItem]] = []
        for item in self.items:
            if filter_fn is not None and not filter_fn(item):
                continue
            score = self._cosine(query_vector, item.vector)
            results.append((score, item))
        results.sort(key=lambda x: x[0], reverse=True)
        return results[: max(1, top_k)]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return float(dot / (na * nb))

    def __len__(self) -> int:
        return len(self.items)


# =====================================================================
# BM25 (used by EverMemOS)
# =====================================================================

class SimpleBM25:
    """Lightweight BM25 implementation for hybrid retrieval (no rank_bm25 dependency)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: List[List[str]] = []
        self.doc_ids: List[str] = []
        self.doc_meta: List[Dict[str, Any]] = []
        self.df: Dict[str, int] = {}
        self.avg_dl: float = 0.0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[\w一-鿿]+", str(text).lower())

    def add(self, doc_id: str, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        tokens = self._tokenize(text)
        self.docs.append(tokens)
        self.doc_ids.append(doc_id)
        self.doc_meta.append(dict(meta or {"text": text}))
        for term in set(tokens):
            self.df[term] = self.df.get(term, 0) + 1
        total_dl = sum(len(d) for d in self.docs)
        self.avg_dl = total_dl / max(len(self.docs), 1)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[float, str, Dict[str, Any]]]:
        if not self.docs:
            return []
        q_tokens = self._tokenize(query)
        n_docs = len(self.docs)
        scores: List[Tuple[float, str, Dict[str, Any]]] = []
        for i, doc in enumerate(self.docs):
            score = 0.0
            dl = len(doc) or 1
            for term in q_tokens:
                if term not in self.df:
                    continue
                tf = doc.count(term)
                if tf == 0:
                    continue
                idf = math.log((n_docs - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_dl))
                score += idf * tf * (self.k1 + 1.0) / max(denom, 1e-9)
            if score > 0:
                scores.append((score, self.doc_ids[i], self.doc_meta[i]))
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[: max(1, top_k)]


# =====================================================================
# OpenAI-compatible LLM call
# =====================================================================

def llm_chat(api_key: str, base_url: str, model: str, messages: List[Dict[str, str]],
             temperature: float = 0.0, max_tokens: int = 1024, timeout: float = 60.0) -> str:
    """Synchronous OpenAI-compatible chat completion. Returns the text content."""
    import requests  # local import to avoid import-time hard dep
    if not api_key or not base_url:
        return ""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return str(data["choices"][0]["message"]["content"]).strip()
    except Exception as exc:
        logger.warning(f"llm_chat failed: {exc}")
        return ""


# =====================================================================
# Time bucketing helpers (used by TiMem / MemoryOS)
# =====================================================================

def parse_timestamp(ts: str) -> Optional[datetime]:
    """Parse common timestamp formats; return None if all parsers fail."""
    if not ts:
        return None
    # Try ISO 8601 first
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        pass
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
        "%d %B %Y %H:%M:%S",
        "%I:%M %p on %d %B %Y",
    ):
        try:
            return datetime.strptime(ts, fmt)
        except Exception:
            continue
    return None


def time_bucket(ts: Optional[datetime], grain: str) -> str:
    """Return a stable bucket key for the given timestamp at given granularity."""
    if ts is None:
        return f"unknown_{grain}"
    if grain == "hour":
        return ts.strftime("%Y%m%d_%H")
    if grain == "day":
        return ts.strftime("%Y%m%d")
    if grain == "week":
        iso_year, iso_week, _ = ts.isocalendar()
        return f"{iso_year}W{iso_week:02d}"
    if grain == "month":
        return ts.strftime("%Y%m")
    if grain == "session":
        return ts.strftime("%Y%m%d_%H")
    return ts.strftime("%Y%m%d")


__all__ = [
    "EmbeddingService",
    "InMemoryVectorStore",
    "VectorStoreItem",
    "SimpleBM25",
    "llm_chat",
    "parse_timestamp",
    "time_bucket",
]
