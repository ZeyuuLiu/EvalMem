"""
并行多视图 + BM25 检索 + RRF 融合。

- parallel_retrieve: 调度 dense (views 1..N) + sparse (BM25) 检索
- build_bm25_index:  根据当前 memory_corpus 构建 BM25 倒排索引
- rrf_fuse:          Reciprocal Rank Fusion，多视图候选融合
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from memory_eval.eval_core.agentic_rag.views import QueryView


@dataclass
class Candidate:
    """检索阶段统一候选记录。"""

    id: str
    text: str
    score: float = 0.0
    source_view: str = ""
    rank_in_view: int = -1
    meta: dict[str, Any] = field(default_factory=dict)


# Type alias: embed_fn 接受文本列表返回向量列表
EmbedFn = Callable[[list[str]], list[list[float]]]


@dataclass
class BM25Index:
    """BM25 倒排索引数据结构（skeleton 轻量实现，避免引入 rank_bm25 依赖）。

    TODO: replace with `rank_bm25.BM25Okapi` once external dep is allowed.
    """

    docs: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    doc_tokens: list[list[str]] = field(default_factory=list)
    df: dict[str, int] = field(default_factory=dict)
    avg_doc_len: float = 0.0
    k1: float = 1.5
    b: float = 0.75


# ---------------------------------------------------------------------------
# BM25 (sparse view 5)
# ---------------------------------------------------------------------------

_BM25_TOKEN_RE = re.compile(r"[\w一-鿿]+", flags=re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """skeleton 简单正则切分；中文按字、英文按词。

    TODO: 中英混合 LoCoMo 需替换为 jieba + regex（见设计文档增量 4 风险表）。
    """
    return [t.lower() for t in _BM25_TOKEN_RE.findall(text or "")]


def build_bm25_index(memory_corpus: list[dict[str, Any]]) -> BM25Index:
    """一次性构建 BM25 倒排索引。内存预算 ~100MB/1500 chunks。"""
    docs, doc_ids, doc_tokens, df = [], [], [], {}
    for record in memory_corpus:
        if not isinstance(record, dict):
            continue
        text = str(record.get("text", ""))
        if not text:
            continue
        rid = str(record.get("id", f"bm25-doc-{len(docs)}"))
        tokens = _tokenize(text)
        docs.append(text)
        doc_ids.append(rid)
        doc_tokens.append(tokens)
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1
    avg = sum(len(t) for t in doc_tokens) / len(doc_tokens) if doc_tokens else 0.0
    return BM25Index(docs=docs, doc_ids=doc_ids, doc_tokens=doc_tokens, df=df, avg_doc_len=avg)


def _bm25_score(index: BM25Index, query_tokens: list[str]) -> list[float]:
    if not index.docs:
        return []
    n_docs = len(index.docs)
    scores = [0.0] * n_docs
    for term in query_tokens:
        if term not in index.df:
            continue
        df = index.df[term]
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        for i, doc in enumerate(index.doc_tokens):
            tf = doc.count(term)
            if tf == 0:
                continue
            denom = tf + index.k1 * (1.0 - index.b + index.b * (len(doc) / max(index.avg_doc_len, 1.0)))
            scores[i] += idf * tf * (index.k1 + 1.0) / max(denom, 1e-9)
    return scores


def _bm25_retrieve(index: BM25Index, query: str, top_k: int) -> list[Candidate]:
    tokens = _tokenize(query)
    scores = _bm25_score(index, tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    out: list[Candidate] = []
    for rank, (doc_idx, score) in enumerate(ranked):
        if score <= 0:
            continue
        out.append(
            Candidate(
                id=index.doc_ids[doc_idx],
                text=index.docs[doc_idx],
                score=float(score),
                source_view="bm25",
                rank_in_view=rank,
                meta={"retrieval": "bm25"},
            )
        )
    return out


# ---------------------------------------------------------------------------
# Dense semantic retrieval
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    return num / (da * db) if da and db else 0.0


def _dense_retrieve(
    query: str,
    memory_corpus: list[dict[str, Any]],
    embed_fn: EmbedFn | None,
    top_k: int,
    view_type: str,
) -> list[Candidate]:
    """单视图密集向量检索；embed_fn=None 时退化到字符串包含兜底。"""
    if not memory_corpus:
        return []
    if embed_fn is None:
        return _string_match_fallback(query, memory_corpus, top_k, view_type)

    texts = [str(r.get("text", "")) for r in memory_corpus]
    try:
        vectors = embed_fn([query] + texts)
    except Exception:
        return _string_match_fallback(query, memory_corpus, top_k, view_type)
    if not vectors or len(vectors) < 2:
        return _string_match_fallback(query, memory_corpus, top_k, view_type)

    q_vec, doc_vecs = vectors[0], vectors[1:]
    scored = [(i, _cosine(q_vec, dv)) for i, dv in enumerate(doc_vecs)]
    scored.sort(key=lambda x: x[1], reverse=True)
    out: list[Candidate] = []
    for rank, (i, score) in enumerate(scored[:top_k]):
        record = memory_corpus[i]
        out.append(
            Candidate(
                id=str(record.get("id", f"{view_type}-{i}")),
                text=str(record.get("text", "")),
                score=float(score),
                source_view=view_type,
                rank_in_view=rank,
                meta=dict(record.get("meta", {}) or {}),
            )
        )
    return out


def _string_match_fallback(
    query: str,
    memory_corpus: list[dict[str, Any]],
    top_k: int,
    view_type: str,
) -> list[Candidate]:
    """无 embedding 时的最后兜底，保证 retriever 不崩。"""
    q_lower = query.lower()
    scored: list[tuple[int, float]] = []
    for i, record in enumerate(memory_corpus):
        text = str(record.get("text", "")).lower()
        if not text:
            continue
        hits = sum(1 for tok in q_lower.split() if tok and tok in text)
        scored.append((i, float(hits)))
    scored.sort(key=lambda x: x[1], reverse=True)
    out: list[Candidate] = []
    for rank, (i, score) in enumerate(scored[:top_k]):
        if score <= 0:
            break
        record = memory_corpus[i]
        out.append(
            Candidate(
                id=str(record.get("id", f"{view_type}-{i}")),
                text=str(record.get("text", "")),
                score=score,
                source_view=view_type,
                rank_in_view=rank,
                meta={"fallback": "string_match"},
            )
        )
    return out


# ---------------------------------------------------------------------------
# Parallel multi-view dispatcher
# ---------------------------------------------------------------------------


def parallel_retrieve(
    views: list[QueryView],
    memory_corpus: list[dict[str, Any]],
    top_k: int,
    *,
    embed_fn: EmbedFn | None = None,
    bm25_index: BM25Index | None = None,
    max_workers: int = 5,
) -> dict[str, list[Candidate]]:
    """并发执行多视图检索；同 view_type 多次出现时按 score 合并去重。"""
    if not views:
        return {}

    def _do(view: QueryView) -> tuple[str, list[Candidate]]:
        if view.view_type == "bm25":
            if bm25_index is None:
                return view.view_type, []
            return view.view_type, _bm25_retrieve(bm25_index, view.query, top_k)
        return view.view_type, _dense_retrieve(view.query, memory_corpus, embed_fn, top_k, view.view_type)

    workers = max(1, min(max_workers, len(views)))
    results: dict[str, list[Candidate]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for view_type, hits in ex.map(_do, views):
            results[view_type] = _merge_view_hits(results.get(view_type, []), hits)
    return results


def _merge_view_hits(prev: list[Candidate], new: list[Candidate]) -> list[Candidate]:
    """同视图多轮合并：按 id 去重，保留较高 score，并重打 rank。"""
    by_id: dict[str, Candidate] = {c.id: c for c in prev}
    for c in new:
        if c.id not in by_id or c.score > by_id[c.id].score:
            by_id[c.id] = c
    out = sorted(by_id.values(), key=lambda x: x.score, reverse=True)
    for i, c in enumerate(out):
        c.rank_in_view = i
    return out


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def rrf_fuse(
    view_results: dict[str, list[Candidate]],
    *,
    k_param: int = 60,
    view_weights: dict[str, float] | None = None,
) -> list[Candidate]:
    """Reciprocal Rank Fusion：每候选累计 weight / (k_param + rank + 1)。"""
    if not view_results:
        return []
    weights = view_weights or {}
    rrf_scores: dict[str, float] = {}
    contributing: dict[str, list[str]] = {}
    text_map: dict[str, str] = {}
    meta_map: dict[str, dict[str, Any]] = {}
    best_orig: dict[str, float] = {}

    for view_type, cands in view_results.items():
        w = float(weights.get(view_type, 1.0))
        for rank, c in enumerate(cands):
            rrf_scores[c.id] = rrf_scores.get(c.id, 0.0) + w / (k_param + rank + 1)
            contributing.setdefault(c.id, []).append(view_type)
            text_map.setdefault(c.id, c.text)
            meta_map.setdefault(c.id, dict(c.meta))
            best_orig[c.id] = max(best_orig.get(c.id, 0.0), float(c.score))

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    out: list[Candidate] = []
    for rank, (cid, score) in enumerate(fused):
        meta = dict(meta_map.get(cid, {}))
        meta["rrf_contributing_views"] = list(contributing.get(cid, []))
        meta["max_view_score"] = best_orig.get(cid, 0.0)
        out.append(
            Candidate(
                id=cid,
                text=text_map.get(cid, ""),
                score=float(score),
                source_view="rrf",
                rank_in_view=rank,
                meta=meta,
            )
        )
    return out
