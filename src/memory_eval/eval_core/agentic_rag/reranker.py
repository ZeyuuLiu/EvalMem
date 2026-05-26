"""
LLM-as-Reranker — v3 增量 1（借鉴 RankRAG）。

在 RRF top-N 之后插入一次 LLM 精排，对每个候选给出 time / speaker / content
三轴 0-10 评分，输出按 total 降序的 top-K。
"""

from __future__ import annotations

from typing import Any

from memory_eval.eval_core.agentic_rag.config import AgenticRAGConfig
from memory_eval.eval_core.agentic_rag.prompts import build_rerank_prompt
from memory_eval.eval_core.agentic_rag.retrieval import Candidate
from memory_eval.eval_core.agentic_rag.views import _call_llm_json


def rerank(
    candidates: list[Candidate],
    question: str,
    f_key: list[str],
    cfg: AgenticRAGConfig,
    *,
    top_k: int | None = None,
) -> list[Candidate]:
    """
    Listwise LLM 精排。

    参数:
        candidates: RRF 之后的 top-N 候选（默认 N = cfg.rerank_top_n = 30）
        question:   原问题
        f_key:      关键事实单元
        cfg:        AgenticRAGConfig
        top_k:      输出数量；None 则用 cfg.rerank_output_k

    返回:
        重排后的候选列表，长度 <= top_k；保留原 Candidate 元数据，meta 中追加
        "rerank_score" / "rerank_reason" 等字段。
    """
    if not candidates:
        return []
    take_n = min(cfg.rerank_top_n, len(candidates))
    pool = candidates[:take_n]
    out_k = top_k if top_k is not None else cfg.rerank_output_k

    cand_dicts: list[dict[str, Any]] = [
        {
            "id": c.id,
            "text": c.text,
            "score": c.score,
            "source_view": c.source_view,
        }
        for c in pool
    ]
    prompt = build_rerank_prompt(candidates=cand_dicts, question=question, f_key=f_key)

    raw = _call_llm_json(prompt, cfg)
    if not isinstance(raw, dict) or not isinstance(raw.get("ranked"), list):
        # LLM 失败 -> 直接按原 RRF 排序兜底
        return _fallback_sort_by_rrf(pool, out_k)

    # majority voting (cfg.enable_majority_voting) -> 在此可循环 3 次取中位
    # TODO: implement majority voting if cfg.enable_majority_voting

    by_id = {c.id: c for c in pool}
    ranked_items = raw["ranked"]
    ordered: list[Candidate] = []
    seen: set[str] = set()
    for item in ranked_items:
        cid = str(item.get("id", "")).strip()
        if not cid or cid not in by_id or cid in seen:
            continue
        seen.add(cid)
        c = by_id[cid]
        try:
            time_s = int(item.get("time", 0))
            speaker_s = int(item.get("speaker", 0))
            content_s = int(item.get("content", 0))
            total = int(item.get("total", time_s + speaker_s + content_s))
        except (TypeError, ValueError):
            time_s = speaker_s = content_s = total = 0
        new_meta = dict(c.meta)
        new_meta.update(
            {
                "rerank_score": total,
                "rerank_time": time_s,
                "rerank_speaker": speaker_s,
                "rerank_content": content_s,
                "rerank_reason": str(item.get("reason", "")),
            }
        )
        ordered.append(
            Candidate(
                id=c.id,
                text=c.text,
                score=float(total),
                source_view="rerank",
                rank_in_view=len(ordered),
                meta=new_meta,
            )
        )

    # 兜底：若 LLM 漏掉某些候选，按 RRF 顺序补齐
    if len(ordered) < out_k:
        for c in pool:
            if c.id not in seen and len(ordered) < out_k:
                new_meta = dict(c.meta)
                new_meta.setdefault("rerank_score", 0)
                new_meta.setdefault("rerank_reason", "fallback_rrf_order")
                ordered.append(
                    Candidate(
                        id=c.id,
                        text=c.text,
                        score=c.score,
                        source_view="rerank",
                        rank_in_view=len(ordered),
                        meta=new_meta,
                    )
                )

    return ordered[:out_k]


def _fallback_sort_by_rrf(candidates: list[Candidate], top_k: int) -> list[Candidate]:
    """LLM 失败时按 RRF 原顺序保留前 top_k 条。"""
    return list(candidates[:top_k])
