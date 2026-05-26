"""
CRAG 风格质量自评 — v3 增量 2。

在 RRF 之前对每路视图打 0-1 质量分，决定下游动作：
  - 全部 OK     -> PROCEED
  - 1-2 路低质  -> CROSS_VIEW_FALLBACK（高质视图反哺低质查询）
  - 全部低质    -> VIEW_RECONSTRUCT（LLM 重解析 F_key）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from memory_eval.eval_core.agentic_rag.config import AgenticRAGConfig
from memory_eval.eval_core.agentic_rag.prompts import build_quality_eval_prompt
from memory_eval.eval_core.agentic_rag.retrieval import Candidate
from memory_eval.eval_core.agentic_rag.views import _call_llm_json


class QualityAction(str, Enum):
    """CRAG 决策后续动作枚举。

    Returns:
        PROCEED:             所有视图质量达标，直接进入 RRF 融合。
        CROSS_VIEW_FALLBACK: 1-2 路低质，使用高质视图上下文反哺低质 query 后重检。
        VIEW_RECONSTRUCT:    全部低质，触发 LLM 重新解析 F_key 并重新生成视图。
    """

    PROCEED = "proceed"
    CROSS_VIEW_FALLBACK = "cross_view_fallback"
    VIEW_RECONSTRUCT = "view_reconstruct"


@dataclass(frozen=True)
class QualityScore:
    """单视图质量自评结果。"""

    view_type: str
    """{original, time, entity, topic, bm25}。"""

    score: float
    """LLM 输出的 0-1 分数；低于 tau_quality 视为低质。"""

    reason: str = ""
    """LLM 给出的简短理由，进入 diagnostics。"""

    top_supporting_ids: list[str] = field(default_factory=list)
    """LLM 标注的关键支撑候选 id，用于跨视图回退时反哺。"""


def evaluate_per_view(
    view_results: dict[str, list[Candidate]],
    question: str,
    f_key: list[str],
    cfg: AgenticRAGConfig,
) -> dict[str, QualityScore]:
    """
    对每路视图调用一次 LLM 打质量分。

    skeleton：串行调用以保持 prompt 行为可复现；
    TODO: 大规模评测时可并发，但需注意 OpenAI-compatible endpoint 限速。

    参数:
        view_results: parallel_retrieve 的输出
        question:     原问题
        f_key:        关键事实单元
        cfg:          AgenticRAGConfig

    返回:
        {view_type: QualityScore}
    """
    scores: dict[str, QualityScore] = {}
    for view_type, cands in view_results.items():
        if not cands:
            scores[view_type] = QualityScore(view_type=view_type, score=0.0, reason="empty candidates")
            continue
        # 把 Candidate 转成 LLM prompt 所需的 dict 列表
        cand_dicts = [{"id": c.id, "text": c.text} for c in cands[:10]]
        prompt = build_quality_eval_prompt(
            view_type=view_type,
            candidates=cand_dicts,
            question=question,
            f_key=f_key,
        )
        raw = _call_llm_json(prompt, cfg)
        if isinstance(raw, dict):
            try:
                s = float(raw.get("score", 0.0))
            except (TypeError, ValueError):
                s = 0.0
            scores[view_type] = QualityScore(
                view_type=view_type,
                score=max(0.0, min(1.0, s)),
                reason=str(raw.get("reason", "")),
                top_supporting_ids=[str(x) for x in raw.get("top_supporting_ids", []) if str(x).strip()],
            )
        else:
            # LLM 失败 -> 给一个中性分，避免错杀
            scores[view_type] = QualityScore(view_type=view_type, score=0.5, reason="llm_failure")
    return scores


def decide_action(
    scores: dict[str, QualityScore],
    *,
    tau_quality: float = 0.7,
) -> QualityAction:
    """
    根据每视图分数决定 PROCEED / CROSS_VIEW_FALLBACK / VIEW_RECONSTRUCT。

    决策规则（与设计文档 §三增量 2 对齐）：
      - 所有 q_i >= tau_quality      -> PROCEED
      - 1-2 路 q_i < tau_quality     -> CROSS_VIEW_FALLBACK
      - >=3 路或全部 q_i < tau_quality -> VIEW_RECONSTRUCT
    """
    if not scores:
        return QualityAction.PROCEED
    low_count = sum(1 for s in scores.values() if s.score < tau_quality)
    if low_count == 0:
        return QualityAction.PROCEED
    if low_count >= max(3, len(scores)):
        return QualityAction.VIEW_RECONSTRUCT
    return QualityAction.CROSS_VIEW_FALLBACK


def pick_donor_views(
    scores: dict[str, QualityScore],
    *,
    tau_quality: float = 0.7,
) -> tuple[list[str], list[str]]:
    """
    挑选高质视图（donor）与低质视图（recipient），用于跨视图回退。

    返回:
        (high_quality_view_types, low_quality_view_types)

    TODO: implement adaptive donor weighting; skeleton 仅按阈值划分。
    """
    high = [vt for vt, s in scores.items() if s.score >= tau_quality]
    low = [vt for vt, s in scores.items() if s.score < tau_quality]
    return high, low


def build_cross_view_query(
    recipient_view: str,
    donor_candidates: list[Candidate],
    original_query: str,
) -> str:
    """
    跨视图回退：用高质视图的候选片段反哺低质视图的查询。

    skeleton 用简单拼接策略；正式实现可叫一次 LLM 做精细融合。

    TODO: replace with LLM-based query rewrite using donor context.
    """
    if not donor_candidates:
        return original_query
    snippets = " | ".join(c.text[:140] for c in donor_candidates[:3])
    return f"{original_query} (context: {snippets})"
