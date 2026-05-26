"""
F_key 覆盖度判定 — v2 创新点 2。

LLM 判断 F_key 中每个 fact-unit 是否被当前候选集合覆盖。
覆盖率只作为诊断指标与 gap-query 生成依据，不作为循环终止条件。
循环终止由 termination.py 的有界候选池与轮间收敛控制。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memory_eval.eval_core.agentic_rag.config import AgenticRAGConfig
from memory_eval.eval_core.agentic_rag.prompts import build_coverage_check_prompt
from memory_eval.eval_core.agentic_rag.retrieval import Candidate
from memory_eval.eval_core.agentic_rag.views import _call_llm_json


@dataclass(frozen=True)
class CoverageResult:
    """覆盖度判定结果。"""

    coverage_rate: float
    """0-1，已覆盖 fact-unit 数 / 总 fact-unit 数。"""

    covered_units: list[str] = field(default_factory=list)
    """被语义支撑的 fact-unit 列表。"""

    uncovered_units: list[str] = field(default_factory=list)
    """未被覆盖、需在下一轮重写中补充的 fact-unit 列表。"""

    evidence_by_unit: dict[str, list[str]] = field(default_factory=dict)
    """每个 fact-unit -> 支撑候选 id 列表（便于下游归因）。"""

    reasoning: str = ""
    """LLM 简短理由。"""


def check_coverage(
    candidates: list[Candidate],
    f_key: list[str],
    cfg: AgenticRAGConfig,
) -> CoverageResult:
    """
    调用一次 LLM 判定覆盖度。

    参数:
        candidates: 精排后的 top-K 候选
        f_key:      关键事实单元
        cfg:        AgenticRAGConfig

    返回:
        CoverageResult dataclass

    skeleton 行为：LLM 失败时回退到字符串包含的简单规则估算，
    避免 retrieve 整体崩溃；正式实验应启用严格 LLM 模式。
    TODO: align with EvalMem strict-mode flag (cfg.require_llm_judgement).
    """
    if not f_key:
        return CoverageResult(coverage_rate=0.0, reasoning="empty f_key")

    if not candidates:
        return CoverageResult(
            coverage_rate=0.0,
            uncovered_units=list(f_key),
            reasoning="empty candidates",
        )

    cand_dicts = [{"id": c.id, "text": c.text} for c in candidates[:30]]
    prompt = build_coverage_check_prompt(candidates=cand_dicts, f_key=f_key)
    raw = _call_llm_json(prompt, cfg)

    if isinstance(raw, dict):
        try:
            rate = float(raw.get("coverage_rate", 0.0))
        except (TypeError, ValueError):
            rate = 0.0
        covered = [str(u).strip() for u in raw.get("covered_units", []) if str(u).strip()]
        uncovered = [str(u).strip() for u in raw.get("uncovered_units", []) if str(u).strip()]
        evidence = raw.get("evidence_by_unit", {})
        if not isinstance(evidence, dict):
            evidence = {}
        evidence_clean = {
            str(k): [str(x) for x in (v if isinstance(v, list) else [])]
            for k, v in evidence.items()
        }
        # 一致性检查：若 LLM 没给 rate 但给了 covered/uncovered
        if rate == 0.0 and (covered or uncovered):
            total = len(covered) + len(uncovered)
            rate = len(covered) / total if total else 0.0
        return CoverageResult(
            coverage_rate=max(0.0, min(1.0, rate)),
            covered_units=covered,
            uncovered_units=uncovered,
            evidence_by_unit=evidence_clean,
            reasoning=str(raw.get("reasoning", "")),
        )

    # 规则兜底：包含匹配
    return _rule_based_coverage(candidates, f_key)


def _rule_based_coverage(
    candidates: list[Candidate],
    f_key: list[str],
) -> CoverageResult:
    """LLM 失败时的规则兜底。skeleton 阶段只做简单 lowercase 包含匹配。"""
    covered: list[str] = []
    uncovered: list[str] = []
    evidence: dict[str, list[str]] = {}
    for unit in f_key:
        unit_norm = str(unit).lower()
        hits = [c.id for c in candidates if unit_norm and unit_norm in c.text.lower()]
        if hits:
            covered.append(unit)
            evidence[unit] = hits[:5]
        else:
            uncovered.append(unit)
    total = len(f_key)
    rate = len(covered) / total if total else 0.0
    return CoverageResult(
        coverage_rate=rate,
        covered_units=covered,
        uncovered_units=uncovered,
        evidence_by_unit=evidence,
        reasoning="rule_fallback_substring",
    )


def is_satisfied(result: CoverageResult, cfg: AgenticRAGConfig) -> bool:
    """Deprecated diagnostic helper; retrieval loops should not stop by coverage."""
    return result.coverage_rate >= cfg.tau_cover
