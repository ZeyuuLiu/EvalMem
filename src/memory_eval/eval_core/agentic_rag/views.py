"""
F_key 多视图查询生成 — v2 创新点 1 + v3 增量 3。

包含：
- Plan-RAG 风格的 F_key 复杂度分析（增量 3，决定自适应视图数 N ∈ [2, 7]）
- 多视图查询生成（原 Q / 时间 / 实体 / 主题 / BM25 token）
- 覆盖度未达时的缺口查询生成
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from memory_eval.eval_core.agentic_rag.config import AgenticRAGConfig
from memory_eval.eval_core.agentic_rag.prompts import (
    build_complexity_analysis_prompt,
    build_gap_query_prompt,
    build_view_generation_prompt,
)

# 与 EvalMem llm_assist._chat_json 行为对齐
_VIEW_TYPE_VOCAB = {"original", "time", "entity", "topic", "bm25"}


@dataclass(frozen=True)
class ViewPlan:
    """Plan-RAG 复杂度分析输出。"""

    view_count: int
    """自适应视图数 N ∈ [view_count_min, view_count_max]。"""

    view_types: list[str]
    """对应数量的视图类型，必须为 _VIEW_TYPE_VOCAB 的子集。"""

    confidence: float = 1.0
    """LLM 判别置信度；< plan_confidence_threshold 时调用方回退兜底。"""

    reasoning: str = ""
    """LLM 给出的判别依据，供 diagnostics 留痕。"""

    anchors: dict[str, bool] = field(default_factory=dict)
    """time / speaker / topic / location 锚点存在性。"""


@dataclass(frozen=True)
class QueryView:
    """单个视图的查询配置。"""

    view_type: str
    """{original, time, entity, topic, bm25} 之一。"""

    query: str
    """实际下发的查询字符串。bm25 视图传 token 串。"""

    weight: float = 1.0
    """RRF 阶段的权重微调，默认 1.0。"""


def analyze_complexity(
    question: str,
    f_key: list[str],
    cfg: AgenticRAGConfig,
) -> ViewPlan:
    """
    Plan-RAG 复杂度分析（增量 3）。

    调用一次 LLM，解析 F_key 复杂度，输出 ViewPlan。
    判别置信度 < cfg.plan_confidence_threshold 时回退到固定 4 dense + 1 sparse。

    参数:
        question: 原问题字符串
        f_key: 关键事实单元列表
        cfg: AgenticRAGConfig 实例

    返回:
        ViewPlan dataclass
    """
    # 极端兜底：空 F_key 时退回单视图（原 Q）
    if not f_key:
        return ViewPlan(view_count=1, view_types=["original"], confidence=0.0, reasoning="empty f_key")

    prompt = build_complexity_analysis_prompt(f_key)
    raw = _call_llm_json(prompt, cfg)

    if not isinstance(raw, dict):
        # LLM 调用失败 -> 回退固定 4 dense + 1 sparse
        return _fallback_plan(cfg)

    try:
        view_count = int(raw.get("recommended_view_count", 4))
        view_types_raw = raw.get("recommended_view_types", [])
        confidence = float(raw.get("confidence", 0.0))
        reasoning = str(raw.get("reasoning", ""))
        anchors = raw.get("anchors", {}) if isinstance(raw.get("anchors"), dict) else {}
    except (TypeError, ValueError):
        return _fallback_plan(cfg)

    # 校验合法性
    view_count = max(cfg.view_count_min, min(cfg.view_count_max, view_count))
    view_types = [str(v).strip().lower() for v in view_types_raw if str(v).strip().lower() in _VIEW_TYPE_VOCAB]
    if not view_types:
        view_types = ["original", "time", "entity", "topic"]
    if cfg.enable_bm25 and "bm25" not in view_types:
        view_types.append("bm25")

    # 数量对齐：截断或补齐
    if len(view_types) > view_count:
        view_types = view_types[:view_count]
    while len(view_types) < view_count:
        for fallback in ("original", "time", "entity", "topic", "bm25"):
            if fallback not in view_types and (fallback != "bm25" or cfg.enable_bm25):
                view_types.append(fallback)
                break
        else:
            break  # 词表用尽

    # 置信度低 -> 回退兜底
    if confidence < cfg.plan_confidence_threshold:
        return _fallback_plan(cfg, reasoning=f"low confidence {confidence:.2f}; fallback")

    return ViewPlan(
        view_count=len(view_types),
        view_types=view_types,
        confidence=confidence,
        reasoning=reasoning,
        anchors={str(k): bool(v) for k, v in anchors.items()},
    )


def generate_views(
    question: str,
    f_key: list[str],
    plan: ViewPlan,
    cfg: AgenticRAGConfig,
) -> list[QueryView]:
    """
    根据 ViewPlan 生成多视图查询字符串（创新点 1）。

    调用一次 LLM 一次性产出全部视图，避免 N 次 prompt 拆分调用。
    LLM 失败时回退到规则：
      original -> question
      time/entity/topic -> 简单 join f_key
      bm25 -> 抽取 f_key 中的关键 token
    """
    plan_dict = {"view_count": plan.view_count, "view_types": list(plan.view_types)}
    prompt = build_view_generation_prompt(question=question, f_key=f_key, view_plan=plan_dict)
    raw = _call_llm_json(prompt, cfg)

    if not isinstance(raw, dict) or not isinstance(raw.get("views"), list):
        return _fallback_generate_views(question, f_key, plan)

    views: list[QueryView] = []
    for item in raw["views"][: plan.view_count]:
        try:
            view_type = str(item.get("view_type", "original")).strip().lower()
            query = str(item.get("query", "")).strip()
            weight = float(item.get("weight", 1.0))
        except (TypeError, ValueError):
            continue
        if view_type not in _VIEW_TYPE_VOCAB or not query:
            continue
        views.append(QueryView(view_type=view_type, query=query, weight=weight))

    if not views:
        return _fallback_generate_views(question, f_key, plan)

    # 补齐缺失视图（按 plan 指定的类型，确保数量一致）
    seen_types = {v.view_type for v in views}
    for vt in plan.view_types:
        if vt not in seen_types:
            views.append(_default_view_for_type(vt, question, f_key))
            seen_types.add(vt)
    return views[: plan.view_count]


def generate_gap_query(uncovered_facts: list[str], cfg: AgenticRAGConfig) -> QueryView:
    """
    覆盖度未达 tau_cover 时，生成补充查询用于下一轮检索（创新点 2 闭环）。
    """
    if not uncovered_facts:
        return QueryView(view_type="original", query="", weight=1.0)

    prompt = build_gap_query_prompt(uncovered_facts)
    raw = _call_llm_json(prompt, cfg)

    if isinstance(raw, dict) and isinstance(raw.get("gap_query"), str) and raw["gap_query"].strip():
        view_type = str(raw.get("view_type", "entity")).strip().lower()
        if view_type not in _VIEW_TYPE_VOCAB:
            view_type = "entity"
        return QueryView(view_type=view_type, query=raw["gap_query"].strip(), weight=1.2)

    # 兜底：把未覆盖事实串成一条 entity 查询
    return QueryView(view_type="entity", query=" ; ".join(uncovered_facts[:5]), weight=1.2)


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _fallback_plan(cfg: AgenticRAGConfig, reasoning: str = "llm_failure") -> ViewPlan:
    """LLM 不可用或置信度过低时的固定 4 dense + 1 sparse 兜底。"""
    base = ["original", "time", "entity", "topic"]
    if cfg.enable_bm25:
        base.append("bm25")
    return ViewPlan(view_count=len(base), view_types=base, confidence=0.0, reasoning=reasoning)


def _fallback_generate_views(question: str, f_key: list[str], plan: ViewPlan) -> list[QueryView]:
    """LLM 视图生成失败时的规则兜底。"""
    return [_default_view_for_type(vt, question, f_key) for vt in plan.view_types]


def _default_view_for_type(view_type: str, question: str, f_key: list[str]) -> QueryView:
    """每个视图类型的规则模板。"""
    if view_type == "original":
        return QueryView(view_type="original", query=question, weight=1.0)
    if view_type == "time":
        # TODO: implement smarter time anchor extraction
        return QueryView(view_type="time", query=" ; ".join(f_key[:3]), weight=1.0)
    if view_type == "entity":
        return QueryView(view_type="entity", query=" ; ".join(f_key[:5]), weight=1.0)
    if view_type == "topic":
        return QueryView(view_type="topic", query=question + " " + " ".join(f_key[:3]), weight=1.0)
    if view_type == "bm25":
        # 简单 token 抽取：拆 f_key 上的空白
        tokens = []
        for fact in f_key:
            tokens.extend(str(fact).split())
        return QueryView(view_type="bm25", query=" ".join(tokens[:30]), weight=1.0)
    return QueryView(view_type="original", query=question, weight=1.0)


def _call_llm_json(prompt: str, cfg: AgenticRAGConfig) -> dict[str, Any] | None:
    """
    最小可用的 OpenAI-兼容 chat 调用，输出 JSON 对象。
    复用 llm_assist._chat_json 思路但本地实现，避免循环依赖。

    TODO: implement caching + majority voting per cfg.enable_majority_voting
    """
    if not cfg.llm_api_key or not cfg.llm_base_url:
        return None
    payload = {
        "model": cfg.llm_model,
        "temperature": cfg.llm_temperature,
        "messages": [
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        url=f"{cfg.llm_base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg.llm_api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg.llm_timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
        obj = json.loads(raw)
        content = str(obj["choices"][0]["message"]["content"]).strip()
        return _extract_json_object(content)
    except Exception:
        return None


def _extract_json_object(text: str) -> dict[str, Any]:
    """复用 EvalMem 的 JSON 抽取容错。"""
    content = (text or "").strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    try:
        return json.loads(content)
    except Exception:
        pass
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        return json.loads(content[start : end + 1])
    raise ValueError("no JSON object in LLM response")
