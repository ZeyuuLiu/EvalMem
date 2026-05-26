"""
AgenticRAG v3 全部 LLM 调用使用的 prompt 模板。

风格与 `memory_eval.eval_core.prompts` 对齐：
- 严格 JSON 输出（无 Markdown 包裹、无解释性前后文）
- 字段含义中文显式定义
- 输入数据用 ensure_ascii=False 序列化，保证中英混合 LoCoMo 不被转义
"""

from __future__ import annotations

import json
from typing import Any


def _dump(obj: Any) -> str:
    """ensure_ascii=False 的紧凑 JSON 序列化，与 EvalMem 现有 prompts.py 对齐。"""
    return json.dumps(obj, ensure_ascii=False)


def _json_only_notice() -> str:
    """所有 prompt 末尾的严格 JSON 提示。复用 EvalMem prompts.py 的措辞。"""
    return (
        "输出要求：\n"
        "1. 只输出一个纯文本 JSON 对象。\n"
        "2. 不要输出 ```json 或任何 Markdown 包裹。\n"
        "3. 不要在 JSON 前后输出解释性文字。\n"
    )


# ---------------------------------------------------------------------------
# 增量 3：Plan-RAG 复杂度分析
# ---------------------------------------------------------------------------


def build_complexity_analysis_prompt(f_key: list[str]) -> str:
    """
    分析 F_key 的复杂度，决定 v3 的自适应视图数 N。

    输出 schema:
      {"atomic_fact_count": int,
       "anchors": {"time": bool, "speaker": bool, "topic": bool, "location": bool},
       "cross_anchor_relations": int,
       "recommended_view_count": int (2-7),
       "recommended_view_types": [str],   # 子集 of {original, time, entity, topic, bm25}
       "confidence": float (0-1),
       "reasoning": str}
    """
    return (
        "你是 Plan-RAG 风格的查询规划器，需要根据 F_key 的复杂度建议多视图查询数量。\n"
        "F_key 是评测构造期写好的金线索（关键事实单元集合）。\n"
        "评估维度：\n"
        "1. atomic_fact_count：F_key 中可独立成立的原子事实数量。\n"
        "2. anchors：是否含 time / speaker / topic / location 锚点。\n"
        "3. cross_anchor_relations：跨锚点关系数（如 speaker↔time）。\n"
        "4. recommended_view_count ∈ [2, 7]：综合复杂度建议的视图数。\n"
        "   - 单事实简单 F_key  -> 2-3 视图\n"
        "   - 标配复合 F_key    -> 4-5 视图\n"
        "   - 多事实长尾 F_key  -> 6-7 视图\n"
        "5. recommended_view_types：从 {original, time, entity, topic, bm25} 中选出对应数量的视图类型；\n"
        "   缺 topic anchor 时用 entity 扩展替代 topic；bm25 始终作为最后一路（如启用）。\n"
        "6. confidence ∈ [0, 1]：判别置信度；< 0.7 调用方将回退固定 4 视图作兜底。\n"
        + _json_only_notice()
        + "请只输出符合下列 schema 的 JSON：\n"
        '{"atomic_fact_count":0,'
        '"anchors":{"time":false,"speaker":false,"topic":false,"location":false},'
        '"cross_anchor_relations":0,'
        '"recommended_view_count":4,'
        '"recommended_view_types":["original","time","entity","topic"],'
        '"confidence":0.0,'
        '"reasoning":"..."}\n'
        f"F_key: {_dump(f_key)}\n"
    )


# ---------------------------------------------------------------------------
# 创新点 1：F_key 多视图查询生成
# ---------------------------------------------------------------------------


def build_view_generation_prompt(
    question: str,
    f_key: list[str],
    view_plan: dict[str, Any],
) -> str:
    """
    根据视图规划，把 Q + F_key 改写为多视图查询字符串。

    输入 view_plan 形如 {"view_count": int, "view_types": [str]}。

    输出 schema:
      {"views": [{"view_type": "original|time|entity|topic|bm25",
                  "query": "...",
                  "weight": 1.0}],
       "reasoning": "..."}
    """
    return (
        "你是 EvalMem 编码探针的多视图查询生成器（v3 创新点 1）。\n"
        "目标：基于 Q + F_key 输出多个互补的查询字符串，最大化高召回（宁滥勿缺）。\n"
        "视图语义：\n"
        "- original: 直接复用原问题 Q。\n"
        "- time:     以 F_key 中的时间锚点为主，可重写为多种归一化形式（YYYY-MM-DD / 自然语言）。\n"
        "- entity:   以 F_key 中的实体（人名/物体/概念）为主，可加同义/扩展。\n"
        "- topic:    以 F_key 内容上位主题为主，便于跨表述召回。\n"
        "- bm25:     输出便于 BM25 倒排的精确 token（不要长句子，给一串关键 token）。\n"
        "原则：\n"
        "1. 每条 query 必须独立可用作下游 retrieve 的输入。\n"
        "2. weight 默认 1.0，可在 0.5-1.5 之间按重要性微调（time/speaker 锚点强可给到 1.2）。\n"
        "3. 数量必须严格等于 view_plan.view_count。\n"
        + _json_only_notice()
        + "请只输出符合下列 schema 的 JSON：\n"
        '{"views":[{"view_type":"original","query":"...","weight":1.0}],'
        '"reasoning":"..."}\n'
        f"Question: {question}\n"
        f"F_key: {_dump(f_key)}\n"
        f"ViewPlan: {_dump(view_plan)}\n"
    )


def build_gap_query_prompt(uncovered_facts: list[str]) -> str:
    """
    覆盖度未达 tau_cover 时，针对未覆盖 fact-units 生成补充查询。

    输出 schema:
      {"gap_query": "...", "view_type": "entity|time|topic|original", "reasoning": "..."}
    """
    return (
        "你是 EvalMem 编码探针的缺口查询生成器（v2 创新点 2 闭环的一部分）。\n"
        "上一轮多视图召回后，下列 fact-units 仍未被任何候选支撑。\n"
        "请生成一条新的查询，专门针对这些缺口；可选择把焦点放在最缺的锚点上。\n"
        + _json_only_notice()
        + '请只输出符合下列 schema 的 JSON：\n'
        '{"gap_query":"...","view_type":"entity","reasoning":"..."}\n'
        f"UncoveredFacts: {_dump(uncovered_facts)}\n"
    )


# ---------------------------------------------------------------------------
# 增量 2：CRAG 质量自评
# ---------------------------------------------------------------------------


def build_quality_eval_prompt(
    view_type: str,
    candidates: list[dict[str, Any]],
    question: str,
    f_key: list[str],
) -> str:
    """
    对单视图召回的候选打 0-1 质量分。

    输出 schema:
      {"score": float (0-1), "reason": "...", "top_supporting_ids": [str]}
    """
    items = "\n".join(
        f"- id={c.get('id', '')} text={c.get('text', '')[:280]}"
        for c in candidates[:10]
    )
    return (
        "你是 CRAG 风格的检索质量评估器（v3 增量 2）。\n"
        "判定当前视图召回的候选**作为整体**有多大概率能支撑 F_key 中的原子事实。\n"
        "评分指引：\n"
        "- 0.0-0.3：噪声为主，几乎没有可用证据。\n"
        "- 0.4-0.6：相关但缺关键锚点（如时间/实体错位）。\n"
        "- 0.7-0.9：含强证据，能覆盖大部分 fact-units。\n"
        "- 0.9-1.0：几乎所有 fact-units 都有直接候选支持。\n"
        + _json_only_notice()
        + '请只输出符合下列 schema 的 JSON：\n'
        '{"score":0.0,"reason":"...","top_supporting_ids":[]}\n'
        f"ViewType: {view_type}\n"
        f"Question: {question}\n"
        f"F_key: {_dump(f_key)}\n"
        f"Candidates:\n{items}\n"
    )


# ---------------------------------------------------------------------------
# 增量 1：LLM-as-Reranker
# ---------------------------------------------------------------------------


def build_rerank_prompt(
    candidates: list[dict[str, Any]],
    question: str,
    f_key: list[str],
) -> str:
    """
    Listwise 精排：让 LLM 一次看完 N (默认 30) 个候选，输出 ranked id 列表。

    输出 schema:
      {"ranked": [{"id": str, "time": int, "speaker": int, "content": int, "total": int, "reason": str}],
       "top_ids": [str]}

    time / speaker / content 各 0-10。total = time + speaker + content。
    """
    items = "\n".join(
        f"- id={c.get('id', '')} score={c.get('score', 0):.4f} view={c.get('source_view', '')} text={c.get('text', '')[:320]}"
        for c in candidates
    )
    return (
        "你是 RankRAG 风格的 listwise reranker（v3 增量 1）。\n"
        "对下列候选按相对 F_key 的支撑度从高到低排序。\n"
        "三个评分轴（各 0-10 整数）：\n"
        "- time:    时间锚点匹配度（若 F_key 无时间，则该轴打 5）\n"
        "- speaker: 说话人/主语匹配度\n"
        "- content: 内容语义等价度（与 F_key 的最强 fact-unit 对齐）\n"
        "total = time + speaker + content；排序按 total 降序。\n"
        + _json_only_notice()
        + '请只输出符合下列 schema 的 JSON：\n'
        '{"ranked":[{"id":"...","time":0,"speaker":0,"content":0,"total":0,"reason":"..."}],'
        '"top_ids":[]}\n'
        f"Question: {question}\n"
        f"F_key: {_dump(f_key)}\n"
        f"Candidates:\n{items}\n"
    )


# ---------------------------------------------------------------------------
# 创新点 2：覆盖度判定
# ---------------------------------------------------------------------------


def build_coverage_check_prompt(
    candidates: list[dict[str, Any]],
    f_key: list[str],
) -> str:
    """
    LLM 判定 F_key 中每个 fact-unit 是否被候选集合覆盖。

    输出 schema:
      {"covered_units": [str], "uncovered_units": [str],
       "evidence_by_unit": {fact_str: [candidate_id]},
       "coverage_rate": float (0-1),
       "reasoning": str}
    """
    items = "\n".join(
        f"- id={c.get('id', '')} text={c.get('text', '')[:320]}"
        for c in candidates[:30]
    )
    return (
        "你是 EvalMem 编码探针的覆盖度判定器（v2 创新点 2）。\n"
        "对 F_key 中**每一个**事实单元，给出当前候选集合是否提供了语义支撑。\n"
        "判定原则：\n"
        "1. 允许摘要、改写、时间归一化、多条候选联合支撑。\n"
        "2. 同主题但实体/时间错位的不算覆盖。\n"
        "3. coverage_rate = len(covered_units) / len(f_key)。\n"
        + _json_only_notice()
        + '请只输出符合下列 schema 的 JSON：\n'
        '{"covered_units":[],"uncovered_units":[],'
        '"evidence_by_unit":{},'
        '"coverage_rate":0.0,'
        '"reasoning":"..."}\n'
        f"F_key: {_dump(f_key)}\n"
        f"Candidates:\n{items}\n"
    )
