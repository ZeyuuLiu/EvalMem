from __future__ import annotations

"""
MemWiki v4 所有 LLM 提示词构建函数。

设计原则（与 ``memory_eval.eval_core.prompts`` 对齐）：
- 每个 prompt site 提供一个 ``build_*_prompt`` 函数
- 返回纯字符串，调用端负责包装到 chat/completions 的 messages
- 严格 JSON 输出协议：禁用 markdown 包裹、禁用解释性前缀，与
  :func:`memory_eval.eval_core.llm_assist._extract_json_object` 配合
- 全部使用中文说明（与 eval_core/prompts.py 一致），便于审稿人审阅
"""

import json
from typing import Any

from memory_eval.memwiki.schema import WikiEntry


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _json_only_notice() -> str:
    return (
        "输出要求：\n"
        "1. 只输出一个纯文本 JSON 对象，不要包裹 ```json 等 markdown。\n"
        "2. JSON 前后不得添加解释性文字。\n"
        "3. 字段缺失时使用空数组或空字符串，不要省略 key。\n"
    )


# ---------------------------------------------------------------------------
# Wikify 提示词（5 类页面分别构建）
# ---------------------------------------------------------------------------


def build_wikify_source_prompt(record_text: str, num_questions: int = 4) -> str:
    """source 页改写：从单条 record 抽出 schema 五元组 + 假设问题。"""
    return (
        "你是 MemWiki 的 Source 页面构建者。\n"
        "目标：把单条原始对话记录改写为结构化 wiki source 页。\n"
        "约束：\n"
        "- atomic_facts 使用 (subject, predicate, object, time) 四元组；\n"
        "- hypothetical_questions 必须覆盖所有 atomic_facts，且彼此表层不重复（覆盖度/颗粒度/表层差异三轴控制，详见 v3 优化 2）；\n"
        "- topics 只能从受控词表中挑选，不在词表时输出 \"other\"；\n"
        "- 实体名使用规范全名，不要使用代词。\n"
        + _json_only_notice()
        + "请只输出符合下列 schema 的 JSON：\n"
        "{\"title\":\"...\","
        "\"tags\":{\"entities\":[],\"topics\":[]},"
        "\"time_anchors\":[{\"raw\":\"...\",\"iso\":\"...\",\"session\":null,\"certainty\":\"high|medium|fuzzy\"}],"
        "\"atomic_facts\":[{\"subject\":\"...\",\"predicate\":\"...\",\"object\":\"...\",\"time\":\"...\"}],"
        f"\"hypothetical_questions\":[/* {num_questions} 个 */]"
        "}\n"
        f"Record:\n\"\"\"\n{record_text}\n\"\"\"\n"
    )


def build_wikify_entity_prompt(entity: str, related_sources: list[str]) -> str:
    """entity 页综合：跨多 source 抽出该实体的稳定属性与状态。"""
    joined = "\n".join(f"- {s}" for s in related_sources[:30])
    return (
        "你是 MemWiki 的 Entity 页面构建者。\n"
        f"目标：综合下列 source 片段，为实体「{entity}」生成一个稳定的综合页。\n"
        "约束：\n"
        "- 在 atomic_facts 中聚合实体的稳定属性（职业、关系、地点等）；\n"
        "- current_state 描述实体在最新 session 的状态（如果各 source 间有冲突，仅记录最新者，旧值由 versioning 模块保留）；\n"
        "- hypothetical_questions 应是用户可能围绕该实体提出的 4-6 个问题；\n"
        + _json_only_notice()
        + "请只输出符合下列 schema 的 JSON：\n"
        "{\"title\":\"...\",\"summary\":\"...\",\"current_state\":\"...\","
        "\"atomic_facts\":[],\"hypothetical_questions\":[],\"tags\":{\"entities\":[],\"topics\":[]}}\n"
        f"Related sources:\n{joined}\n"
    )


def build_wikify_topic_prompt(topic: str, related_sources: list[str]) -> str:
    """topic 页综合：围绕受控主题词跨多 source 抽出演化轨迹。"""
    joined = "\n".join(f"- {s}" for s in related_sources[:30])
    return (
        "你是 MemWiki 的 Topic 页面构建者。\n"
        f"目标：综合下列 source 片段，为主题「{topic}」生成一个演化轨迹页。\n"
        "约束：\n"
        "- evolution 字段按 session_id 升序列出该主题的关键节点；\n"
        "- 不得引入受控词表外的 topics；\n"
        + _json_only_notice()
        + "请只输出符合下列 schema 的 JSON：\n"
        "{\"title\":\"...\",\"summary\":\"...\",\"evolution\":[{\"session\":1,\"event\":\"...\"}],"
        "\"atomic_facts\":[],\"hypothetical_questions\":[]}\n"
        f"Related sources:\n{joined}\n"
    )


def build_wikify_event_prompt(event_clue: str, related_sources: list[str]) -> str:
    """event 页综合：基于触发线索 + 证据 source 抽出事件结构。"""
    joined = "\n".join(f"- {s}" for s in related_sources[:30])
    return (
        "你是 MemWiki 的 Event 页面构建者。\n"
        "目标：基于下列证据 source 抽出事件的发生时间、参与者、影响。\n"
        f"事件触发线索：{event_clue}\n"
        "约束：\n"
        "- 必须给出 occurred_at_session（最佳估计）；\n"
        "- participants 应使用规范实体名；\n"
        "- causes / effects 字段可为空字符串；不要编造未在证据中出现的因果。\n"
        + _json_only_notice()
        + "请只输出符合下列 schema 的 JSON：\n"
        "{\"title\":\"...\",\"summary\":\"...\",\"occurred_at_session\":null,"
        "\"participants\":[],\"causes\":\"\",\"effects\":\"\","
        "\"atomic_facts\":[],\"hypothetical_questions\":[]}\n"
        f"Evidence sources:\n{joined}\n"
    )


# ---------------------------------------------------------------------------
# Composer 决策（v4 §三 核心）
# ---------------------------------------------------------------------------


def build_compose_decision_prompt(old_page: WikiEntry, new_record: dict) -> str:
    """
    复合更新决策：让 LLM 在 rewrite / append / warn / no_change 中选择。
    输出 schema 与 v4 §3.3 对齐。
    """
    return (
        "你是 MemWiki 的页面维护者，负责【复合更新】决策。\n"
        "判断三件事：\n"
        "1. 新 record 是否带来 P 中尚未提及的关于该实体/主题/事件的新维度？\n"
        "2. 新 record 是否与 P 中已有信息矛盾？\n"
        "3. 新 record 是否是 P 中已有事实的更近期状态版本？\n"
        "据此从 action ∈ {rewrite, append, warn, no_change} 中选择：\n"
        "- rewrite：新信息融入 P，输出 rewritten_content 全文；\n"
        "- append：仅追加 source 引用，不改正文；\n"
        "- warn：标记矛盾并触发多版本机制（旧版本保留，新版本另起）；\n"
        "- no_change：新 record 不带来增量价值。\n"
        + _json_only_notice()
        + "请只输出符合下列 schema 的 JSON：\n"
        "{\"action\":\"rewrite|append|warn|no_change\",\"reason\":\"...\","
        "\"rewritten_content\":\"\",\"warning_text\":\"\",\"new_version_content\":\"\"}\n"
        f"Old page (entry_id={old_page.entry_id}, type={old_page.page_type}):\n"
        f"\"\"\"{old_page.current_content}\"\"\"\n"
        f"New record:\n{_dump(new_record)}\n"
    )


# ---------------------------------------------------------------------------
# 多版本：状态变更检测
# ---------------------------------------------------------------------------


def build_state_change_detection_prompt(page: WikiEntry, new_record: dict) -> str:
    """SPO 模式冲突 LLM 二次确认（v3 优化 4.3）。"""
    facts = [
        {"subject": f.subject, "predicate": f.predicate, "object": f.object, "time": f.time}
        for f in page.atomic_facts
    ]
    return (
        "你是 MemWiki 的状态变更检测器。\n"
        "判断 new_record 是否对 old_page 中某个 (subject, predicate) 给出了新的 object，"
        "且这种变更确实是【状态变化】（而非并列偏好）。\n"
        "示例：\n"
        "- (Caroline, lives_in, Shanghai) -> (Caroline, lives_in, Beijing) 是状态变更，is_state_change=true；\n"
        "- (Caroline, likes, coffee) -> (Caroline, likes, tea) 不是状态变更，可并列存在。\n"
        + _json_only_notice()
        + "请只输出符合下列 schema 的 JSON：\n"
        "{\"is_state_change\":true,\"changed_subject\":\"\",\"changed_predicate\":\"\","
        "\"old_object\":\"\",\"new_object\":\"\",\"reason\":\"\"}\n"
        f"Old page atomic_facts: {_dump(facts)}\n"
        f"New record: {_dump(new_record)}\n"
    )


# ---------------------------------------------------------------------------
# 检索：Query Parse
# ---------------------------------------------------------------------------


def build_query_parse_prompt(query: str) -> str:
    """NL 查询 → 结构化 (entities, time_refs, topic_terms)，用于倒排路径。"""
    return (
        "你是 MemWiki 的检索 Query 解析器。\n"
        "把自然语言查询解析为结构化字段：实体名列表、时间引用列表、主题词列表。\n"
        + _json_only_notice()
        + "请只输出符合下列 schema 的 JSON：\n"
        "{\"entities\":[],\"time_refs\":[{\"raw\":\"\",\"iso\":\"\",\"session\":null,\"certainty\":\"high|medium|fuzzy\"}],"
        "\"topic_terms\":[]}\n"
        f"Query: {query}\n"
    )


# ---------------------------------------------------------------------------
# 假设问题生成（v3 优化 2 的三轴控制）
# ---------------------------------------------------------------------------


def build_hypothetical_questions_prompt(record_text: str, num_questions: int = 4) -> str:
    """独立的假设问题生成（也可与 wikify_source 合并调用）。"""
    return (
        f"为下列对话记录生成 {num_questions} 个假设问题。\n"
        "约束：\n"
        "[Coverage] 覆盖 record 中所有 atomic_facts，每个 fact 至少一问；\n"
        "[Granularity] 混合粒度：1 个 broad 问题 + N-2 个 specific + 1 个 yes/no 验证型；\n"
        "[Surface variation] 4 个问题不重复同一关键名词，使用 what/when/how/is-it-true 等多种句型；\n"
        "[Answerability] 每个问题必须仅靠本 record 即可回答，拒绝 \"what did people talk about\" 这类泛问。\n"
        + _json_only_notice()
        + "请只输出 JSON：\n"
        "{\"questions\":[\"...\"]}\n"
        f"Record:\n\"\"\"\n{record_text}\n\"\"\"\n"
    )


# ---------------------------------------------------------------------------
# Lint：跨页矛盾检测
# ---------------------------------------------------------------------------


def build_contradiction_check_prompt(page_a: WikiEntry, page_b: WikiEntry) -> str:
    """检测两个页面是否存在未被 warning 标记的语义矛盾。"""
    return (
        "你是 MemWiki Lint 的跨页矛盾检测器。\n"
        "判断 page_a 与 page_b 是否存在 (subject, predicate, object) 模式上同 S+P 不同 O 且未被双方 warnings 标记的矛盾。\n"
        + _json_only_notice()
        + "请只输出 JSON：\n"
        "{\"has_contradiction\":false,\"subject\":\"\",\"predicate\":\"\",\"a_object\":\"\",\"b_object\":\"\",\"reason\":\"\"}\n"
        f"page_a (entry_id={page_a.entry_id}): \"\"\"{page_a.current_content}\"\"\"\n"
        f"page_b (entry_id={page_b.entry_id}): \"\"\"{page_b.current_content}\"\"\"\n"
    )


__all__ = [
    "build_wikify_source_prompt",
    "build_wikify_entity_prompt",
    "build_wikify_topic_prompt",
    "build_wikify_event_prompt",
    "build_compose_decision_prompt",
    "build_state_change_detection_prompt",
    "build_query_parse_prompt",
    "build_hypothetical_questions_prompt",
    "build_contradiction_check_prompt",
]
