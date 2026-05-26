from __future__ import annotations

"""
MemWiki v4 LLM 客户端封装。

把 builder / composer / retriever 共用的 LLM 调用抽到这里，避免每个模块
都重复封装 retries / negative-hint / JSON 解析逻辑。

设计原则（与 :mod:`memory_eval.eval_core.llm_assist` 一致）：
- 所有公开函数返回 ``dict | None``；
- 失败时不抛异常（让 caller 自行决定 L2 / L3 兜底）；
- ``llm_cfg=None`` 时进入 dry-run（返回 None），供单测使用。
"""

import json
import time
from typing import Any

import urllib.request

from memory_eval.memwiki.prompts import (
    build_query_parse_prompt,
    build_wikify_entity_prompt,
    build_wikify_event_prompt,
    build_wikify_source_prompt,
    build_wikify_topic_prompt,
)


def _cfg_value(llm_cfg: Any, *names: str, default: Any = "") -> Any:
    for name in names:
        value = getattr(llm_cfg, name, None)
        if value is not None and value != "":
            return value
    if isinstance(llm_cfg, dict):
        for name in names:
            value = llm_cfg.get(name)
            if value is not None and value != "":
                return value
    return default


def _extract_json_object(text: str) -> dict[str, Any]:
    content = str(text or "").strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    if content.lower().startswith("json"):
        content = content[4:].strip()
    try:
        obj = json.loads(content)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(content[start : end + 1])
        return obj if isinstance(obj, dict) else {}
    raise ValueError("no JSON object in LLM response")


def _chat_json(
    llm_cfg: Any | None,
    prompt: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    retries: int = 2,
) -> dict[str, Any] | None:
    if llm_cfg is None:
        return None
    api_key = str(_cfg_value(llm_cfg, "api_key", default="")).strip()
    base_url = str(_cfg_value(llm_cfg, "base_url", default="")).strip()
    model = str(_cfg_value(llm_cfg, "model", "llm_model", default="gpt-4o-mini")).strip()
    temp = float(temperature if temperature is not None else _cfg_value(llm_cfg, "temperature", "llm_temperature", default=0.0))
    timeout = int(_cfg_value(llm_cfg, "timeout_sec", "llm_timeout_sec", default=120) or 120)
    if not api_key or not base_url:
        return None

    payload: dict[str, Any] = {
        "model": model,
        "temperature": temp,
        "messages": [
            {"role": "system", "content": "Return strict JSON only. Do not include markdown fences."},
            {"role": "user", "content": prompt},
        ],
    }
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)

    last_exc: Exception | None = None
    for attempt in range(max(1, int(retries or 1))):
        req = urllib.request.Request(
            url=f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
            content = str(obj["choices"][0]["message"]["content"]).strip()
            return _extract_json_object(content)
        except Exception as exc:
            last_exc = exc
            if attempt < max(1, int(retries or 1)) - 1:
                time.sleep(0.5 * (attempt + 1))
    return None

def call_wikify(
    text: str,
    *,
    llm_cfg: Any | None,
    temperature: float,
    max_tokens: int,
    negative_hint: bool = False,
    num_questions: int = 4,
) -> dict | None:
    """
    Wikify source 页 LLM 调用。

    """
    prompt = build_wikify_source_prompt(text, num_questions=num_questions)
    if negative_hint:
        prompt += "\n上一轮输出不是合法 JSON 或不符合 schema。请修正并只返回 JSON。"
    raw = _chat_json(llm_cfg, prompt, temperature=temperature, max_tokens=max_tokens)
    return _validate_wikify_payload(raw)


def call_entity_synthesize(
    canonical: str,
    related_source_texts: list[str],
    *,
    llm_cfg: Any | None,
) -> dict | None:
    """Entity 综合页 LLM 调用。"""
    raw = _chat_json(
        llm_cfg,
        build_wikify_entity_prompt(canonical, related_source_texts),
        temperature=0.0,
        max_tokens=1200,
    )
    return _validate_synthesis_payload(raw)


def call_topic_synthesize(
    topic: str,
    related_source_texts: list[str],
    *,
    llm_cfg: Any | None,
) -> dict | None:
    """Topic 综合页 LLM 调用。"""
    raw = _chat_json(
        llm_cfg,
        build_wikify_topic_prompt(topic, related_source_texts),
        temperature=0.0,
        max_tokens=1200,
    )
    return _validate_synthesis_payload(raw)


def call_event_synthesize(
    event_clue: str,
    related_source_texts: list[str],
    *,
    llm_cfg: Any | None,
) -> dict | None:
    """Event 综合页 LLM 调用。"""
    raw = _chat_json(
        llm_cfg,
        build_wikify_event_prompt(event_clue, related_source_texts),
        temperature=0.0,
        max_tokens=1200,
    )
    return _validate_synthesis_payload(raw)


def call_compose_decision(
    old_page_content: str,
    new_record: dict,
    *,
    llm_cfg: Any | None,
) -> dict | None:
    """Composer rewrite/append/warn/no_change 决策。"""
    prompt = (
        "You maintain a MemWiki page. Decide how to merge a new memory record.\n"
        "Return JSON only: {\"action\":\"rewrite|append|warn|no_change\",\"reason\":\"...\","
        "\"rewritten_content\":\"\",\"warning_text\":\"\",\"new_version_content\":\"\"}\n"
        f"Old page content:\n{old_page_content}\n\n"
        f"New record:\n{json.dumps(new_record, ensure_ascii=False)}\n"
    )
    raw = _chat_json(llm_cfg, prompt, max_tokens=1200)
    if not isinstance(raw, dict):
        return None
    action = str(raw.get("action", "")).strip()
    if action not in {"rewrite", "append", "warn", "no_change"}:
        return None
    return raw


def call_state_change_detection(
    old_page_facts: list[dict],
    new_record: dict,
    *,
    llm_cfg: Any | None,
) -> dict | None:
    """SPO 冲突 LLM 二次确认（区分状态变更 vs 并列偏好）。"""
    prompt = (
        "Decide whether the new record changes an old state rather than adding a parallel preference.\n"
        "Return JSON only: {\"is_state_change\":true|false,\"changed_subject\":\"\","
        "\"changed_predicate\":\"\",\"old_object\":\"\",\"new_object\":\"\",\"reason\":\"\"}\n"
        f"Old atomic facts:\n{json.dumps(old_page_facts, ensure_ascii=False)}\n\n"
        f"New record:\n{json.dumps(new_record, ensure_ascii=False)}\n"
    )
    raw = _chat_json(llm_cfg, prompt, max_tokens=800)
    return raw if isinstance(raw, dict) else None


def call_query_parse(
    query: str,
    *,
    llm_cfg: Any | None,
) -> dict | None:
    """检索期把 NL 查询解析为 (entities, time_refs, topic_terms)。"""
    raw = _chat_json(llm_cfg, build_query_parse_prompt(query), max_tokens=500)
    return raw if isinstance(raw, dict) else None


def call_contradiction_check(
    page_a_content: str,
    page_b_content: str,
    *,
    llm_cfg: Any | None,
) -> dict | None:
    """跨页矛盾检测（lint 用）。"""
    prompt = (
        "Check whether two MemWiki page contents contain an unmarked contradiction.\n"
        "Return JSON only: {\"has_contradiction\":true|false,\"subject\":\"\","
        "\"predicate\":\"\",\"a_object\":\"\",\"b_object\":\"\",\"reason\":\"\"}\n"
        f"Page A:\n{page_a_content}\n\nPage B:\n{page_b_content}\n"
    )
    raw = _chat_json(llm_cfg, prompt, max_tokens=800)
    return raw if isinstance(raw, dict) else None


def _validate_wikify_payload(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    tags = raw.get("tags", {})
    if not isinstance(tags, dict):
        tags = {}
    facts = raw.get("atomic_facts", [])
    hqs = raw.get("hypothetical_questions", [])
    times = raw.get("time_anchors", [])
    return {
        "title": str(raw.get("title", "")).strip(),
        "tags": {
            "entities": _as_str_list(tags.get("entities", [])),
            "topics": _as_str_list(tags.get("topics", [])),
        },
        "time_anchors": [x for x in times if isinstance(x, dict)] if isinstance(times, list) else [],
        "atomic_facts": [x for x in facts if isinstance(x, dict)] if isinstance(facts, list) else [],
        "hypothetical_questions": _as_str_list(hqs),
    }


def _validate_synthesis_payload(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    tags = raw.get("tags", {})
    if not isinstance(tags, dict):
        tags = {}
    facts = raw.get("atomic_facts", [])
    hqs = raw.get("hypothetical_questions", [])
    return {
        **dict(raw),
        "title": str(raw.get("title", "")).strip(),
        "summary": str(raw.get("summary", "")).strip(),
        "current_state": str(raw.get("current_state", "")).strip(),
        "tags": {
            "entities": _as_str_list(tags.get("entities", [])),
            "topics": _as_str_list(tags.get("topics", [])),
        },
        "atomic_facts": [x for x in facts if isinstance(x, dict)] if isinstance(facts, list) else [],
        "hypothetical_questions": _as_str_list(hqs),
    }


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


__all__ = [
    "call_wikify",
    "call_entity_synthesize",
    "call_topic_synthesize",
    "call_event_synthesize",
    "call_compose_decision",
    "call_state_change_detection",
    "call_query_parse",
    "call_contradiction_check",
]
