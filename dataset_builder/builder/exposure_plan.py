"""
阶段 4：暴露计划生成。

计算每 session 的 must_expose 列表，并调一次 LLM 为每个 must_expose 生成 tone_hint
+ 为整个 session 生成 sentiment + theme。

无变更 session 也要暴露 σ_t 中尚未暴露过的项（保证全部 var 在 10 session 内被暴露）。
"""
from __future__ import annotations

import json
import logging

from .config import DatasetConfig, persona_data_dir
from .llm_client import LLMClient
from .schemas import (
    ExposureItem,
    ExposurePlan,
    Persona,
    SessionExposurePlan,
    StateEvolution,
    StateSchema,
)

logger = logging.getLogger(__name__)


EXPOSURE_METADATA_PROMPT = """画像背景：{backstory}

session {session} 的当前状态：
{state_dict_text}

本 session 必须暴露的状态项（你为每项写一条 tone_hint）：
{must_expose_text}

本 session 事件背景：{event}
本 session 是否含状态变更：{has_change}

【任务】
1. 给每个必须暴露项写一条 tone_hint（10-30 字）：告诉用户模拟器该如何在对话中自然带出这个信息
2. 给整个 session 选一个情感基调：neutral / positive / anxious / frustrated / excited
3. 给整个 session 写一个 session_theme（5-15 字中文）：本 session 对话主题

【输出格式（严格 JSON）】
{{
  "tone_hints": [{{"var": "var_name1", "hint": "..."}}, ...],
  "sentiment": "...",
  "session_theme": "..."
}}

注意 tone_hints 数组长度必须等于必须暴露项的数量，且每项的 var 字段对应。
"""


def _format_state_dict(state: dict, schema: StateSchema) -> str:
    lines = []
    for var, val in state.items():
        try:
            display = schema.get(var).display_name
        except KeyError:
            display = var
        lines.append(f"- {display} ({var}): {val}")
    return "\n".join(lines)


def _format_must_expose(items: list[ExposureItem], schema: StateSchema) -> str:
    lines = []
    for it in items:
        try:
            display = schema.get(it.var).display_name
        except KeyError:
            display = it.var
        change_marker = "（本 session 变更项）" if it.is_state_change else ""
        lines.append(f"- {display} ({it.var}) = {it.value} {change_marker}")
    return "\n".join(lines)


def _collect_already_exposed_vars(plans_so_far: list[SessionExposurePlan]) -> set[str]:
    exposed = set()
    for plan in plans_so_far:
        for item in plan.must_expose:
            exposed.add(item.var)
    return exposed


def build_exposure_plan(
    persona: Persona,
    schema: StateSchema,
    evolution: StateEvolution,
    cfg: DatasetConfig,
    client: LLMClient,
    write_file: bool = True,
) -> ExposurePlan:
    """计算 10 session 的暴露计划。"""
    sessions: list[SessionExposurePlan] = []
    all_vars = [v.var_name for v in schema.variables]

    for t in range(1, cfg.scale.num_sessions + 1):  # session 1..10
        snapshot = evolution.trajectory[t]
        prev_snapshot = evolution.trajectory[t - 1]

        must_expose: list[ExposureItem] = []

        # 规则 1：本 session 状态变化的项必须暴露
        for var, new_val in snapshot.state.items():
            if prev_snapshot.state.get(var) != new_val:
                must_expose.append(ExposureItem(
                    var=var, value=str(new_val),
                    is_state_change=True, tone_hint="",
                ))

        # 规则 2：若本 session 没有变更，暴露尚未暴露过的 var
        if not must_expose:
            already_exposed = _collect_already_exposed_vars(sessions)
            unexposed_vars = [v for v in all_vars if v not in already_exposed]
            # 每 session 最多 2 项
            for var in unexposed_vars[:2]:
                must_expose.append(ExposureItem(
                    var=var, value=str(snapshot.state.get(var, "")),
                    is_state_change=False, tone_hint="",
                ))

        # 规则 3：兜底——若依然没有 must_expose（全部已暴露过），退回到强调一个核心 var
        if not must_expose:
            must_expose.append(ExposureItem(
                var=schema.main_core,
                value=str(snapshot.state[schema.main_core]),
                is_state_change=False,
                tone_hint="",
            ))

        # === 调 LLM 生成 tone_hint + sentiment + theme ===
        prompt = EXPOSURE_METADATA_PROMPT.format(
            backstory=persona.backstory,
            session=t,
            state_dict_text=_format_state_dict(snapshot.state, schema),
            must_expose_text=_format_must_expose(must_expose, schema),
            event=snapshot.event,
            has_change="是" if snapshot.changes_from_prev else "否",
        )

        try:
            response = client.chat_json(
                prompt,
                model=cfg.llm.builder_model,
                temperature=cfg.llm.builder_temperature,
                max_tokens=600,
            )
            # 把 tone_hints 装回 items
            tone_map: dict[str, str] = {}
            for entry in response.get("tone_hints", []):
                if isinstance(entry, dict) and "var" in entry and "hint" in entry:
                    tone_map[str(entry["var"])] = str(entry["hint"])
            for item in must_expose:
                item.tone_hint = tone_map.get(item.var, f"自然地在对话中带出 {item.var}={item.value}")
            sentiment = str(response.get("sentiment", "neutral"))
            if sentiment not in {"neutral", "positive", "anxious", "frustrated", "excited"}:
                sentiment = "neutral"
            theme = str(response.get("session_theme", f"session {t} 对话")).strip()
        except Exception as e:
            logger.warning(f"exposure metadata LLM failed at session {t}: {e}; using fallbacks")
            for item in must_expose:
                item.tone_hint = f"自然带出 {item.var}={item.value}"
            sentiment = "neutral"
            theme = f"session {t} 日常对话"

        sessions.append(SessionExposurePlan(
            session=t,
            must_expose=must_expose,
            expose_sentiment=sentiment,
            session_theme=theme,
        ))
        logger.info(
            f"persona {persona.persona_id} session {t}: must_expose={[i.var for i in must_expose]}, "
            f"theme={theme}"
        )

    plan = ExposurePlan(persona_id=persona.persona_id, sessions=sessions)

    if write_file:
        pdir = persona_data_dir(cfg, persona.persona_id)
        path = pdir / "exposure_plan.json"
        path.write_text(
            json.dumps(plan.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"wrote {path}")

    return plan


def load_exposure_plan(cfg: DatasetConfig, persona_id: str) -> ExposurePlan:
    pdir = persona_data_dir(cfg, persona_id)
    path = pdir / "exposure_plan.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ExposurePlan.from_dict(raw)


__all__ = [
    "build_exposure_plan",
    "load_exposure_plan",
]
