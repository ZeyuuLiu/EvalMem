"""
阶段 7：Oracle context 生成。

两步走：
1. 为每 session 生成"理想暴露剧本"（"理想助手"参与的简短对话，把 must_expose 全部包含）
2. 对每条评测问题在每个询问 session，从理想剧本中切片出 oracle_context

oracle_context 仅服务于生成探针，不参与在线 on-policy 评测。
"""
from __future__ import annotations

import json
import logging

from .config import DatasetConfig, persona_data_dir
from .llm_client import LLMClient
from .schemas import (
    EvalQuestion,
    ExposurePlan,
    OracleContext,
    Persona,
    StateSchema,
    StateEvolution,
)

logger = logging.getLogger(__name__)


ORACLE_DIALOGUE_PROMPT = """你将为以下场景生成一份"理想"对话脚本（中文，user 与 assistant 的多轮交替）。

【画像】
{backstory}
姓名：{name}
沟通风格：{communication_style}

【本 session 状态】
{state_dict_text}

【本 session 必须包含的信息（请通过自然对话方式全部覆盖）】
{must_expose_text}

【session 主题】{session_theme}
【情感基调】{sentiment}

【生成要求】
1. 共生成 {n_turns} 轮对话（每轮含 user 一条 + assistant 一条）
2. user 视角是上述画像；assistant 是"理想"AI 助手（友善、准确、追问得当）
3. 必须把 must_expose 全部包含在 user 消息中（可以分多轮）
4. 不要使用"AI"、"评测"等元层词
5. user 的措辞要符合 communication_style

【输出格式（严格 JSON）】
{{
  "dialogue": [
    {{"user": "...", "assistant": "..."}},
    ...
  ]
}}
"""


def _format_state_dict(state: dict, schema: StateSchema) -> str:
    lines = []
    for var, val in state.items():
        try:
            display = schema.get(var).display_name
        except KeyError:
            display = var
        lines.append(f"- {display}: {val}")
    return "\n".join(lines)


def _format_must_expose(items, schema: StateSchema) -> str:
    lines = []
    for it in items:
        try:
            display = schema.get(it.var).display_name
        except KeyError:
            display = it.var
        lines.append(f"- {display}={it.value}")
    return "\n".join(lines)


def build_oracle_dialogues(
    persona: Persona,
    schema: StateSchema,
    evolution: StateEvolution,
    plan: ExposurePlan,
    cfg: DatasetConfig,
    client: LLMClient,
    n_turns_per_session: int = 8,
    write_file: bool = True,
) -> dict[int, list[dict]]:
    """每 session 生成一份理想剧本（list of {user, assistant}）。"""
    dialogues: dict[int, list[dict]] = {}

    for sess_plan in plan.sessions:
        t = sess_plan.session
        snapshot = evolution.trajectory[t]
        prompt = ORACLE_DIALOGUE_PROMPT.format(
            backstory=persona.backstory,
            name=persona.name,
            communication_style=persona.communication_style,
            state_dict_text=_format_state_dict(snapshot.state, schema),
            must_expose_text=_format_must_expose(sess_plan.must_expose, schema),
            session_theme=sess_plan.session_theme,
            sentiment=sess_plan.expose_sentiment,
            n_turns=n_turns_per_session,
        )
        try:
            response = client.chat_json(
                prompt,
                model=cfg.llm.oracle_dialogue_model,
                temperature=cfg.llm.builder_temperature,
                max_tokens=2000,
            )
            raw_dialogue = response.get("dialogue", [])
            cleaned = []
            for turn in raw_dialogue:
                if not isinstance(turn, dict):
                    continue
                u = str(turn.get("user", "")).strip()
                a = str(turn.get("assistant", "")).strip()
                if u and a:
                    cleaned.append({"user": u, "assistant": a})
            if len(cleaned) < 3:
                raise ValueError(f"too few turns ({len(cleaned)})")
            dialogues[t] = cleaned
        except Exception as e:
            logger.warning(f"oracle dialogue gen failed at session {t}: {e}; using minimal stub")
            stub = []
            for item in sess_plan.must_expose:
                try:
                    display = schema.get(item.var).display_name
                except KeyError:
                    display = item.var
                stub.append({
                    "user": f"我的{display}是{item.value}。",
                    "assistant": "好的，我记下了。",
                })
            dialogues[t] = stub

        logger.debug(f"  session {t} oracle dialogue: {len(dialogues[t])} turns")

    if write_file:
        pdir = persona_data_dir(cfg, persona.persona_id)
        path = pdir / "oracle_dialogues.json"
        out_dict = {str(s): dialogues[s] for s in sorted(dialogues.keys())}
        path.write_text(
            json.dumps(out_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"wrote {path}")

    return dialogues


def build_oracle_contexts(
    persona: Persona,
    eval_questions: list[EvalQuestion],
    oracle_dialogues: dict[int, list[dict]],
    cfg: DatasetConfig,
    write_file: bool = True,
) -> dict[str, dict[int, OracleContext]]:
    """
    对每条 eval_question 在每个 ask_at_session 切出 oracle_context。

    切片规则（多层兜底）：
    1. NEG-A/D：context = "无相关证据"占位符
    2. 字面匹配：在 oracle_dialogues[1..asked_at] 中找含 f_keys 值的 turn
    3. 兜底（找不到字面匹配时）：取最近 3 个 session 的所有 turns 作为完整上下文，
       让 LLM 在更宽语境中判断证据是否存在
    """
    out: dict[str, dict[int, OracleContext]] = {}

    for q in eval_questions:
        per_session: dict[int, OracleContext] = {}
        for asked_at in q.ask_at_sessions:
            # === Case 1：NEG-A/D 直接占位 ===
            if q.task_type == "NEG" and q.neg_subtype in {"NEG-A", "NEG-D"}:
                ctx_text = "（剧本中无相关证据，理想答案应为'无法回答'）"
                pointers = []
                per_session[asked_at] = OracleContext(
                    question_id=q.question_id, asked_at_session=asked_at,
                    context_text=ctx_text, source_pointers=pointers,
                )
                continue

            # === Case 2：字面匹配 ===
            f_keys = q.f_keys.get(asked_at, [])
            values_to_find = [str(v).strip() for _, v in f_keys if v]
            pointers = []
            ctx_lines = []
            for sess_idx in range(1, asked_at + 1):
                dialogue = oracle_dialogues.get(sess_idx, [])
                for turn_idx, turn in enumerate(dialogue):
                    text = (turn["user"] + " " + turn["assistant"]).lower()
                    if any(v.lower() in text for v in values_to_find if v):
                        ctx_lines.append(
                            f"[Session {sess_idx} turn {turn_idx}] "
                            f"User: {turn['user']}\nAssistant: {turn['assistant']}"
                        )
                        pointers.append(f"oracle_dialogues/session_{sess_idx}/turn_{turn_idx}")

            # === Case 3：字面匹配为空 → 兜底用最近 3 session 的所有 turns ===
            if not ctx_lines:
                fallback_start = max(1, asked_at - 2)
                for sess_idx in range(fallback_start, asked_at + 1):
                    dialogue = oracle_dialogues.get(sess_idx, [])
                    for turn_idx, turn in enumerate(dialogue):
                        ctx_lines.append(
                            f"[Session {sess_idx} turn {turn_idx}] "
                            f"User: {turn['user']}\nAssistant: {turn['assistant']}"
                        )
                        pointers.append(
                            f"oracle_dialogues/session_{sess_idx}/turn_{turn_idx}"
                        )

            ctx_text = "\n\n".join(ctx_lines) if ctx_lines else "（剧本无对话）"
            per_session[asked_at] = OracleContext(
                question_id=q.question_id, asked_at_session=asked_at,
                context_text=ctx_text, source_pointers=pointers,
            )
        out[q.question_id] = per_session

    if write_file:
        pdir = persona_data_dir(cfg, persona.persona_id)
        path = pdir / "oracle_contexts.json"
        serializable = {
            qid: {str(s): ctx.to_dict() for s, ctx in per_s.items()}
            for qid, per_s in out.items()
        }
        path.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"wrote {path}")

    return out


def load_oracle_dialogues(cfg: DatasetConfig, persona_id: str) -> dict[int, list[dict]]:
    pdir = persona_data_dir(cfg, persona_id)
    path = pdir / "oracle_dialogues.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


def load_oracle_contexts(cfg: DatasetConfig, persona_id: str) -> dict[str, dict[int, OracleContext]]:
    pdir = persona_data_dir(cfg, persona_id)
    path = pdir / "oracle_contexts.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        qid: {int(s): OracleContext.from_dict(d) for s, d in per_s.items()}
        for qid, per_s in raw.items()
    }


__all__ = [
    "build_oracle_dialogues",
    "build_oracle_contexts",
    "load_oracle_dialogues",
    "load_oracle_contexts",
]
