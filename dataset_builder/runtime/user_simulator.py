"""
Runtime 阶段 10：用户模拟器（user simulator）。

核心机制（参考 AMemGym §3.2）：
- 每 session 开始用预生成的种子话术（grounded utterance）
- 中间 turn 由用户模拟器实时生成，conditioning on:
  画像 + 当前状态 σ_t + 还未暴露的 exposed_t + 助手最新回复
- 最后一轮若 exposed_t 还有残余，调用 force_expose 兜底强制暴露

每个被测系统跑出的对话不同（因为助手的回复不同导致用户模拟器后续不同），
但 exposed_t 必须 100% 暴露完毕（公平性硬约束）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from builder.config import DatasetConfig, load_persona_dimensions
from builder.llm_client import LLMClient
from builder.schemas import DialogueTurn, ExposureItem, Persona

logger = logging.getLogger(__name__)


USER_SIM_PROMPT = """你正在扮演一位虚拟用户与 AI 助手对话。**严格保持人物一致性**。

【你的画像】
{backstory}

【你的姓名】{name}

【你的当前真实状态（仅你自己知道）】
{current_state_text}

【沟通风格规则（必须严格执行）】
{style_detail}

【本 session 信息暴露任务】
你必须在本 session 自然地把以下信息透露给 AI 助手：
{exposed_remaining_text}

【本 session 情感基调】{sentiment}
【本 session 主题】{session_theme}

【对话上下文（最近 6 轮）】
{recent_dialogue_text}

【AI 助手刚刚的回复】
{last_assistant_reply}

【请生成你下一条消息】
要求：
1. 必须严格保持画像沟通风格
2. 优先回应助手的回复，让对话自然延续
3. 在自然的语境下隐式暴露 exposed_remaining 中至少一项（如果还有剩余）
4. 不要生硬地报告状态，要像真实用户那样在交流中带出
5. 长度 10-60 字
6. 不要出现"AI"、"助手"、"测试"等元层概念

【输出格式（严格 JSON）】
{{
  "text": "你的下一条消息",
  "exposed_in_this_msg": ["var_name1", ...],
  "internal_reasoning": "为什么这样说（一句话）"
}}
"""


FORCE_EXPOSE_PROMPT = """你正在扮演虚拟用户。本 session 即将结束，但你还有以下信息**必须**告诉 AI 助手：
{exposed_remaining_text}

【你的画像】
{backstory}
姓名：{name}
沟通风格：{communication_style}

【强制暴露任务】
请生成最后一条消息，把这些信息**全部说出来**。允许稍显刻意（比如"对了，我想起来还没告诉你..."的过渡），但仍需保持画像沟通风格。

【输出格式（严格 JSON）】
{{
  "text": "...",
  "exposed_in_this_msg": [{vars_list}]
}}
"""


EXPOSURE_CHECK_PROMPT = """判断下面这条用户消息是否（哪怕用暗示、比喻、委婉表达）传达了 "{var_display} = {value}" 这个信息。

用户消息：{user_msg}

【输出格式（严格 JSON）】
{{"exposed": true 或 false, "reason": "一句话理由"}}
"""


@dataclass
class UserSimResponse:
    text: str
    exposed_in_this_msg: list[str]
    internal_reasoning: str = ""


def _format_state(state: dict) -> str:
    return "\n".join([f"- {k}: {v}" for k, v in state.items()])


def _format_exposed_remaining(items: list[ExposureItem]) -> str:
    if not items:
        return "（已无未暴露项）"
    return "\n".join([f"- {it.var} = {it.value}（提示：{it.tone_hint}）" for it in items])


def _format_recent_dialogue(turns: list[DialogueTurn], max_recent: int = 6) -> str:
    if not turns:
        return "（暂无对话历史）"
    recent = turns[-max_recent:]
    return "\n".join([f"User: {t.user_message}\nAssistant: {t.assistant_reply}" for t in recent])


def call_user_simulator(
    persona: Persona,
    current_state: dict,
    exposed_remaining: list[ExposureItem],
    dialogue_history: list[DialogueTurn],
    last_assistant_reply: str,
    sentiment: str,
    session_theme: str,
    cfg: DatasetConfig,
    client: LLMClient,
) -> UserSimResponse:
    """主用户模拟器调用。返回下一条用户消息 + 自报暴露的 var 列表。"""
    dims = load_persona_dimensions()
    style_detail = dims.communication_style_detailed.get(persona.communication_style, "")

    prompt = USER_SIM_PROMPT.format(
        backstory=persona.backstory,
        name=persona.name,
        current_state_text=_format_state(current_state),
        style_detail=style_detail,
        exposed_remaining_text=_format_exposed_remaining(exposed_remaining),
        sentiment=sentiment,
        session_theme=session_theme,
        recent_dialogue_text=_format_recent_dialogue(dialogue_history),
        last_assistant_reply=last_assistant_reply or "（这是 session 第二轮，请自然延续）",
    )

    try:
        response = client.chat_json(
            prompt,
            model=cfg.llm.user_sim_model,
            temperature=cfg.llm.user_sim_temperature,
            max_tokens=300,
            use_cache=False,  # 在线对话不缓存
        )
        text = str(response.get("text", "")).strip()
        exposed = response.get("exposed_in_this_msg", [])
        if not isinstance(exposed, list):
            exposed = []
        if not text:
            raise ValueError("empty text")
        if len(text) > 200:
            text = text[:200]
        return UserSimResponse(
            text=text,
            exposed_in_this_msg=[str(v) for v in exposed],
            internal_reasoning=str(response.get("internal_reasoning", "")),
        )
    except Exception as e:
        logger.warning(f"user_simulator failed: {e}; using fallback")
        # Fallback: 直接讲下一项 must_expose
        if exposed_remaining:
            item = exposed_remaining[0]
            return UserSimResponse(
                text=f"对了，我的{item.var}最近是{item.value}。",
                exposed_in_this_msg=[item.var],
            )
        return UserSimResponse(text="嗯，我先想想。", exposed_in_this_msg=[])


def call_force_expose(
    persona: Persona,
    exposed_remaining: list[ExposureItem],
    cfg: DatasetConfig,
    client: LLMClient,
) -> UserSimResponse:
    """最后一轮强制暴露。"""
    if not exposed_remaining:
        return UserSimResponse(text="差不多就这些了，谢谢！", exposed_in_this_msg=[])

    vars_list_str = ", ".join([f'"{it.var}"' for it in exposed_remaining])
    prompt = FORCE_EXPOSE_PROMPT.format(
        exposed_remaining_text=_format_exposed_remaining(exposed_remaining),
        backstory=persona.backstory,
        name=persona.name,
        communication_style=persona.communication_style,
        vars_list=vars_list_str,
    )

    try:
        response = client.chat_json(
            prompt,
            model=cfg.llm.user_sim_model,
            temperature=cfg.llm.user_sim_temperature,
            max_tokens=400,
            use_cache=False,
        )
        text = str(response.get("text", "")).strip()
        exposed = response.get("exposed_in_this_msg", [])
        return UserSimResponse(
            text=text or "对了，我想起来还有几件事要告诉你。",
            exposed_in_this_msg=[str(v) for v in exposed] if isinstance(exposed, list) else [],
        )
    except Exception as e:
        logger.warning(f"force_expose failed: {e}; using fallback")
        # Fallback: 把所有剩余项串成一句话
        parts = [f"{it.var}={it.value}" for it in exposed_remaining]
        return UserSimResponse(
            text=f"对了，我还想告诉你：{'; '.join(parts)}。",
            exposed_in_this_msg=[it.var for it in exposed_remaining],
        )


def llm_check_exposure(
    user_msg: str,
    item: ExposureItem,
    cfg: DatasetConfig,
    client: LLMClient,
) -> bool:
    """LLM 二次确认 user_msg 是否暴露了 item（针对暗示 / 婉转表达）。"""
    prompt = EXPOSURE_CHECK_PROMPT.format(
        var_display=item.var,
        value=item.value,
        user_msg=user_msg,
    )
    try:
        response = client.chat_json(
            prompt,
            model=cfg.llm.exposure_check_model,
            temperature=0.0,
            max_tokens=100,
            use_cache=True,  # 同样的 (msg, item) 重复用同样判定
        )
        return bool(response.get("exposed", False))
    except Exception as e:
        logger.warning(f"llm_check_exposure failed: {e}; defaulting to False")
        return False


def mark_exposed(
    user_msg: str,
    self_reported: list[str],
    exposed_remaining: list[ExposureItem],
    cfg: DatasetConfig,
    client: LLMClient,
) -> list[str]:
    """
    确认哪些 var 真的被暴露在 user_msg 中（双层校验）。
    - 字面包含 → 直接通过
    - LLM 二次确认 → 兜底
    返回确认暴露的 var 列表，并 in-place 修改 exposed_remaining。
    """
    confirmed: list[str] = []
    items_by_var = {it.var: it for it in exposed_remaining}

    for var in self_reported:
        item = items_by_var.get(var)
        if item is None:
            continue
        # Layer 1: 字面字符串包含
        if str(item.value).lower() in user_msg.lower():
            confirmed.append(var)
            continue
        # Layer 2: LLM 校验
        if llm_check_exposure(user_msg, item, cfg, client):
            confirmed.append(var)

    # in-place 移除已确认
    exposed_remaining[:] = [it for it in exposed_remaining if it.var not in confirmed]
    return confirmed


__all__ = [
    "UserSimResponse",
    "call_user_simulator",
    "call_force_expose",
    "mark_exposed",
    "llm_check_exposure",
]
