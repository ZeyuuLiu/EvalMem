"""
阶段 5：种子话术生成。

为每 session 生成 1 条用户对助手说的"第一句话"（强制起始 turn），
必须自然带出 must_expose 中的第一项信息。
"""
from __future__ import annotations

import json
import logging

from .config import DatasetConfig, load_persona_dimensions, persona_data_dir
from .llm_client import LLMClient
from .schemas import (
    ExposureItem,
    ExposurePlan,
    Persona,
    SeedUtterance,
    StateSchema,
)

logger = logging.getLogger(__name__)


SEED_PROMPT = """画像背景：
{backstory}

姓名：{name}
沟通风格：{communication_style}

【沟通风格规则（必须严格执行）】
{style_detail}

本 session 必须暴露的首项信息：{first_var_display}（var_name={first_var}）= {first_value}
情感基调：{sentiment}
session 主题：{session_theme}

【任务】
生成本 session 用户对 AI 助手说的**第一句话**（用户视角，对助手说话）。

【约束】
1. 长度 15-50 字
2. 必须自然带出信息：{first_var}={first_value}
3. 严格按沟通风格
4. 不要直接陈述（如不要说"我住在北京"），要在生活场景中带出（如"今天北京下雨，想找点室内活动..."）
5. 不要使用"AI"、"助手"等元层称呼
6. 不要写成报告体，要像真人发消息

【输出格式（严格 JSON）】
{{
  "seed_text": "...",
  "expects_to_expose": ["{first_var}"]
}}
"""


def build_seed_utterances(
    persona: Persona,
    schema: StateSchema,
    plan: ExposurePlan,
    cfg: DatasetConfig,
    client: LLMClient,
    write_file: bool = True,
) -> dict[int, SeedUtterance]:
    """为每 session 生成种子话术。"""
    dims = load_persona_dimensions()
    style_detail = dims.communication_style_detailed.get(
        persona.communication_style, ""
    )

    seeds: dict[int, SeedUtterance] = {}

    for session_plan in plan.sessions:
        if not session_plan.must_expose:
            logger.warning(f"session {session_plan.session} has no must_expose, skipping seed")
            continue

        first = session_plan.must_expose[0]
        try:
            first_var_display = schema.get(first.var).display_name
        except KeyError:
            first_var_display = first.var

        prompt = SEED_PROMPT.format(
            backstory=persona.backstory,
            name=persona.name,
            communication_style=persona.communication_style,
            style_detail=style_detail,
            first_var=first.var,
            first_var_display=first_var_display,
            first_value=first.value,
            sentiment=session_plan.expose_sentiment,
            session_theme=session_plan.session_theme,
        )

        try:
            response = client.chat_json(
                prompt,
                model=cfg.llm.builder_model,
                temperature=cfg.llm.user_sim_temperature,  # 种子话术用 0.7 增加自然度
                max_tokens=200,
            )
            seed_text = str(response.get("seed_text", "")).strip()
            expects = response.get("expects_to_expose", [first.var])
            if not seed_text:
                raise ValueError("seed_text 为空")
            if len(seed_text) > 100:
                seed_text = seed_text[:100]
        except Exception as e:
            logger.warning(f"seed gen LLM failed at session {session_plan.session}: {e}; using fallback")
            seed_text = f"嗨，最近{first_var_display}方面有点事想聊聊。"
            expects = [first.var]

        seeds[session_plan.session] = SeedUtterance(
            persona_id=persona.persona_id,
            session=session_plan.session,
            seed_text=seed_text,
            expects_to_expose=list(expects) if isinstance(expects, list) else [first.var],
        )
        logger.debug(f"  session {session_plan.session} seed: {seed_text}")

    if write_file:
        pdir = persona_data_dir(cfg, persona.persona_id)
        path = pdir / "seed_utterances.json"
        # 按 session 索引落盘
        out_dict = {str(s): seeds[s].to_dict() for s in sorted(seeds.keys())}
        path.write_text(
            json.dumps(out_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"wrote {path}")

    return seeds


def load_seed_utterances(cfg: DatasetConfig, persona_id: str) -> dict[int, SeedUtterance]:
    pdir = persona_data_dir(cfg, persona_id)
    path = pdir / "seed_utterances.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): SeedUtterance.from_dict(v) for k, v in raw.items()}


__all__ = ["build_seed_utterances", "load_seed_utterances"]
