"""
阶段 0：画像池构建。

10 个画像的 6 维结构化属性已在 configs/persona_dimensions.yaml 预设。
本模块负责：
1. 读取预设画像
2. 调 LLM 渲染中文姓名 + 200-300 字背景故事
3. 写入 personas.json + 各画像目录的 persona.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import (
    DatasetConfig,
    PersonaDimensionsConfig,
    load_persona_dimensions,
    persona_data_dir,
)
from .llm_client import LLMClient
from .schemas import Persona

logger = logging.getLogger(__name__)


PERSONA_RENDER_PROMPT = """你的任务是为以下用户属性渲染一份自然连贯的中文背景故事（200-300 字），用于后续 AI 助手对话场景。

【用户结构化属性】
- 年龄：{age}
- 性别：{gender}
- 教育程度：{education}
- 生活阶段：{stage}
- 沟通风格：{communication_style}
- 状态变动倾向：{change_propensity}（这影响人物的生活/工作变化频率）
- NEG 易感性：{neg_sensitivity}
- 偏好领域：{preferred_domains_str}

【生成要求】
1. 必须给出一个具体的中文姓名（符合年龄/性别习惯）
2. 背景故事应自然反映上述属性（特别是沟通风格与偏好领域）
3. 不要直接列出属性，要用叙事方式融入
4. 200-300 字
5. 不要出现"AI"、"助手"、"测试"等元层概念
6. 描述应含具体的城市、职业、家庭等细节，但避免与状态变量库里"会变化"的内容冲突

【输出格式（严格 JSON）】
{{
  "name": "中文姓名",
  "backstory": "200-300 字背景故事"
}}
"""


def render_persona(
    predefined: dict,
    rho_map: dict[str, float],
    client: LLMClient,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    use_cache: bool = True,
) -> Persona:
    """把一条预设属性渲染成完整 Persona（含 LLM 生成的 name + backstory）。"""
    prompt = PERSONA_RENDER_PROMPT.format(
        age=predefined["age"],
        gender=predefined["gender"],
        education=predefined["education"],
        stage=predefined["stage"],
        communication_style=predefined["communication_style"],
        change_propensity=predefined["change_propensity"],
        neg_sensitivity=predefined["neg_sensitivity"],
        preferred_domains_str="、".join(predefined["preferred_domains"]),
    )

    last_err = None
    for attempt in range(3):  # 校验失败重试
        try:
            response = client.chat_json(
                prompt,
                model=model,
                temperature=temperature,
                max_tokens=600,
                use_cache=use_cache and attempt == 0,
            )
            name = str(response.get("name", "")).strip()
            backstory = str(response.get("backstory", "")).strip()

            if not name:
                raise ValueError("LLM 返回缺少 name")
            if len(backstory) < 100:
                raise ValueError(f"backstory 太短 ({len(backstory)} chars)")
            if len(backstory) > 500:
                # 太长则截断到 500
                backstory = backstory[:500]

            persona = Persona(
                persona_id=predefined["persona_id"],
                name=name,
                age=predefined["age"],
                gender=predefined["gender"],
                education=predefined["education"],
                stage=predefined["stage"],
                communication_style=predefined["communication_style"],
                change_propensity=predefined["change_propensity"],
                rho=rho_map[predefined["change_propensity"]],
                neg_sensitivity=predefined["neg_sensitivity"],
                preferred_domains=list(predefined["preferred_domains"]),
                backstory=backstory,
            )
            return persona
        except (ValueError, KeyError) as e:
            last_err = e
            logger.warning(f"persona render attempt {attempt+1} failed: {e}")
            use_cache = False  # 重试不用缓存
    raise RuntimeError(f"render_persona failed: {last_err}")


def build_persona_pool(
    cfg: DatasetConfig,
    client: LLMClient,
    write_files: bool = True,
) -> list[Persona]:
    """
    构建全部 10 画像。
    write_files=True 时同时落盘 data/personas.json 与 data/per_persona/<id>/persona.json。
    """
    dims = load_persona_dimensions()
    personas: list[Persona] = []

    for predefined in dims.personas_predefined:
        logger.info(f"rendering persona {predefined['persona_id']}...")
        persona = render_persona(
            predefined,
            rho_map=dims.rho_map,
            client=client,
            model=cfg.llm.builder_model,
            temperature=cfg.llm.builder_temperature,
        )
        personas.append(persona)
        logger.info(f"  → {persona.name} ({persona.age}, {persona.stage})")

    if write_files:
        # 1. 全局 personas.json
        cfg.paths.data_root.mkdir(parents=True, exist_ok=True)
        all_path = cfg.paths.data_root / "personas.json"
        all_path.write_text(
            json.dumps([p.to_dict() for p in personas], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"wrote {all_path}")

        # 2. per-persona persona.json
        for p in personas:
            pdir = persona_data_dir(cfg, p.persona_id)
            ppath = pdir / "persona.json"
            ppath.write_text(
                json.dumps(p.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.debug(f"wrote {ppath}")

    return personas


def load_persona_pool(cfg: DatasetConfig) -> list[Persona]:
    """从已落盘的 personas.json 加载。"""
    all_path = cfg.paths.data_root / "personas.json"
    if not all_path.exists():
        raise FileNotFoundError(f"personas pool not built yet: {all_path}")
    raw = json.loads(all_path.read_text(encoding="utf-8"))
    return [Persona.from_dict(p) for p in raw]


__all__ = [
    "render_persona",
    "build_persona_pool",
    "load_persona_pool",
]
