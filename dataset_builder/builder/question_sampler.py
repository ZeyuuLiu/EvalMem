"""
阶段 1：问题集采样。

输入：Persona + StateVarLibrary
输出：每画像 100 道问题草稿（按偏好领域加权 + 各题指定 required_vars）

注意：本阶段的 100 道问题是"概念问题"草稿，后续阶段 5 会：
- 增加 ask_at_sessions（哪些 session 末问）
- 增加 gold_answers（不同 session 的答案）
- 替换部分为 NEG 题型
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from .config import DatasetConfig, StateVarLibrary, persona_data_dir
from .llm_client import LLMClient
from .schemas import Persona

logger = logging.getLogger(__name__)


QUESTION_SAMPLE_PROMPT = """你将为以下虚拟用户生成 {n_questions} 道她/他在未来 10 个 session 中可能向 AI 助手提出的问题（中文）。

【画像】
{backstory}
姓名：{name}
沟通风格：{communication_style}（问题的措辞要符合此风格）

【偏好领域权重】
{domain_weights_text}

【候选状态变量库（按领域分类）】
请只选择与对应领域相关的变量来构造问题：

{state_var_library_text}

【生成要求】
1. 每个问题必须依赖 1-2 个状态变量（这些状态在不同时刻会有不同取值，问题的答案因此会变化）
2. 问题应自然口语化（按 communication_style 调整）
3. 不要重复（语义相同的问题视为重复）
4. 共 {n_questions} 道
5. **不要透露**这是评测场景，不要使用"测试"、"评估"等词
6. 用户视角发问：是用户问助手、关于自己（用户自己）的问题（如"我现在的工作是什么"）

【输出格式（严格 JSON）】
{{
  "questions": [
    {{
      "question": "我现在的通勤方式是什么？",
      "required_vars": ["commute_mode"],
      "domain": "工作"
    }},
    ...
  ]
}}
"""


def _format_domain_weights(persona: Persona, all_domains: list[str], n_total: int) -> tuple[str, dict[str, int]]:
    """计算每领域目标题数 + 渲染 prompt 文本。"""
    pref = persona.preferred_domains
    other = [d for d in all_domains if d not in pref]
    # 偏好领域 30 题/各，其他 ~10 题/各
    pref_n = 30
    other_n = max(1, (n_total - pref_n * len(pref)) // max(1, len(other)))
    targets: dict[str, int] = {}
    for d in pref:
        targets[d] = pref_n
    for d in other:
        targets[d] = other_n
    # 微调到正好 n_total
    diff = n_total - sum(targets.values())
    if diff > 0 and pref:
        targets[pref[0]] += diff
    elif diff < 0:
        # 砍掉过量的，从 other 中扣
        for d in other:
            if diff >= 0:
                break
            cut = min(targets[d] - 1, abs(diff))
            targets[d] -= cut
            diff += cut

    text_lines = []
    for d in pref:
        text_lines.append(f"- {d}（高权重，约 {targets[d]} 题）")
    for d in other:
        text_lines.append(f"- {d}（低权重，约 {targets[d]} 题）")
    return "\n".join(text_lines), targets


def _format_state_var_library(lib: StateVarLibrary) -> str:
    """渲染候选状态变量库为 prompt 友好格式。"""
    lines = []
    for domain, vars_in_domain in lib.by_domain.items():
        lines.append(f"### {domain}")
        for v in vars_in_domain:
            lines.append(
                f"- var_name: {v['var_name']}（{v['display_name']}）"
                f" 取值范围: {v['values']}"
            )
        lines.append("")
    return "\n".join(lines)


def _validate_question(q: dict, lib: StateVarLibrary, all_domains: list[str]) -> bool:
    """基本校验：question 非空，required_vars 在库中，domain 合法。"""
    if not isinstance(q, dict):
        return False
    if not str(q.get("question", "")).strip():
        return False
    if q.get("domain") not in all_domains:
        return False
    required = q.get("required_vars", [])
    if not isinstance(required, list) or not (1 <= len(required) <= 2):
        return False
    for var in required:
        if lib.find(var) is None:
            return False
    return True


def sample_questions(
    persona: Persona,
    lib: StateVarLibrary,
    cfg: DatasetConfig,
    client: LLMClient,
    n_questions: int = 100,
) -> list[dict]:
    """
    为单画像采样问题。
    若 n_questions > 100 则分多批调用，避免单次 max_tokens 截断。
    返回 list of {question, required_vars, domain}。
    """
    all_domains = list(lib.by_domain.keys())
    state_var_library_text = _format_state_var_library(lib)

    # 分批：每批最多 90 题（在 4500 max_tokens 下安全）
    batch_size = 90
    n_batches = (n_questions + batch_size - 1) // batch_size
    per_batch = (n_questions + n_batches - 1) // n_batches

    all_valid: list[dict] = []
    seen_text: set[str] = set()

    for batch_idx in range(n_batches):
        batch_target = min(per_batch, n_questions - len(all_valid))
        if batch_target <= 0:
            break
        domain_weights_text, _ = _format_domain_weights(persona, all_domains, batch_target)

        prompt = QUESTION_SAMPLE_PROMPT.format(
            n_questions=batch_target,
            backstory=persona.backstory,
            name=persona.name,
            communication_style=persona.communication_style,
            domain_weights_text=domain_weights_text,
            state_var_library_text=state_var_library_text,
        )

        # 关键：增加 max_tokens 给单批留够空间
        try:
            response = client.chat_json(
                prompt,
                model=cfg.llm.builder_model,
                temperature=cfg.llm.builder_temperature,
                max_tokens=5000,
                use_cache=batch_idx == 0,  # 第一批可缓存；后续批用不同 prompt 也能缓存
            )
        except Exception as e:
            logger.warning(f"persona {persona.persona_id} batch {batch_idx + 1}/{n_batches} sampling failed: {e}")
            continue

        raw_questions = response.get("questions", [])
        for q in raw_questions:
            if not _validate_question(q, lib, all_domains):
                continue
            key = q["question"].strip()[:30]
            if key in seen_text:
                continue
            seen_text.add(key)
            all_valid.append({
                "question": q["question"].strip(),
                "required_vars": list(q["required_vars"]),
                "domain": q["domain"],
            })

    logger.info(f"persona {persona.persona_id}: sampled {len(all_valid)}/{n_questions} valid questions in {n_batches} batches")
    return all_valid


def build_questions_for_persona(
    persona: Persona,
    lib: StateVarLibrary,
    cfg: DatasetConfig,
    client: LLMClient,
    write_file: bool = True,
) -> list[dict]:
    """采样并落盘 questions_draft.json。请求量取自 cfg.question_distribution.question_sampler_target。"""
    target_n = cfg.question_distribution.question_sampler_target
    questions = sample_questions(persona, lib, cfg, client, n_questions=target_n)
    if write_file:
        pdir = persona_data_dir(cfg, persona.persona_id)
        path = pdir / "questions_draft.json"
        path.write_text(
            json.dumps(questions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"wrote {path} ({len(questions)} questions)")
    return questions


def load_questions_draft(cfg: DatasetConfig, persona_id: str) -> list[dict]:
    pdir = persona_data_dir(cfg, persona_id)
    path = pdir / "questions_draft.json"
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "sample_questions",
    "build_questions_for_persona",
    "load_questions_draft",
]
