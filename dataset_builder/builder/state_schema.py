"""
阶段 2：状态 schema 构造 + 双核心变量选择 + 级联检测。

输入：questions_draft.json（含每题的 required_vars）
输出：state_schema.json
"""
from __future__ import annotations

import json
import logging
from collections import Counter

from .config import DatasetConfig, StateVarLibrary, persona_data_dir
from .llm_client import LLMClient
from .schemas import CascadeRule, Persona, StateSchema, StateVariable

logger = logging.getLogger(__name__)


CASCADE_DETECTION_PROMPT = """你将分析一组状态变量，找出与"主核心变量"存在因果连锁关系的辅核心变量。

【画像】
{backstory}

【主核心变量】
{main_core_display}（var_name = {main_core}）
该变量在第 6 个 session 会发生一次变更。

【候选普通变量】
{candidates_text}

【判断规则】
- "因果连锁"指：主核心一旦变化，该变量在现实中通常也会跟着变化（或变化的概率显著上升）
- 例如：搬家(city) → 通勤方式(commute_mode)、就业状态变化 → 收入水平
- 不要把无关变量（如 age / education）选入
- cascade_lag = "immediate" 表示主核心变更同 session 内就跟着变；"delayed" 表示 2-4 session 后才跟着变

【输出格式（严格 JSON）】
{{
  "auxiliary_cores": [
    {{
      "var_name": "...",
      "cascade_lag": "immediate" 或 "delayed",
      "cascade_reason": "一句话解释为什么"
    }}
  ]
}}

至多输出 2 个辅核心；若主核心实在没有合适的辅核心，输出空列表。
"""


def select_main_core(questions: list[dict], lib: StateVarLibrary) -> str:
    """
    选取被最多 question 依赖的状态变量为主核心。
    若多个变量依赖数相同，按 semantic_dynamic_score 优先。
    """
    var_count: Counter = Counter()
    for q in questions:
        for v in q.get("required_vars", []):
            var_count[v] += 1

    if not var_count:
        raise ValueError("no required_vars in any question")

    candidates = []
    for var, count in var_count.items():
        meta = lib.find(var)
        score = meta.get("semantic_dynamic_score", 0.5) if meta else 0.5
        candidates.append((var, count, score))

    candidates.sort(key=lambda x: (-x[1], -x[2]))
    return candidates[0][0]


def collect_used_vars(questions: list[dict]) -> list[str]:
    """收集所有 question 中实际用到的 var_name（去重）。"""
    seen = []
    seen_set = set()
    for q in questions:
        for v in q.get("required_vars", []):
            if v not in seen_set:
                seen.append(v)
                seen_set.add(v)
    return seen


def build_state_schema(
    persona: Persona,
    questions: list[dict],
    lib: StateVarLibrary,
    cfg: DatasetConfig,
    client: LLMClient,
    cascade_lag_override: str | None = None,
    write_file: bool = True,
) -> StateSchema:
    """
    构造状态 schema：变量集 + 主核心 + 辅核心。
    cascade_lag_override：若指定（"immediate"/"delayed"），强制 LLM 选定的辅核心使用该 lag。
    """
    # === Step 1: 选主核心 ===
    main_core = select_main_core(questions, lib)
    logger.info(f"persona {persona.persona_id}: main_core = {main_core}")

    # === Step 2: 收集 schema 变量集（必须含所有问题用到的）===
    used_vars = collect_used_vars(questions)
    if main_core not in used_vars:
        used_vars.insert(0, main_core)

    # 上限 8 个变量；若问题用到的太多，按依赖次数排序取前 8
    if len(used_vars) > 8:
        var_count = Counter()
        for q in questions:
            for v in q.get("required_vars", []):
                var_count[v] += 1
        used_vars_sorted = sorted(used_vars, key=lambda v: -var_count.get(v, 0))
        used_vars = used_vars_sorted[:8]
        if main_core not in used_vars:
            used_vars[-1] = main_core

    variables: list[StateVariable] = []
    for var_name in used_vars:
        meta = lib.find(var_name)
        if meta is None:
            logger.warning(f"  var {var_name} not in library, skipped")
            continue
        # 找该变量所在 domain
        domain = None
        for d, vs in lib.by_domain.items():
            if any(v["var_name"] == var_name for v in vs):
                domain = d
                break
        variables.append(StateVariable(
            var_name=meta["var_name"],
            display_name=meta["display_name"],
            values=list(meta["values"]),
            domain=domain or "未知",
            description=meta.get("description", ""),
            semantic_dynamic_score=float(meta.get("semantic_dynamic_score", 0.5)),
        ))

    if main_core not in [v.var_name for v in variables]:
        # 强制把 main_core 加入
        meta = lib.find(main_core)
        if meta is None:
            raise ValueError(f"main_core {main_core} not in library")
        variables.append(StateVariable(
            var_name=meta["var_name"],
            display_name=meta["display_name"],
            values=list(meta["values"]),
            domain="未知",
            description=meta.get("description", ""),
            semantic_dynamic_score=float(meta.get("semantic_dynamic_score", 0.5)),
        ))

    # === Step 3: LLM 检测辅核心 ===
    main_var_meta = lib.find(main_core)
    candidates = [v for v in variables if v.var_name != main_core]
    candidates_text = "\n".join([
        f"- var_name: {v.var_name}（{v.display_name}） 取值: {v.values} 领域: {v.domain}"
        for v in candidates
    ])

    prompt = CASCADE_DETECTION_PROMPT.format(
        backstory=persona.backstory,
        main_core=main_core,
        main_core_display=main_var_meta["display_name"] if main_var_meta else main_core,
        candidates_text=candidates_text,
    )

    try:
        response = client.chat_json(
            prompt,
            model=cfg.llm.builder_model,
            temperature=cfg.llm.schema_temperature,
            max_tokens=600,
        )
        raw_aux = response.get("auxiliary_cores", [])
    except Exception as e:
        logger.warning(f"cascade detection LLM failed: {e}; using 0 auxiliary cores")
        raw_aux = []

    aux_cores: list[CascadeRule] = []
    valid_var_names = {v.var_name for v in variables}
    for entry in raw_aux[:2]:  # 至多 2 个
        if not isinstance(entry, dict):
            continue
        var_name = str(entry.get("var_name", "")).strip()
        if not var_name or var_name not in valid_var_names or var_name == main_core:
            continue
        lag = str(entry.get("cascade_lag", "delayed")).strip().lower()
        if lag not in {"immediate", "delayed"}:
            lag = "delayed"
        if cascade_lag_override:
            lag = cascade_lag_override
        aux_cores.append(CascadeRule(
            main_var=main_core,
            aux_var=var_name,
            cascade_lag=lag,
            delayed_session_min=2,
            delayed_session_max=4,
            cascade_reason=str(entry.get("cascade_reason", "")).strip(),
        ))

    if not aux_cores:
        # 兜底：若 LLM 没给出，自动选 score 第二高的非 main_core 变量
        if len(variables) > 1:
            sorted_v = sorted(
                [v for v in variables if v.var_name != main_core],
                key=lambda v: -v.semantic_dynamic_score,
            )
            aux_cores.append(CascadeRule(
                main_var=main_core,
                aux_var=sorted_v[0].var_name,
                cascade_lag=cascade_lag_override or "delayed",
                cascade_reason="fallback: highest dynamic score",
            ))

    schema = StateSchema(
        persona_id=persona.persona_id,
        variables=variables,
        main_core=main_core,
        auxiliary_cores=aux_cores,
    )

    logger.info(
        f"persona {persona.persona_id}: schema with {len(variables)} vars, "
        f"main_core={main_core}, aux={[c.aux_var for c in aux_cores]}"
    )

    if write_file:
        pdir = persona_data_dir(cfg, persona.persona_id)
        path = pdir / "state_schema.json"
        path.write_text(
            json.dumps(schema.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"wrote {path}")

    return schema


def load_state_schema(cfg: DatasetConfig, persona_id: str) -> StateSchema:
    pdir = persona_data_dir(cfg, persona_id)
    path = pdir / "state_schema.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return StateSchema.from_dict(raw)


__all__ = [
    "select_main_core",
    "build_state_schema",
    "load_state_schema",
]
