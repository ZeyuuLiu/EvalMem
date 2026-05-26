"""
阶段 8：异源模型验证（5 项检查）。

对每条评测样本及其 (question, asked_at_session) 元组执行 5 项独立检查：
1. 证据指针有效性（evidence_pointer）：oracle_context 是否包含足以推出 gold 的证据
2. 时序一致性（temporal_consistency）：POS 题的 gold 是否与 σ_t 一致；多 session 间的 gold 序列与状态轨迹是否吻合
3. 陷阱可分性（trap_distinguishability）：trap_answer 是否与 gold 明显不同（NEG-B/C 适用）
4. 级联因果合理性（cascade_causality）：cascade_metadata 标记的因果链在 state_evolution 中是否真的出现
5. NEG 子类区分度（neg_subtype_distinction）：NEG-B/C/D 的设计是否满足子类定义

输出：verification_report.json，每画像每检查项的通过率。
质量门槛：整体 ≥ 90% + 单项 ≥ 85%。
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from .config import DatasetConfig, persona_data_dir
from .llm_client import LLMClient
from .schemas import EvalQuestion, OracleContext, StateEvolution, StateSchema

logger = logging.getLogger(__name__)


CHECK_EVIDENCE_POINTER_PROMPT = """你正在验证一道评测样本的"证据指针有效性"。

【题目】{question}
【询问 session】{asked_at}
【标准答案】{gold}
【oracle 上下文（应包含支撑证据）】
{context}

【任务】
判断 oracle 上下文是否确实包含足以推出标准答案的信息（哪怕需要简单推理）。

【输出格式（严格 JSON）】
{{"pass": true 或 false, "reason": "..."}}

注意：若 context 为"无相关证据"占位符（NEG-A/D 题），且 task_type == NEG，则 pass=true。
"""


CHECK_TRAP_PROMPT = """你正在验证一道 NEG-B/C 题的"陷阱答案可分性"。

【题目】{question}
【询问 session】{asked_at}
【标准答案（gold）】{gold}
【陷阱答案（trap）】{trap}

【任务】
判断 gold 与 trap 在语义上是否明显不同（能被区分），且 trap 看起来确实像一个"过期"或"被覆盖"的旧值。

【输出格式（严格 JSON）】
{{"pass": true 或 false, "reason": "..."}}
"""


CHECK_NEG_D_PROMPT = """你正在验证一道 NEG-D（诱导型）题的子类合规性。

【题目】{question}
【画像 schema 中的变量】{schema_vars_text}

【任务】
判断该问题是否"表面与 schema 变量相似但语义指向另一对象"（合法 NEG-D）。例如：
- schema 含 job_title (用户自己的工作) → 合法 NEG-D = "你哥哥的工作是什么？"
- schema 含 city (用户自己所在城市) → 合法 NEG-D = "你姐姐住在哪里？"

【输出格式（严格 JSON）】
{{"pass": true 或 false, "reason": "..."}}
"""


# =====================================================================
# 单项检查实现
# =====================================================================

def check_evidence_pointer(
    question: EvalQuestion,
    asked_at: int,
    oracle_ctx: OracleContext,
    cfg: DatasetConfig,
    client: LLMClient,
) -> dict:
    """检查 1：oracle_context 是否含证据。"""
    gold = question.gold_answers.get(asked_at, "")
    prompt = CHECK_EVIDENCE_POINTER_PROMPT.format(
        question=question.question,
        asked_at=asked_at,
        gold=gold,
        context=oracle_ctx.context_text[:1500],
    )
    try:
        response = client.chat_json(
            prompt, model=cfg.llm.verifier_model,
            temperature=cfg.llm.verifier_temperature, max_tokens=300,
        )
        return {
            "pass": bool(response.get("pass", False)),
            "reason": str(response.get("reason", "")),
        }
    except Exception as e:
        return {"pass": False, "reason": f"LLM error: {e}"}


def check_temporal_consistency(
    question: EvalQuestion,
    evolution: StateEvolution,
) -> dict:
    """检查 2：纯算法。
    POS 题：gold_answers[t] 应等于 compute_gold(σ_t, required_vars)。
    """
    if question.task_type != "POS":
        return {"pass": True, "reason": "non-POS skipped"}

    for s in question.ask_at_sessions:
        snap = next((sn for sn in evolution.trajectory if sn.session == s), None)
        if snap is None:
            return {"pass": False, "reason": f"session {s} not in trajectory"}
        if len(question.required_vars) == 1:
            expected = str(snap.state.get(question.required_vars[0], ""))
        else:
            expected = "；".join([f"{v}={snap.state.get(v, '')}" for v in question.required_vars])
        actual = str(question.gold_answers.get(s, ""))
        # 单 var 情况下直接对比；多 var 时只要 expected 中所有 value 都在 actual 中即可
        if len(question.required_vars) == 1:
            if expected != actual:
                return {"pass": False, "reason": f"session {s}: expected {expected!r}, got {actual!r}"}
        else:
            for v in question.required_vars:
                val = str(snap.state.get(v, ""))
                if val and val not in actual:
                    return {"pass": False, "reason": f"session {s}: missing {v}={val} in gold {actual!r}"}
    return {"pass": True, "reason": "ok"}


def check_trap_distinguishability(
    question: EvalQuestion,
    asked_at: int,
    cfg: DatasetConfig,
    client: LLMClient,
) -> dict:
    """检查 3：trap_answer 是否与 gold 明显不同（NEG-B/C）。"""
    if question.neg_subtype not in {"NEG-B", "NEG-C"}:
        return {"pass": True, "reason": "not applicable"}
    trap = question.trap_answers.get(asked_at, "")
    gold = question.gold_answers.get(asked_at, "")
    if not trap or not gold:
        return {"pass": True, "reason": "no trap defined (skipped)"}
    # 字面相同 → 直接 fail
    if trap.strip() == gold.strip():
        return {"pass": False, "reason": "trap equals gold literally"}
    # LLM 校验
    prompt = CHECK_TRAP_PROMPT.format(
        question=question.question, asked_at=asked_at, gold=gold, trap=trap,
    )
    try:
        response = client.chat_json(
            prompt, model=cfg.llm.verifier_model,
            temperature=cfg.llm.verifier_temperature, max_tokens=200,
        )
        return {
            "pass": bool(response.get("pass", False)),
            "reason": str(response.get("reason", "")),
        }
    except Exception as e:
        return {"pass": False, "reason": f"LLM error: {e}"}


def check_cascade_causality(
    question: EvalQuestion,
    evolution: StateEvolution,
) -> dict:
    """检查 4：级联标记是否在 state_evolution 中真的出现。"""
    cascade = question.cascade_metadata
    if not cascade or not cascade.get("is_cascade_var"):
        return {"pass": True, "reason": "not cascade"}
    main_core = cascade.get("main_core")
    main_change_session = cascade.get("main_core_change_session")
    # 检查 main_core 是否在 main_change_session 真的变了
    if main_change_session is None:
        return {"pass": False, "reason": "missing main_core_change_session"}
    snap = next((s for s in evolution.trajectory if s.session == main_change_session), None)
    if snap is None:
        return {"pass": False, "reason": f"session {main_change_session} not found"}
    if not snap.is_main_core_change:
        return {"pass": False, "reason": f"session {main_change_session} not marked is_main_core_change"}
    # 检查 required_vars 中有 var 在 main_core 之后发生过 cascaded 变化
    for s in evolution.trajectory:
        if s.session <= main_change_session:
            continue
        if not s.is_aux_cascade:
            continue
        for change_desc in s.changes_from_prev:
            if "cascaded" in change_desc:
                for var in question.required_vars:
                    if change_desc.startswith(var + ":"):
                        return {"pass": True, "reason": f"cascaded at session {s.session}"}
    return {"pass": False, "reason": "no cascade chain found in evolution"}


def check_neg_subtype_distinction(
    question: EvalQuestion,
    schema: StateSchema,
    evolution: StateEvolution,
    cfg: DatasetConfig,
    client: LLMClient,
) -> dict:
    """检查 5：NEG 子类的设计是否合规。"""
    if question.task_type != "NEG":
        return {"pass": True, "reason": "non-NEG"}

    sub = question.neg_subtype
    if sub == "NEG-A":
        # required_vars 应为空
        if not question.required_vars:
            return {"pass": True, "reason": "ok (no required_vars)"}
        return {"pass": False, "reason": "NEG-A has required_vars"}
    if sub == "NEG-B":
        # I_t 中应有该 var 的旧值
        var = question.required_vars[0] if question.required_vars else None
        if not var:
            return {"pass": False, "reason": "missing required_vars"}
        found_in_I_t = False
        for items in evolution.interference_set.values():
            for it in items:
                if it.var == var:
                    found_in_I_t = True
                    break
            if found_in_I_t:
                break
        if not found_in_I_t:
            return {"pass": False, "reason": f"var {var} not in any I_t (no expired value)"}
        return {"pass": True, "reason": "ok"}
    if sub == "NEG-C":
        # 至少一个 var；本期作合规校验放宽
        if question.required_vars:
            return {"pass": True, "reason": "ok"}
        return {"pass": False, "reason": "NEG-C missing required_vars"}
    if sub == "NEG-D":
        # LLM 校验"表层相似但语义不同"
        schema_vars_text = "、".join([f"{v.var_name}({v.display_name})" for v in schema.variables])
        prompt = CHECK_NEG_D_PROMPT.format(
            question=question.question, schema_vars_text=schema_vars_text,
        )
        try:
            response = client.chat_json(
                prompt, model=cfg.llm.verifier_model,
                temperature=cfg.llm.verifier_temperature, max_tokens=200,
            )
            return {
                "pass": bool(response.get("pass", False)),
                "reason": str(response.get("reason", "")),
            }
        except Exception as e:
            return {"pass": False, "reason": f"LLM error: {e}"}
    return {"pass": False, "reason": f"unknown NEG subtype {sub}"}


# =====================================================================
# 主入口
# =====================================================================

def verify_persona(
    persona_id: str,
    cfg: DatasetConfig,
    client: LLMClient,
) -> dict:
    """对单画像跑 5 项检查，返回 dict 报告。"""
    pdir = persona_data_dir(cfg, persona_id)

    questions = [EvalQuestion.from_dict(d) for d in
                 json.loads((pdir / "eval_questions.json").read_text(encoding="utf-8"))]
    schema = StateSchema.from_dict(
        json.loads((pdir / "state_schema.json").read_text(encoding="utf-8"))
    )
    evolution = StateEvolution.from_dict(
        json.loads((pdir / "state_evolution.json").read_text(encoding="utf-8"))
    )
    oracle_raw = json.loads((pdir / "oracle_contexts.json").read_text(encoding="utf-8"))
    oracle_map: dict[str, dict[int, OracleContext]] = {}
    for qid, per_s in oracle_raw.items():
        oracle_map[qid] = {int(s): OracleContext.from_dict(d) for s, d in per_s.items()}

    results: dict[str, list[dict]] = {
        "evidence_pointer": [],
        "temporal_consistency": [],
        "trap_distinguishability": [],
        "cascade_causality": [],
        "neg_subtype_distinction": [],
    }

    for question in questions:
        # 时序一致性 + 级联因果（一题一次，非 per-ask）
        results["temporal_consistency"].append({
            "qid": question.question_id,
            **check_temporal_consistency(question, evolution),
        })
        results["cascade_causality"].append({
            "qid": question.question_id,
            **check_cascade_causality(question, evolution),
        })
        results["neg_subtype_distinction"].append({
            "qid": question.question_id,
            **check_neg_subtype_distinction(question, schema, evolution, cfg, client),
        })

        # 证据指针 + trap 可分性（per-ask 各一次）
        for asked_at in question.ask_at_sessions:
            ctx = oracle_map.get(question.question_id, {}).get(asked_at)
            if ctx is not None:
                results["evidence_pointer"].append({
                    "qid": question.question_id, "asked_at": asked_at,
                    **check_evidence_pointer(question, asked_at, ctx, cfg, client),
                })
            results["trap_distinguishability"].append({
                "qid": question.question_id, "asked_at": asked_at,
                **check_trap_distinguishability(question, asked_at, cfg, client),
            })

    # 通过率
    pass_rates = {}
    for check, items in results.items():
        if not items:
            pass_rates[check] = 1.0
            continue
        n_pass = sum(1 for it in items if it.get("pass"))
        pass_rates[check] = n_pass / len(items)

    overall = sum(pass_rates.values()) / max(1, len(pass_rates))

    report = {
        "persona_id": persona_id,
        "pass_rates": pass_rates,
        "overall_pass_rate": overall,
        "details": results,
        "thresholds": {
            "overall_min": cfg.quality_thresholds.verification_pass_rate_min,
            "single_check_min": 0.85,
        },
        "all_thresholds_met": (
            overall >= cfg.quality_thresholds.verification_pass_rate_min
            and all(v >= 0.85 for v in pass_rates.values())
        ),
    }

    # 落盘
    out_path = pdir / "verification_report.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    logger.info(
        f"persona {persona_id}: verification overall={overall:.2%}, "
        f"per-check={pass_rates}"
    )
    return report


__all__ = [
    "verify_persona",
    "check_evidence_pointer",
    "check_temporal_consistency",
    "check_trap_distinguishability",
    "check_cascade_causality",
    "check_neg_subtype_distinction",
]
