from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from memory_eval.eval_core.llm_assist import LLMAssistConfig, llm_judge_answer_correctness
from memory_eval.eval_core.models import EvaluatorConfig
from memory_eval.eval_core.utils import is_abstain, normalize_text


@dataclass(frozen=True)
class CorrectnessJudgement:
    rule_correct: bool
    llm_correct: Optional[bool]
    final_correct: bool
    judge_label: str
    judge_reason: str
    judge_payload: Dict[str, Any]


def judge_answer_correctness(
    *,
    task_type: str,
    question: str,
    answer_gold: str,
    answer_pred: str,
    cfg: EvaluatorConfig,
    judge_mode: str = "online",
    oracle_context: str = "",
    retrieved_context: str = "",
) -> CorrectnessJudgement:
    rule_correct = _rule_correct(task_type=task_type, answer_gold=answer_gold, answer_pred=answer_pred)
    hard_veto = _hard_veto(task_type=task_type, answer_pred=answer_pred)
    llm_payload: Dict[str, Any] | None = None
    llm_correct: Optional[bool] = None
    llm_available = False
    if cfg.correctness_use_llm_judge:
        llm_payload = llm_judge_answer_correctness(
            LLMAssistConfig(
                api_key=cfg.llm_api_key,
                base_url=cfg.llm_base_url,
                model=cfg.llm_model,
                temperature=cfg.llm_temperature,
            ),
            task_type=task_type,
            question=question,
            answer_gold=answer_gold,
            answer_pred=answer_pred,
            judge_mode=judge_mode,
            oracle_context=oracle_context,
            retrieved_context=retrieved_context,
            must_succeed=cfg.correctness_require_llm_judge,
        )
        if isinstance(llm_payload, dict):
            llm_available = True
            llm_correct = bool(llm_payload.get("correct", False))
    if cfg.correctness_use_llm_judge and llm_available:
        final_correct = bool(llm_correct)
        judge_label = "LLM"
        judge_reason = str((llm_payload or {}).get("reason", "LLM correctness judge.")) if llm_payload is not None else "LLM judge unavailable."
    else:
        final_correct = rule_correct and not hard_veto
        judge_label = "RULE" if not cfg.correctness_use_llm_judge else "RULE_FALLBACK"
        judge_reason = "Rule correctness judge." if not cfg.correctness_use_llm_judge else "LLM correctness judge unavailable; fallback to rule correctness judge."
    judge_payload = dict(llm_payload or {})
    judge_payload["hard_veto"] = bool(hard_veto)
    judge_payload["llm_available"] = bool(llm_available)
    judge_payload["judge_mode"] = str(judge_mode or "online")
    judge_payload["rule_correct"] = bool(rule_correct)
    return CorrectnessJudgement(
        rule_correct=rule_correct,
        llm_correct=llm_correct,
        final_correct=final_correct,
        judge_label=judge_label,
        judge_reason=judge_reason,
        judge_payload=judge_payload,
    )


def _rule_correct(*, task_type: str, answer_gold: str, answer_pred: str) -> bool:
    gold = normalize_text(answer_gold)
    pred = normalize_text(answer_pred)
    if task_type == "NEG":
        return is_abstain(answer_pred)
    if not gold or not pred:
        return False
    return pred == gold or gold in pred or pred in gold


def _hard_veto(*, task_type: str, answer_pred: str) -> bool:
    if str(task_type or "").upper() == "POS" and is_abstain(answer_pred):
        return True
    return False
