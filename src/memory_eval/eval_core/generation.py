from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from memory_eval.eval_core.adapter_protocol import GenerationAdapterProtocol
from memory_eval.eval_core.correctness_judge import CorrectnessJudgement, judge_answer_correctness
from memory_eval.eval_core.llm_assist import LLMAssistConfig, llm_judge_generation_answer, llm_judge_generation_comparison
from memory_eval.eval_core.models import EvalSample, EvaluatorConfig, ProbeResult
from memory_eval.eval_core.utils import grounding_overlap, is_abstain, is_strict_llm_probe, normalize_text


@dataclass(frozen=True)
class GenerationProbeInput:
    """
    Generation probe explicit input.
    生成探针显式输入：Q + C_oracle + A_oracle + A_gold。
    """

    question: str
    oracle_context: str
    answer_online: str
    answer_oracle: str
    answer_gold: str
    task_type: str
    retrieved_context: str = ""


def evaluate_generation_probe_with_adapter(
    sample: EvalSample,
    adapter: GenerationAdapterProtocol,
    run_ctx: Any,
    cfg: EvaluatorConfig,
) -> ProbeResult:
    """
    Adapter-integrated generation probe entrypoint.
    生成探针入口：通过适配器调用原始记忆系统模型，基于 C_oracle 生成 A_oracle。
    """
    answer_oracle = adapter.generate_oracle_answer(run_ctx, sample.question, sample.oracle_context)
    online_fn = getattr(adapter, "generate_online_answer", None)
    answer_online = ""
    if callable(online_fn):
        try:
            answer_online = str(online_fn(run_ctx, sample.question, 5) or "")
        except Exception as exc:
            if cfg.strict_adapter_call:
                raise RuntimeError(f"generation generate_online_answer failed: {exc}") from exc
            answer_online = ""
    elif cfg.require_online_answer:
        raise RuntimeError("generation adapter missing generate_online_answer in strict mode")
    if cfg.require_online_answer and not answer_online.strip():
        raise RuntimeError("generation adapter returned empty online answer in strict mode")
    retrieved_context = ""
    retrieve_fn = getattr(adapter, "retrieve_original", None)
    if callable(retrieve_fn):
        try:
            retrieved_context = _render_retrieved_context(retrieve_fn(run_ctx, sample.question, 5))
        except Exception as exc:
            if cfg.strict_adapter_call:
                raise RuntimeError(f"generation retrieve_original failed: {exc}") from exc
            retrieved_context = ""
    return evaluate_generation_probe(
        GenerationProbeInput(
            question=sample.question,
            oracle_context=sample.oracle_context,
            answer_online=answer_online,
            answer_oracle=str(answer_oracle or ""),
            answer_gold=sample.answer_gold,
            task_type=sample.task_type,
            retrieved_context=retrieved_context,
        ),
        cfg=cfg,
    )


def _is_correct_rule(inp: GenerationProbeInput) -> bool:
    if inp.task_type == "NEG":
        return is_abstain(inp.answer_oracle)
    a = normalize_text(inp.answer_oracle)
    g = normalize_text(inp.answer_gold)
    if not g:
        return False
    return a == g or g in a


def evaluate_generation_probe(inp: GenerationProbeInput, cfg: EvaluatorConfig) -> ProbeResult:
    """
    Pure generation probe logic:
    1) produce/receive A_oracle
    2) compare A_oracle and A_gold (rule or LLM judge)
    纯生成探针逻辑：先拿 A_oracle，再与 A_gold 比较。
    """
    strict = is_strict_llm_probe(cfg)
    llm_judgement: Dict[str, Any] | None = None
    llm_comparison: Dict[str, Any] | None = None
    oracle_judgement = CorrectnessJudgement(False, None, False, "RULE", "", {})
    online_judgement = CorrectnessJudgement(False, None, False, "RULE", "", {})
    if strict:
        correct = False
        online_correct = False
    else:
        oracle_judgement = judge_answer_correctness(
            task_type=inp.task_type,
            question=inp.question,
            answer_gold=inp.answer_gold,
            answer_pred=inp.answer_oracle,
            cfg=cfg,
            judge_mode="oracle",
            oracle_context=inp.oracle_context,
        )
        online_judgement = judge_answer_correctness(
            task_type=inp.task_type,
            question=inp.question,
            answer_gold=inp.answer_gold,
            answer_pred=inp.answer_online,
            cfg=cfg,
            judge_mode="online",
            retrieved_context=inp.retrieved_context,
        )
        correct = oracle_judgement.final_correct
        online_correct = online_judgement.final_correct

    llm_must = bool(cfg.use_llm_assist and cfg.require_llm_judgement)

    if cfg.use_llm_assist:
        llm_judgement = llm_judge_generation_answer(
            LLMAssistConfig(
                api_key=cfg.llm_api_key,
                base_url=cfg.llm_base_url,
                model=cfg.llm_model,
                temperature=cfg.llm_temperature,
            ),
            question=inp.question,
            oracle_context=inp.oracle_context,
            answer_oracle=inp.answer_oracle,
            answer_gold=inp.answer_gold,
            task_type=inp.task_type,
            must_succeed=llm_must,
        )
        if cfg.require_llm_judgement and not isinstance(llm_judgement, dict):
            raise RuntimeError("generation llm answer judgement failed or empty")
        llm_comparison = llm_judge_generation_comparison(
            LLMAssistConfig(
                api_key=cfg.llm_api_key,
                base_url=cfg.llm_base_url,
                model=cfg.llm_model,
                temperature=cfg.llm_temperature,
            ),
            question=inp.question,
            task_type=inp.task_type,
            answer_gold=inp.answer_gold,
            answer_online=inp.answer_online,
            answer_oracle=inp.answer_oracle,
            oracle_context=inp.oracle_context,
            must_succeed=llm_must,
        )
        if cfg.require_llm_judgement and not isinstance(llm_comparison, dict):
            raise RuntimeError("generation llm comparison judgement failed or empty")
        if not isinstance(llm_comparison, dict) and cfg.disable_rule_fallback:
            raise RuntimeError("generation strict mode rejects rule fallback")

    overlap, overlap_meta = grounding_overlap(inp.answer_oracle, inp.oracle_context)

    if correct:
        return ProbeResult(
            probe="gen",
            state="PASS",
            defects=[],
            attrs={"grounding_overlap": overlap},
            evidence={
                "reason": "A_oracle is judged correct against A_gold.",
                "answer_online": inp.answer_online,
                "answer_oracle": inp.answer_oracle,
                "answer_gold": inp.answer_gold,
                "online_correct": online_correct,
                "oracle_correct": True,
                "online_correctness": {
                    "rule_correct": online_judgement.rule_correct,
                    "llm_correct": online_judgement.llm_correct,
                    "final_correct": online_judgement.final_correct,
                    "judge_reason": online_judgement.judge_reason,
                    "judge_payload": online_judgement.judge_payload,
                },
                "oracle_correctness": {
                    "rule_correct": oracle_judgement.rule_correct,
                    "llm_correct": oracle_judgement.llm_correct,
                    "final_correct": oracle_judgement.final_correct,
                    "judge_reason": oracle_judgement.judge_reason,
                    "judge_payload": oracle_judgement.judge_payload,
                },
                "retrieved_context": inp.retrieved_context,
                "comparative_judgement": (
                    llm_comparison.get("comparative_judgement", {})
                    if isinstance(llm_comparison, dict)
                    else {
                        "online_vs_gold": "match" if online_correct else "mismatch",
                        "oracle_vs_gold": "match",
                        "online_vs_oracle": "match" if normalize_text(inp.answer_online) == normalize_text(inp.answer_oracle) else "mismatch",
                    }
                ),
                "overlap_meta": overlap_meta,
                "llm_judgement": llm_judgement,
                "llm_comparison": llm_comparison,
            },
        )

    # FAIL classification
    if strict and inp.task_type == "NEG":
        strict_neg_defects = []
        if isinstance(llm_comparison, dict):
            strict_neg_defects = [str(x).upper() for x in llm_comparison.get("defects", []) if str(x).strip()]
        sub = str((llm_judgement or {}).get("substate", "")).upper()
        if sub == "GH" and "GH" not in strict_neg_defects:
            strict_neg_defects.append("GH")
        if "GH" not in strict_neg_defects:
            raise RuntimeError(
                "generation strict mode: NEG failure requires GH from LLM judgement/comparison, "
                f"got substate={sub!r}, defects={strict_neg_defects!r}"
            )
        return ProbeResult(
            probe="gen",
            state="FAIL",
            defects=["GH"],
            attrs={"grounding_overlap": overlap},
            evidence={
                "reason": "LLM-only NEG hallucination decision (strict).",
                "answer_online": inp.answer_online,
                "answer_oracle": inp.answer_oracle,
                "answer_gold": inp.answer_gold,
                "online_correct": online_correct,
                "oracle_correct": False,
                "online_correctness": {
                    "rule_correct": online_judgement.rule_correct,
                    "llm_correct": online_judgement.llm_correct,
                    "final_correct": online_judgement.final_correct,
                    "judge_reason": online_judgement.judge_reason,
                    "judge_payload": online_judgement.judge_payload,
                },
                "oracle_correctness": {
                    "rule_correct": oracle_judgement.rule_correct,
                    "llm_correct": oracle_judgement.llm_correct,
                    "final_correct": oracle_judgement.final_correct,
                    "judge_reason": oracle_judgement.judge_reason,
                    "judge_payload": oracle_judgement.judge_payload,
                },
                "retrieved_context": inp.retrieved_context,
                "comparative_judgement": (
                    llm_comparison.get("comparative_judgement", {})
                    if isinstance(llm_comparison, dict)
                    else {}
                ),
                "llm_judgement": llm_judgement,
                "llm_comparison": llm_comparison,
            },
        )

    if inp.task_type == "NEG":
        return ProbeResult(
            probe="gen",
            state="FAIL",
            defects=["GH"],
            attrs={"grounding_overlap": overlap},
            evidence={
                "reason": "NEG sample should abstain but oracle answer is non-abstain.",
                "answer_online": inp.answer_online,
                "answer_oracle": inp.answer_oracle,
                "answer_gold": inp.answer_gold,
                "online_correct": online_correct,
                "oracle_correct": False,
                "online_correctness": {
                    "rule_correct": online_judgement.rule_correct,
                    "llm_correct": online_judgement.llm_correct,
                    "final_correct": online_judgement.final_correct,
                    "judge_reason": online_judgement.judge_reason,
                },
                "oracle_correctness": {
                    "rule_correct": oracle_judgement.rule_correct,
                    "llm_correct": oracle_judgement.llm_correct,
                    "final_correct": oracle_judgement.final_correct,
                    "judge_reason": oracle_judgement.judge_reason,
                },
                "comparative_judgement": (
                    llm_comparison.get("comparative_judgement", {})
                    if isinstance(llm_comparison, dict)
                    else {}
                ),
                "llm_judgement": llm_judgement,
                "llm_comparison": llm_comparison,
            },
        )

    # POS fail => GF or GRF
    if strict and inp.task_type == "POS":
        sub = str((llm_judgement or {}).get("substate", "")).upper()
        if sub not in {"GF", "GRF"}:
            raise RuntimeError(
                "generation strict mode: POS failure requires llm_judgement.substate in {GF, GRF}, "
                f"got {sub!r}; llm_judgement={llm_judgement!r}"
            )
        return ProbeResult(
            probe="gen",
            state="FAIL",
            defects=[sub],
            attrs={"grounding_overlap": overlap},
            evidence={
                "reason": "LLM-only subtype decision (strict).",
                "answer_online": inp.answer_online,
                "answer_oracle": inp.answer_oracle,
                "answer_gold": inp.answer_gold,
                "online_correct": online_correct,
                "oracle_correct": False,
                "online_correctness": {
                    "rule_correct": online_judgement.rule_correct,
                    "llm_correct": online_judgement.llm_correct,
                    "final_correct": online_judgement.final_correct,
                    "judge_reason": online_judgement.judge_reason,
                },
                "oracle_correctness": {
                    "rule_correct": oracle_judgement.rule_correct,
                    "llm_correct": oracle_judgement.llm_correct,
                    "final_correct": oracle_judgement.final_correct,
                    "judge_reason": oracle_judgement.judge_reason,
                },
                "comparative_judgement": (
                    llm_comparison.get("comparative_judgement", {})
                    if isinstance(llm_comparison, dict)
                    else {}
                ),
                "llm_judgement": llm_judgement,
                "llm_comparison": llm_comparison,
                "overlap_meta": overlap_meta,
            },
        )

    if isinstance(llm_judgement, dict):
        sub = str(llm_judgement.get("substate", "")).upper()
        if sub in {"GF", "GRF"}:
            return ProbeResult(
                probe="gen",
                state="FAIL",
                defects=[sub],
                attrs={"grounding_overlap": overlap},
                evidence={
                    "reason": "LLM-assist subtype decision.",
                    "answer_online": inp.answer_online,
                    "answer_oracle": inp.answer_oracle,
                    "answer_gold": inp.answer_gold,
                    "online_correct": online_correct,
                    "oracle_correct": False,
                    "online_correctness": {
                        "rule_correct": online_judgement.rule_correct,
                        "llm_correct": online_judgement.llm_correct,
                        "final_correct": online_judgement.final_correct,
                        "judge_reason": online_judgement.judge_reason,
                        "judge_payload": online_judgement.judge_payload,
                    },
                    "oracle_correctness": {
                        "rule_correct": oracle_judgement.rule_correct,
                        "llm_correct": oracle_judgement.llm_correct,
                        "final_correct": oracle_judgement.final_correct,
                        "judge_reason": oracle_judgement.judge_reason,
                        "judge_payload": oracle_judgement.judge_payload,
                    },
                    "retrieved_context": inp.retrieved_context,
                    "comparative_judgement": (
                    llm_comparison.get("comparative_judgement", {})
                    if isinstance(llm_comparison, dict)
                    else {}
                ),
                    "llm_judgement": llm_judgement,
                    "llm_comparison": llm_comparison,
                    "overlap_meta": overlap_meta,
                },
            )

    # Rule fallback subtype
    if overlap < 0.1:
        defect = "GF"
        reason = "POS fail appears ungrounded in oracle context."
    else:
        defect = "GRF"
        reason = "POS fail appears grounded but still incorrect (reasoning failure)."
    return ProbeResult(
        probe="gen",
        state="FAIL",
        defects=[defect],
        attrs={"grounding_overlap": overlap},
        evidence={
            "reason": reason,
            "answer_online": inp.answer_online,
            "answer_oracle": inp.answer_oracle,
            "answer_gold": inp.answer_gold,
            "online_correct": online_correct,
            "oracle_correct": False,
            "online_correctness": {
                "rule_correct": online_judgement.rule_correct,
                "llm_correct": online_judgement.llm_correct,
                "final_correct": online_judgement.final_correct,
                "judge_reason": online_judgement.judge_reason,
                "judge_payload": online_judgement.judge_payload,
            },
            "oracle_correctness": {
                "rule_correct": oracle_judgement.rule_correct,
                "llm_correct": oracle_judgement.llm_correct,
                "final_correct": oracle_judgement.final_correct,
                "judge_reason": oracle_judgement.judge_reason,
                "judge_payload": oracle_judgement.judge_payload,
            },
            "retrieved_context": inp.retrieved_context,
            "comparative_judgement": (
                llm_comparison.get("comparative_judgement", {})
                if isinstance(llm_comparison, dict)
                else {}
            ),
            "overlap_meta": overlap_meta,
            "llm_judgement": llm_judgement,
            "llm_comparison": llm_comparison,
        },
    )


def _render_retrieved_context(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    lines = []
    for idx, item in enumerate(items[:5], start=1):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "") or item.get("content", "") or "").strip()
        if not text:
            continue
        lines.append(f"[{idx}] {text}")
    return "\n".join(lines)
