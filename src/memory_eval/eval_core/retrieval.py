from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any, Dict, List, Optional

from memory_eval.eval_core.adapter_protocol import RetrievalAdapterProtocol
from memory_eval.eval_core.llm_assist import (
    LLMAssistConfig,
    llm_judge_retrieval_noise,
    llm_judge_retrieval_quality_neg,
    llm_judge_retrieval_quality_pos,
)
from memory_eval.eval_core.models import EvalSample, EvaluatorConfig, ProbeResult
from memory_eval.eval_core.utils import is_strict_llm_probe, rank_and_hit_indices, token_overlap_snr


@dataclass(frozen=True)
class RetrievalProbeInput:
    """
    Retrieval probe explicit input.
    检索探针显式输入：Q + C_original + F_key。
    """

    question: str
    retrieved_items: List[Dict[str, Any]]
    f_key: List[str]
    task_type: str
    evidence_texts: List[str] = field(default_factory=list)


def _normalize_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in items:
        out.append(
            {
                "id": str(it.get("id", "")),
                "text": str(it.get("text", "")),
                "score": float(it.get("score", 0.0) or 0.0),
                "meta": dict(it.get("meta", {})) if isinstance(it.get("meta", {}), dict) else {},
            }
        )
    return out


def evaluate_retrieval_probe_with_adapter(
    sample: EvalSample,
    adapter: RetrievalAdapterProtocol,
    run_ctx: Any,
    cfg: EvaluatorConfig,
    top_k: int,
    s_enc: Optional[str] = None,
) -> ProbeResult:
    """
    Adapter-integrated retrieval probe entrypoint.
    检索探针入口：由适配器提供原始检索结果 C_original。
    """
    c_original = _normalize_items(adapter.retrieve_original(run_ctx, sample.question, top_k))
    evidence_texts = list(getattr(sample, "evidence_with_time", []) or getattr(sample, "evidence_texts", []))
    return evaluate_retrieval_probe(
        RetrievalProbeInput(
            question=sample.question,
            retrieved_items=c_original,
            f_key=list(sample.f_key),
            task_type=sample.task_type,
            evidence_texts=evidence_texts,
        ),
        cfg=cfg,
        s_enc=s_enc,
    )


def evaluate_retrieval_probe(inp: RetrievalProbeInput, cfg: EvaluatorConfig, s_enc: Optional[str] = None) -> ProbeResult:
    """
    Pure retrieval probe logic with explicit input.
    纯检索探针逻辑：显式输入 Q + C_original + F_key。
    """
    items = _normalize_items(inp.retrieved_items)
    rank, hit_indices = rank_and_hit_indices(items, inp.f_key)
    hit_count = len(hit_indices)
    snr, snr_meta = token_overlap_snr(items, inp.f_key)

    llm_must = bool(cfg.use_llm_assist and cfg.require_llm_judgement)

    if inp.task_type == "NEG":
        top_score = float(items[0].get("score", 0.0)) if items else 0.0
        llm_noise_reason: Dict[str, Any] | None = None
        if cfg.use_llm_assist:
            j = llm_judge_retrieval_quality_neg(
                LLMAssistConfig(
                    api_key=cfg.llm_api_key,
                    base_url=cfg.llm_base_url,
                    model=cfg.llm_model,
                    temperature=cfg.llm_temperature,
                ),
                query=inp.question,
                retrieved_items=items,
                must_succeed=llm_must,
            )
            if not isinstance(j, dict) and cfg.require_llm_judgement:
                raise RuntimeError("retrieval NEG llm judgement failed or empty")
            if isinstance(j, dict):
                llm_noise_reason = j
                state_hint = str(llm_noise_reason.get("retrieval_state", "")).upper()
                is_noise = state_hint == "NOISE" or bool(llm_noise_reason.get("is_noise", False))
                return ProbeResult(
                    probe="ret",
                    state="NOISE" if is_noise else "MISS",
                    defects=["NIR"] if is_noise else [],
                    attrs={"rank_index": rank, "snr": snr, "top_score": top_score},
                    evidence={
                        "reason": str(llm_noise_reason.get("reasoning", "LLM NEG retrieval judgement.")),
                        "threshold": cfg.neg_noise_score_threshold,
                        "top_item": items[0] if items else None,
                        "llm_noise_judgement": llm_noise_reason,
                        "snr_meta": snr_meta,
                    },
                )
            if cfg.disable_rule_fallback:
                raise RuntimeError("retrieval NEG strict mode rejects rule fallback")

        noise_by_score = top_score >= cfg.neg_noise_score_threshold
        if not llm_noise_reason and items and not cfg.disable_rule_fallback:
            j = llm_judge_retrieval_noise(
                LLMAssistConfig(
                    api_key=cfg.llm_api_key,
                    base_url=cfg.llm_base_url,
                    model=cfg.llm_model,
                    temperature=cfg.llm_temperature,
                ),
                query=inp.question,
                retrieved_items=items,
            )
            if isinstance(j, dict):
                llm_noise_reason = j
                noise_by_score = noise_by_score or bool(j.get("is_noise", False))

        if noise_by_score:
            return ProbeResult(
                probe="ret",
                state="NOISE",
                defects=["NIR"],
                attrs={"rank_index": -1, "snr": 0.0, "top_score": top_score},
                evidence={
                    "reason": "NEG sample hit high misleading retrieval noise.",
                    "threshold": cfg.neg_noise_score_threshold,
                    "top_item": items[0] if items else None,
                    "llm_noise_judgement": llm_noise_reason,
                },
            )
        return ProbeResult(
            probe="ret",
            state="MISS",
            defects=[],
            attrs={"rank_index": -1, "snr": 0.0, "top_score": top_score},
            evidence={"reason": "NEG sample has no high-noise retrieval evidence.", "llm_noise_judgement": llm_noise_reason},
        )

    # POS tasks
    llm_pos_judgement: Dict[str, Any] | None = None
    if cfg.use_llm_assist:
        llm_pos_judgement = llm_judge_retrieval_quality_pos(
            LLMAssistConfig(
                api_key=cfg.llm_api_key,
                base_url=cfg.llm_base_url,
                model=cfg.llm_model,
                temperature=cfg.llm_temperature,
            ),
            query=inp.question,
            f_key=list(inp.f_key),
            evidence_texts=list(inp.evidence_texts),
            retrieved_items=items,
            rank_index=rank,
            hit_indices=hit_indices,
            snr=snr,
            tau_rank=cfg.tau_rank,
            tau_snr=cfg.tau_snr,
            must_succeed=llm_must,
        )
        if cfg.require_llm_judgement and not isinstance(llm_pos_judgement, dict):
            raise RuntimeError("retrieval POS llm judgement failed or empty")
        if isinstance(llm_pos_judgement, dict):
            state_hint = str(llm_pos_judgement.get("retrieval_state", "")).upper()
            defects = [str(x).upper() for x in llm_pos_judgement.get("defects", []) if str(x).strip() and str(x).upper() in {"RF", "LATE", "NOI"}]
            if not is_strict_llm_probe(cfg) and state_hint in {"HIT", "MISS"}:
                if rank > cfg.tau_rank and "LATE" not in defects:
                    defects.append("LATE")
                if snr < cfg.tau_snr and "NOI" not in defects:
                    defects.append("NOI")
            if state_hint in {"HIT", "MISS"}:
                return ProbeResult(
                    probe="ret",
                    state=state_hint,
                    defects=defects,
                    attrs={"rank_index": rank, "snr": snr, "hit_count": hit_count},
                    evidence={
                        "reason": str(llm_pos_judgement.get("reasoning", "LLM POS retrieval judgement.")),
                        "f_key": list(inp.f_key),
                        "hit_indices": hit_indices,
                        "snr_meta": snr_meta,
                        "top_items": items[:5],
                        "llm_judgement": llm_pos_judgement,
                    },
                )
            if cfg.disable_rule_fallback:
                raise RuntimeError(f"retrieval POS llm returned invalid state: {state_hint or 'EMPTY'}")
    if cfg.use_llm_assist and cfg.disable_rule_fallback:
        raise RuntimeError("retrieval POS strict mode rejects rule fallback")

    if hit_count <= 0:
        # RF should only be assigned if encoding is not MISS (per final metric definition).
        return ProbeResult(
            probe="ret",
            state="MISS",
            defects=[],
            attrs={"rank_index": rank, "snr": snr},
            evidence={
                "reason": "No retrieval hit for f_key in C_original.",
                "f_key": list(inp.f_key),
                "top_items": items[:5],
                "rf_gate_s_enc": s_enc,
                "llm_judgement": llm_pos_judgement,
                "snr_meta": snr_meta,
            },
        )

    defects = []
    if rank > cfg.tau_rank:
        defects.append("LATE")
    if snr < cfg.tau_snr:
        defects.append("NOI")
    if isinstance(llm_pos_judgement, dict):
        for d in llm_pos_judgement.get("defects", []):
            d = str(d).upper()
            if d in {"LATE", "NOI"} and d not in defects:
                defects.append(d)
        state_hint = str(llm_pos_judgement.get("retrieval_state", "")).upper()
        if state_hint == "MISS":
            return ProbeResult(
                probe="ret",
                state="MISS",
                defects=defects,
                attrs={"rank_index": rank, "snr": snr, "hit_count": hit_count},
                evidence={
                    "reason": str(llm_pos_judgement.get("reasoning", "LLM judged miss on retrieval quality.")),
                    "f_key": list(inp.f_key),
                    "hit_indices": hit_indices,
                    "snr_meta": snr_meta,
                    "top_items": items[:5],
                    "llm_judgement": llm_pos_judgement,
                },
            )
    return ProbeResult(
        probe="ret",
        state="HIT",
        defects=defects,
        attrs={"rank_index": rank, "snr": snr, "hit_count": hit_count},
        evidence={
            "reason": "Retrieved items contain key facts.",
            "f_key": list(inp.f_key),
            "hit_indices": hit_indices,
            "snr_meta": snr_meta,
            "top_items": items[:5],
            "llm_judgement": llm_pos_judgement,
        },
    )
