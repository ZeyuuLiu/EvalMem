from __future__ import annotations

from typing import Dict, List, Optional

from memory_eval.eval_core.llm_assist import LLMAssistConfig, llm_judge_attribution
from memory_eval.eval_core.models import AttributionAssessment, AttributionResult, EvalSample, EvaluatorConfig, ProbeResult
from memory_eval.eval_core.utils import ordered_defect_union


class AttributionAgent:
    name = "AttributionAgent"

    def attribute(
        self,
        sample: EvalSample,
        enc: ProbeResult,
        ret: ProbeResult,
        gen: ProbeResult,
        cfg: Optional[EvaluatorConfig] = None,
    ) -> AttributionResult:
        ret_defects = list(ret.defects)
        decision_trace: List[str] = []
        if ret.state == "MISS" and enc.state != "MISS" and "RF" not in ret_defects and sample.task_type == "POS":
            ret_defects.append("RF")
            decision_trace.append("Added RF because retrieval missed while encoding was not MISS.")
        if enc.state == "MISS" and "RF" in ret_defects:
            ret_defects = [defect for defect in ret_defects if defect != "RF"]
            decision_trace.append("Suppressed RF because encoding state is MISS.")
        merged_ret = ProbeResult(
            probe=ret.probe,
            state=ret.state,
            defects=ret_defects,
            evidence=dict(ret.evidence),
            attrs=dict(ret.attrs),
        )
        merged_defects = ordered_defect_union(enc.defects, merged_ret.defects, gen.defects)
        assessment = self._build_assessment(sample, enc, merged_ret, gen, decision_trace, cfg)
        probe_results: Dict[str, ProbeResult] = {"enc": enc, "ret": merged_ret, "gen": gen}
        final_attribution = {
            "agent_name": self.name,
            "final_attribution": assessment.primary_cause if assessment.primary_cause != "none" else "no_major_failure",
            "primary_cause": assessment.primary_cause,
            "secondary_causes": list(assessment.secondary_causes),
            "probe_state_tuple": f"E={enc.state}|R={merged_ret.state}|G={gen.state}",
            "probe_defect_map": {
                "enc": list(enc.defects),
                "ret": list(merged_ret.defects),
                "gen": list(gen.defects),
            },
            "healthy_path": enc.state == "EXIST" and merged_ret.state == "HIT" and gen.state == "PASS",
            "attribution_path": self._attribution_path(assessment.primary_cause, assessment.secondary_causes),
            "decision_logic": self._decision_logic(assessment, enc, merged_ret, gen),
            "decision_trace": list(assessment.decision_trace),
            "summary": assessment.summary,
            "final_judgement": self._final_judgement(gen),
            "llm_payload": dict(assessment.llm_payload),
        }
        return AttributionResult(
            question_id=sample.question_id,
            sample_id=sample.sample_id,
            task_type=sample.task_type,
            states={"enc": enc.state, "ret": merged_ret.state, "gen": gen.state},
            defects=merged_defects,
            probe_results=probe_results,
            attribution_evidence={
                "final_attribution": final_attribution,
                "probe_state_tuple": final_attribution["probe_state_tuple"],
                "probe_defect_map": final_attribution["probe_defect_map"],
            },
        )

    def _build_assessment(
        self,
        sample: EvalSample,
        enc: ProbeResult,
        ret: ProbeResult,
        gen: ProbeResult,
        decision_trace: List[str],
        cfg: Optional[EvaluatorConfig] = None,
    ) -> AttributionAssessment:
        primary_cause = self._primary_cause(enc, ret, gen)
        secondary_causes = [cause for cause in ("encoding", "retrieval", "generation") if cause != primary_cause and self._layer_has_issue(cause, enc, ret, gen)]
        summary = f"primary={primary_cause}; enc={enc.state}; ret={ret.state}; gen={gen.state}"
        llm_payload: Dict[str, object] = {}
        if cfg and cfg.use_llm_assist:
            llm_result = llm_judge_attribution(
                LLMAssistConfig(
                    api_key=cfg.llm_api_key,
                    base_url=cfg.llm_base_url,
                    model=cfg.llm_model,
                    temperature=cfg.llm_temperature,
                ),
                task_type=sample.task_type,
                query=sample.question,
                answer_gold=sample.answer_gold,
                enc_summary={"state": enc.state, "defects": list(enc.defects), "reason": enc.evidence.get("reason", "")},
                ret_summary={"state": ret.state, "defects": list(ret.defects), "reason": ret.evidence.get("reason", "")},
                gen_summary={"state": gen.state, "defects": list(gen.defects), "reason": gen.evidence.get("reason", "")},
                must_succeed=False,
            )
            if isinstance(llm_result, dict):
                candidate_primary = str(llm_result.get("primary_cause", "")).strip().lower()
                if candidate_primary in {"encoding", "retrieval", "generation", "none"}:
                    primary_cause = candidate_primary
                secondary_llm = [str(x).strip().lower() for x in llm_result.get("secondary_causes", []) if str(x).strip()]
                secondary_causes = [x for x in secondary_llm if x in {"encoding", "retrieval", "generation"} and x != primary_cause]
                decision_trace.extend([str(x) for x in llm_result.get("decision_trace", []) if str(x).strip()])
                defect_union = [str(x).strip().upper() for x in llm_result.get("defect_union", []) if str(x).strip()]
                if defect_union:
                    decision_trace.append(f"LLM defect union suggestion: {', '.join(defect_union)}")
                summary = str(llm_result.get("summary", summary))
                llm_payload = dict(llm_result)
        return AttributionAssessment(
            primary_cause=primary_cause,
            secondary_causes=secondary_causes,
            decision_trace=list(decision_trace),
            summary=summary,
            llm_payload=llm_payload,
        )

    def _primary_cause(self, enc: ProbeResult, ret: ProbeResult, gen: ProbeResult) -> str:
        if enc.state in {"MISS", "CORRUPT_AMBIG", "CORRUPT_WRONG", "DIRTY"}:
            return "encoding"
        if ret.state in {"MISS", "NOISE"} or ret.defects:
            return "retrieval"
        if gen.state == "FAIL" or gen.defects:
            return "generation"
        return "none"

    def _layer_has_issue(self, layer: str, enc: ProbeResult, ret: ProbeResult, gen: ProbeResult) -> bool:
        if layer == "encoding":
            return enc.state in {"MISS", "CORRUPT_AMBIG", "CORRUPT_WRONG", "DIRTY"} or bool(enc.defects)
        if layer == "retrieval":
            return ret.state in {"MISS", "NOISE"} or bool(ret.defects)
        if layer == "generation":
            return gen.state == "FAIL" or bool(gen.defects)
        return False

    def _decision_logic(
        self,
        assessment: AttributionAssessment,
        enc: ProbeResult,
        ret: ProbeResult,
        gen: ProbeResult,
    ) -> List[str]:
        logic: List[str] = []
        if enc.state != "EXIST":
            logic.append(f"编码层状态为 {enc.state}，未形成稳定可用的目标记忆。")
        if ret.state != "HIT" or ret.defects:
            logic.append(f"检索层状态为 {ret.state}，缺陷为 {', '.join(ret.defects) or '无'}。")
        if gen.state != "PASS" or gen.defects:
            logic.append(f"生成层状态为 {gen.state}，缺陷为 {', '.join(gen.defects) or '无'}。")
        logic.extend([item for item in assessment.decision_trace if item])
        if not logic:
            logic.append("三层探针均未发现显著问题。")
        return logic[:4]

    def _final_judgement(self, gen: ProbeResult) -> str:
        online_correct = bool(((gen.evidence or {}).get("online_correctness") or {}).get("final_correct", False))
        oracle_correct = bool(((gen.evidence or {}).get("oracle_correctness") or {}).get("final_correct", False))
        if online_correct:
            return "system_answer_correct"
        if oracle_correct:
            return "system_answer_wrong_but_oracle_answer_correct"
        return "oracle_answer_incorrect"

    def _attribution_path(self, primary: str, secondary: List[str]) -> List[str]:
        if primary == "none":
            return ["no_major_failure"]
        path = [primary]
        path.extend([cause for cause in secondary if cause not in path])
        return path
