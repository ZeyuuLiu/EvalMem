from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from memory_eval.eval_core.adapter_protocol import EncodingAdapterProtocol, RetrievalAdapterProtocol
from memory_eval.eval_core.encoding import EncodingProbeInput
from memory_eval.eval_core.high_recall import HighRecallRequest, HighRecallResponse
from memory_eval.eval_core.llm_assist import LLMAssistConfig, llm_judge_encoding_storage, llm_judge_fact_match
from memory_eval.eval_core.models import (
    CandidateGroup,
    EncodingAssessment,
    EvalSample,
    EvaluatorConfig,
    EvidenceSpec,
    MemoryObservation,
    MemoryObservationBundle,
    ProbeResult,
)
from memory_eval.eval_core.utils import is_strict_llm_probe, looks_ambiguous, normalize_text, text_match


class EncodingAgent:
    name = "EncodingAgent"

    def build_evidence_spec(self, sample: EvalSample) -> EvidenceSpec:
        evidence_with_time = list(sample.evidence_with_time or [])
        evidence_texts = list(sample.evidence_texts or [])
        facts = [str(x).strip() for x in sample.f_key if str(x).strip()]
        fact_units = [{"fact": fact, "required": True, "source": "f_key"} for fact in facts]
        if not fact_units:
            fact_units = [{"fact": text, "required": True, "source": "evidence"} for text in (evidence_with_time or evidence_texts)[:5]]
        return EvidenceSpec(
            query=sample.question,
            task_type=sample.task_type,
            question_id=sample.question_id,
            sample_id=sample.sample_id,
            f_key=facts,
            evidence_texts=evidence_texts,
            evidence_with_time=evidence_with_time,
            oracle_context=sample.oracle_context,
            fact_units=fact_units,
            must_have_constraints=[str(unit.get("fact", "")).strip() for unit in fact_units if str(unit.get("fact", "")).strip()],
            soft_constraints=list(evidence_with_time or evidence_texts)[:5],
            negative_constraints=["no memory should support answering the query"] if sample.task_type == "NEG" else [],
            evidence_priority=[str(unit.get("fact", "")).strip() for unit in fact_units if str(unit.get("fact", "")).strip()],
            normalization_notes=["evidence_with_time preferred over evidence_texts when available"],
        )

    def evaluate_with_adapter(
        self,
        sample: EvalSample,
        adapter: EncodingAdapterProtocol,
        run_ctx: Any,
        cfg: Optional[EvaluatorConfig] = None,
        *,
        retrieval_adapter: Optional[RetrievalAdapterProtocol] = None,
        top_k: Optional[int] = None,
    ) -> ProbeResult:
        evidence_spec = self.build_evidence_spec(sample)
        bundle = self.collect_observations(
            sample=sample,
            adapter=adapter,
            run_ctx=run_ctx,
            cfg=cfg,
            retrieval_adapter=retrieval_adapter,
            top_k=top_k,
        )
        assessment = self.assess_bundle(evidence_spec, bundle, cfg)
        return self.to_probe_result(assessment, evidence_spec, bundle)

    def evaluate(
        self,
        inp: EncodingProbeInput,
        candidate_records: Optional[List[Dict[str, Any]]] = None,
        cfg: Optional[EvaluatorConfig] = None,
    ) -> ProbeResult:
        evidence_spec = EvidenceSpec(
            query=inp.question,
            task_type=inp.task_type,
            f_key=[str(x).strip() for x in inp.f_key if str(x).strip()],
            evidence_texts=list(inp.evidence_texts or []),
            evidence_with_time=list(inp.evidence_with_time or []),
            fact_units=[{"fact": str(x).strip(), "required": True, "source": "f_key"} for x in inp.f_key if str(x).strip()],
            must_have_constraints=[str(x).strip() for x in inp.f_key if str(x).strip()],
            soft_constraints=list(inp.evidence_with_time or inp.evidence_texts or []),
            negative_constraints=["no memory should support answering the query"] if inp.task_type == "NEG" else [],
            evidence_priority=[str(x).strip() for x in inp.f_key if str(x).strip()],
            normalization_notes=["direct probe input"],
        )
        memory_corpus = self._normalize_records(inp.memory_corpus)
        candidates = self._normalize_records(candidate_records or [])
        if not candidates and memory_corpus and not (cfg and cfg.disable_rule_fallback):
            candidates = self._fallback_find_records(inp.question, inp.f_key, memory_corpus)
        full_memory_view = self._records_to_observations(memory_corpus, "full_memory_export", "direct_input")
        native_candidate_view = self._records_to_observations(candidates, "native_candidate", "direct_candidates")
        bundle = MemoryObservationBundle(
            full_memory_view=full_memory_view,
            native_candidate_view=native_candidate_view,
            combined_candidates=self._merge_observations(native_candidate_view, []),
            candidate_groups=self._build_candidate_groups(self._merge_observations(native_candidate_view, []), evidence_spec),
            adapter_manifest={"agent": self.name, "mode": "direct_input"},
            observability_notes=["direct probe input path"],
            coverage_report={
                "has_full_memory_export": bool(full_memory_view),
                "native_candidate_count": len(native_candidate_view),
                "framework_candidate_count": 0,
                "retrieval_shadow_count": 0,
                "used_framework_fallback": False,
            },
        )
        assessment = self.assess_bundle(evidence_spec, bundle, cfg)
        return self.to_probe_result(assessment, evidence_spec, bundle)

    def collect_observations(
        self,
        sample: EvalSample,
        adapter: EncodingAdapterProtocol,
        run_ctx: Any,
        cfg: Optional[EvaluatorConfig] = None,
        *,
        retrieval_adapter: Optional[RetrievalAdapterProtocol] = None,
        top_k: Optional[int] = None,
    ) -> MemoryObservationBundle:
        memory_corpus = self._normalize_records(adapter.export_full_memory(run_ctx))
        full_memory_view = self._records_to_observations(memory_corpus, "full_memory_export", "export_full_memory")
        native_candidate_records: List[Dict[str, Any]] = []
        native_candidate_view: List[MemoryObservation] = []
        framework_candidate_view: List[MemoryObservation] = []
        native_retrieval_shadow: List[MemoryObservation] = []
        observability_notes: List[str] = []
        used_framework_fallback = False
        external_retriever = getattr(adapter, "get_external_high_recall_retriever", lambda: None)()
        if external_retriever is not None:
            try:
                external_response = external_retriever.retrieve(
                    HighRecallRequest(
                        query=sample.question,
                        f_key=list(sample.f_key),
                        evidence_texts=list(sample.evidence_with_time or sample.evidence_texts),
                        memory_corpus=memory_corpus,
                        metadata={
                            "sample_id": sample.sample_id,
                            "question_id": sample.question_id,
                            "task_type": sample.task_type,
                            "run_ctx": run_ctx,
                            "adapter_class": adapter.__class__.__name__,
                        },
                    )
                )
                if isinstance(external_response, HighRecallResponse):
                    external_records = self._normalize_records(external_response.candidates)
                    native_candidate_records = self._merge_records(native_candidate_records, external_records)
                    native_candidate_view.extend(
                        self._records_to_observations(external_records, "native_candidate", "external_high_recall_retriever")
                    )
                    diagnostics = dict(external_response.diagnostics)
                    if diagnostics:
                        observability_notes.append(f"external_high_recall_retriever diagnostics: {diagnostics}")
                else:
                    raise RuntimeError("external high recall retriever must return HighRecallResponse")
            except Exception as exc:
                if cfg and cfg.strict_adapter_call:
                    raise RuntimeError(f"encoding external_high_recall_retriever failed: {exc}") from exc
                observability_notes.append(f"external_high_recall_retriever failed: {exc}")
        hybrid_fn = getattr(adapter, "hybrid_retrieve_candidates", None)
        if not native_candidate_records and callable(hybrid_fn):
            try:
                hybrid_records = self._normalize_records(
                    hybrid_fn(
                        run_ctx=run_ctx,
                        query=sample.question,
                        f_key=list(sample.f_key),
                        evidence_texts=list(sample.evidence_with_time or sample.evidence_texts),
                        top_n=100,
                    )
                )
                native_candidate_records = self._merge_records(native_candidate_records, hybrid_records)
                native_candidate_view.extend(self._records_to_observations(hybrid_records, "native_candidate", "hybrid_retrieve_candidates"))
            except Exception as exc:
                if cfg and cfg.strict_adapter_call:
                    raise RuntimeError(f"encoding hybrid_retrieve_candidates failed: {exc}") from exc
                observability_notes.append(f"hybrid_retrieve_candidates failed: {exc}")
        if not native_candidate_records:
            find_records = self._normalize_records(adapter.find_memory_records(run_ctx, sample.question, sample.f_key, memory_corpus))
            native_candidate_records = self._merge_records(native_candidate_records, find_records)
            native_candidate_view.extend(self._records_to_observations(find_records, "native_candidate", "find_memory_records"))
        if cfg and cfg.encoding_merge_native_retrieval:
            rk = self._encoding_merge_top_k(cfg, top_k)
            ro_fn = getattr(adapter, "retrieve_original", None)
            extra_records: List[Dict[str, Any]] = []
            query_shadow_records: List[Dict[str, Any]] = []
            f_key_shadow_records: List[Dict[str, Any]] = []
            try:
                if callable(ro_fn):
                    query_shadow_records = self._normalize_records(ro_fn(run_ctx, sample.question, rk))
                    f_key_query = self._compose_f_key_query(sample.f_key)
                    if f_key_query:
                        f_key_shadow_records = self._normalize_records(ro_fn(run_ctx, f_key_query, rk))
                elif retrieval_adapter is not None:
                    query_shadow_records = self._normalize_records(retrieval_adapter.retrieve_original(run_ctx, sample.question, rk))
                    f_key_query = self._compose_f_key_query(sample.f_key)
                    if f_key_query:
                        f_key_shadow_records = self._normalize_records(retrieval_adapter.retrieve_original(run_ctx, f_key_query, rk))
            except Exception as exc:
                if cfg.strict_adapter_call:
                    raise RuntimeError(f"encoding merge retrieve_original failed: {exc}") from exc
                observability_notes.append(f"retrieve_original merge failed: {exc}")
            extra_records = self._merge_records(query_shadow_records, f_key_shadow_records)
            if query_shadow_records:
                native_retrieval_shadow.extend(
                    self._records_to_observations(query_shadow_records, "native_retrieval_shadow", "retrieve_original_query")
                )
            if f_key_shadow_records:
                native_retrieval_shadow.extend(
                    self._records_to_observations(f_key_shadow_records, "native_retrieval_shadow", "retrieve_original_f_key")
                )
            if extra_records:
                native_candidate_records = self._merge_records(native_candidate_records, extra_records)
        if not native_candidate_records and memory_corpus and not (cfg and cfg.disable_rule_fallback):
            fallback_records = self._fallback_find_records(sample.question, sample.f_key, memory_corpus)
            framework_candidate_view = self._records_to_observations(fallback_records, "framework_candidate", "fallback_find_records")
            native_candidate_records = self._merge_records(native_candidate_records, fallback_records)
            used_framework_fallback = bool(fallback_records)
        combined_candidates = self._merge_observations(
            self._records_to_observations(native_candidate_records, "native_candidate", "combined_native_candidates"),
            framework_candidate_view + native_retrieval_shadow,
        )
        candidate_groups = self._build_candidate_groups(combined_candidates, self.build_evidence_spec(sample))
        coverage_report = {
            "has_full_memory_export": bool(full_memory_view),
            "full_memory_count": len(full_memory_view),
            "native_candidate_count": len(native_candidate_view),
            "framework_candidate_count": len(framework_candidate_view),
            "retrieval_shadow_count": len(native_retrieval_shadow),
            "query_retrieval_shadow_count": sum(1 for obs in native_retrieval_shadow if obs.source_name == "retrieve_original_query"),
            "f_key_retrieval_shadow_count": sum(1 for obs in native_retrieval_shadow if obs.source_name == "retrieve_original_f_key"),
            "combined_candidate_count": len(combined_candidates),
            "candidate_group_count": len(candidate_groups),
            "used_external_high_recall_retriever": any(
                obs.source_name == "external_high_recall_retriever" for obs in native_candidate_view
            ),
            "used_framework_fallback": used_framework_fallback,
            "used_native_retrieval_shadow": bool(native_retrieval_shadow),
        }
        if not full_memory_view:
            observability_notes.append("full memory export returned empty view")
        if not native_candidate_records:
            observability_notes.append("no encoding candidates were produced")
        return MemoryObservationBundle(
            full_memory_view=full_memory_view,
            native_candidate_view=native_candidate_view,
            framework_candidate_view=framework_candidate_view,
            native_retrieval_shadow=native_retrieval_shadow,
            combined_candidates=combined_candidates,
            candidate_groups=candidate_groups,
            adapter_manifest={"adapter_class": adapter.__class__.__name__, "agent": self.name},
            observability_notes=observability_notes,
            coverage_report=coverage_report,
        )

    def assess_bundle(
        self,
        evidence_spec: EvidenceSpec,
        bundle: MemoryObservationBundle,
        cfg: Optional[EvaluatorConfig] = None,
    ) -> EncodingAssessment:
        candidate_records = [self._observation_to_record(obs) for obs in bundle.combined_candidates]
        for group in bundle.candidate_groups:
            if group.group_type == "single_record":
                continue
            candidate_records.append(
                {
                    "id": group.group_id,
                    "text": group.aggregated_text,
                    "meta": {"group_type": group.group_type, "member_ids": list(group.member_ids), "source_breakdown": list(group.source_breakdown)},
                }
            )
        candidate_records = self._normalize_records(candidate_records)
        strict = is_strict_llm_probe(cfg)
        evidence_texts = list(evidence_spec.evidence_with_time or evidence_spec.evidence_texts)
        if strict:
            if not cfg:
                raise RuntimeError("strict LLM encoding probe requires EvaluatorConfig")
            llm_judgement = llm_judge_encoding_storage(
                LLMAssistConfig(
                    api_key=cfg.llm_api_key,
                    base_url=cfg.llm_base_url,
                    model=cfg.llm_model,
                    temperature=cfg.llm_temperature,
                ),
                query=evidence_spec.query,
                f_key=list(evidence_spec.f_key),
                evidence_texts=evidence_texts,
                candidates=candidate_records,
                task_type=evidence_spec.task_type,
                must_succeed=True,
            )
            if not isinstance(llm_judgement, dict):
                raise RuntimeError("encoding llm judgement failed or empty")
            return self._assessment_from_llm(llm_judgement, bundle, candidate_records)
        llm_must = bool(cfg and cfg.use_llm_assist and cfg.require_llm_judgement)
        llm_judgement: Dict[str, Any] | None = None
        if cfg and cfg.use_llm_assist:
            llm_judgement = llm_judge_encoding_storage(
                LLMAssistConfig(
                    api_key=cfg.llm_api_key,
                    base_url=cfg.llm_base_url,
                    model=cfg.llm_model,
                    temperature=cfg.llm_temperature,
                ),
                query=evidence_spec.query,
                f_key=list(evidence_spec.f_key),
                evidence_texts=evidence_texts,
                candidates=candidate_records,
                task_type=evidence_spec.task_type,
                must_succeed=llm_must,
            )
            if cfg.require_llm_judgement and not isinstance(llm_judgement, dict):
                raise RuntimeError("encoding llm judgement failed or empty")
        if isinstance(llm_judgement, dict):
            state = str(llm_judgement.get("encoding_state", "")).upper()
            if state in {"EXIST", "MISS", "CORRUPT_AMBIG", "CORRUPT_WRONG", "DIRTY"}:
                return self._assessment_from_llm(llm_judgement, bundle, candidate_records)
            if cfg and cfg.require_llm_judgement and cfg.disable_rule_fallback:
                raise RuntimeError(f"encoding llm judgement returned invalid state: {state or 'EMPTY'}")
        if cfg and cfg.use_llm_assist and cfg.disable_rule_fallback:
            raise RuntimeError("encoding strict mode rejects rule fallback")
        return self._assessment_from_rules(evidence_spec, candidate_records, bundle, cfg)

    def to_probe_result(
        self,
        assessment: EncodingAssessment,
        evidence_spec: EvidenceSpec,
        bundle: MemoryObservationBundle,
    ) -> ProbeResult:
        evidence = {
            "agent_name": self.name,
            "reason": assessment.reasoning_chain[0] if assessment.reasoning_chain else "",
            "reasoning_chain": list(assessment.reasoning_chain),
            "evidence_spec": {
                "query": evidence_spec.query,
                "task_type": evidence_spec.task_type,
                "f_key": list(evidence_spec.f_key),
                "must_have_constraints": list(evidence_spec.must_have_constraints),
            },
            "matched_candidate_ids": list(assessment.matched_ids),
            "supporting_snippets": list(assessment.supporting_snippets),
            "contradicting_snippets": list(assessment.contradicting_snippets),
            "missing_fact_units": list(assessment.missing_fact_units),
            "ambiguity_hits": list(assessment.ambiguous_fact_units),
            "coverage_report": dict(assessment.coverage_report),
            "observability_notes": list(bundle.observability_notes),
            "candidate_group_count": len(bundle.candidate_groups),
            "debug_payload": dict(assessment.debug_payload),
        }
        attrs = {
            "confidence": float(assessment.confidence),
            "evidence_found_by": assessment.evidence_found_by,
            "risk_flags": list(assessment.risk_flags),
        }
        return ProbeResult(probe="enc", state=assessment.state, defects=list(assessment.defects), evidence=evidence, attrs=attrs)

    def _assessment_from_llm(
        self,
        llm_judgement: Dict[str, Any],
        bundle: MemoryObservationBundle,
        candidate_records: List[Dict[str, Any]],
    ) -> EncodingAssessment:
        state = str(llm_judgement.get("encoding_state", "")).upper()
        valid = {"EXIST", "MISS", "CORRUPT_AMBIG", "CORRUPT_WRONG", "DIRTY"}
        if state not in valid:
            raise RuntimeError(f"encoding llm judgement returned invalid state: {state or 'EMPTY'}")
        matched_ids = [str(x) for x in llm_judgement.get("matched_candidate_ids", []) if str(x).strip()]
        supporting_snippets = [str(x) for x in llm_judgement.get("evidence_snippets", []) if str(x).strip()]
        evidence_found_by = "semantic_equivalence"
        if matched_ids:
            by_id = {str(r.get("id", "")): r for r in candidate_records}
            if any(str(by_id.get(mid, {}).get("meta", {}).get("group_type", "")) for mid in matched_ids):
                evidence_found_by = "record_combination"
            else:
                evidence_found_by = "single_record"
        if state == "DIRTY":
            evidence_found_by = "dirty_memory_detected"
        if state == "MISS":
            evidence_found_by = "not_found"
        return EncodingAssessment(
            state=state,
            defects=[str(x).upper() for x in llm_judgement.get("defects", []) if str(x).strip()],
            confidence=float(llm_judgement.get("confidence", 0.0) or 0.0),
            matched_ids=matched_ids,
            supporting_snippets=supporting_snippets,
            contradicting_snippets=[],
            missing_fact_units=[],
            ambiguous_fact_units=[],
            coverage_report=dict(bundle.coverage_report),
            reasoning_chain=[str(llm_judgement.get("reasoning", "LLM holistic encoding judgement."))],
            evidence_found_by=evidence_found_by,
            risk_flags=["partial_observability"] if bundle.observability_notes else [],
            debug_payload={"llm_encoding_judgement": llm_judgement},
        )

    def _assessment_from_rules(
        self,
        evidence_spec: EvidenceSpec,
        candidate_records: List[Dict[str, Any]],
        bundle: MemoryObservationBundle,
        cfg: Optional[EvaluatorConfig] = None,
    ) -> EncodingAssessment:
        if evidence_spec.task_type == "NEG":
            supportive_candidates = [
                record
                for record in candidate_records
                if any(self._fact_match(fact, str(record.get("text", ""))) for fact in evidence_spec.f_key)
            ]
            if supportive_candidates:
                return EncodingAssessment(
                    state="DIRTY",
                    defects=["DMP"],
                    confidence=0.6,
                    matched_ids=[str(r.get("id", "")) for r in supportive_candidates[:10] if str(r.get("id", "")).strip()],
                    supporting_snippets=[str(r.get("text", "")) for r in supportive_candidates[:5] if str(r.get("text", "")).strip()],
                    coverage_report=dict(bundle.coverage_report),
                    reasoning_chain=["NEG sample has memory candidates that directly support prohibited facts."],
                    evidence_found_by="dirty_memory_detected",
                    risk_flags=["rule_fallback"],
                    debug_payload={"candidate_count": len(candidate_records), "supportive_candidate_count": len(supportive_candidates)},
                )
            return EncodingAssessment(
                state="MISS",
                defects=[],
                confidence=0.6,
                coverage_report=dict(bundle.coverage_report),
                reasoning_chain=["NEG sample has no memory candidate that supports prohibited facts."],
                evidence_found_by="not_found",
                risk_flags=["rule_fallback"],
                debug_payload={"candidate_count": len(candidate_records), "supportive_candidate_count": 0},
            )
        facts = list(evidence_spec.f_key or [])
        if not facts:
            return EncodingAssessment(
                state="MISS",
                defects=["EM"],
                confidence=0.5,
                missing_fact_units=["empty_f_key"],
                coverage_report=dict(bundle.coverage_report),
                reasoning_chain=["POS sample has empty f_key."],
                evidence_found_by="not_found",
                risk_flags=["rule_fallback"],
            )
        matched_facts: Dict[str, List[Dict[str, Any]]] = {}
        unmatched_facts: List[str] = []
        ambiguity_hits: List[str] = []
        for fact in facts:
            fact_matches: List[Dict[str, Any]] = []
            for record in candidate_records:
                base_match = self._fact_match(fact, str(record.get("text", "")))
                if cfg and cfg.use_llm_assist and not base_match:
                    lj = llm_judge_fact_match(
                        LLMAssistConfig(
                            api_key=cfg.llm_api_key,
                            base_url=cfg.llm_base_url,
                            model=cfg.llm_model,
                            temperature=cfg.llm_temperature,
                        ),
                        question=evidence_spec.query,
                        fact=fact,
                        candidate_text=str(record.get("text", "")),
                    )
                    if isinstance(lj, dict):
                        base_match = bool(lj.get("match", False))
                if base_match:
                    fact_matches.append(record)
                    if looks_ambiguous(str(record.get("text", ""))):
                        ambiguity_hits.append(str(record.get("text", "")))
            if fact_matches:
                matched_facts[fact] = fact_matches
            else:
                unmatched_facts.append(fact)
        if not matched_facts:
            return EncodingAssessment(
                state="MISS",
                defects=["EM"],
                confidence=0.5,
                missing_fact_units=list(unmatched_facts),
                coverage_report=dict(bundle.coverage_report),
                reasoning_chain=["No key-fact match in candidate records."],
                evidence_found_by="not_found",
                risk_flags=["rule_fallback"],
                debug_payload={"candidate_count": len(candidate_records)},
            )
        matched_ids = []
        supporting_snippets = []
        for records in matched_facts.values():
            for record in records:
                rid = str(record.get("id", ""))
                if rid and rid not in matched_ids:
                    matched_ids.append(rid)
                txt = str(record.get("text", ""))
                if txt and txt not in supporting_snippets:
                    supporting_snippets.append(txt)
        if unmatched_facts:
            if ambiguity_hits:
                return EncodingAssessment(
                    state="CORRUPT_AMBIG",
                    defects=["EA"],
                    confidence=0.55,
                    matched_ids=matched_ids,
                    supporting_snippets=supporting_snippets[:5],
                    missing_fact_units=list(unmatched_facts),
                    ambiguous_fact_units=list(ambiguity_hits[:10]),
                    coverage_report=dict(bundle.coverage_report),
                    reasoning_chain=["Partial key-fact match with ambiguous memory text."],
                    evidence_found_by="single_record",
                    risk_flags=["rule_fallback"],
                )
            return EncodingAssessment(
                state="CORRUPT_WRONG",
                defects=["EW"],
                confidence=0.55,
                matched_ids=matched_ids,
                supporting_snippets=supporting_snippets[:5],
                missing_fact_units=list(unmatched_facts),
                coverage_report=dict(bundle.coverage_report),
                reasoning_chain=["Partial key-fact match without ambiguity, treated as wrong or corrupt."],
                evidence_found_by="single_record",
                risk_flags=["rule_fallback"],
            )
        evidence_found_by = "record_combination" if any(mid.startswith("cg-") for mid in matched_ids) else "single_record"
        return EncodingAssessment(
            state="EXIST",
            defects=[],
            confidence=0.7,
            matched_ids=matched_ids,
            supporting_snippets=supporting_snippets[:5],
            coverage_report=dict(bundle.coverage_report),
            reasoning_chain=["All key facts matched in candidate records."],
            evidence_found_by=evidence_found_by,
            risk_flags=["rule_fallback"],
        )

    def _normalize_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for record in records:
            out.append(
                {
                    "id": str(record.get("id", "")),
                    "text": str(record.get("text", "")),
                    "score": float(record.get("score", 0.0) or 0.0),
                    "meta": dict(record.get("meta", {})) if isinstance(record.get("meta", {}), dict) else {},
                }
            )
        return out

    def _records_to_observations(self, records: List[Dict[str, Any]], source_type: str, source_name: str) -> List[MemoryObservation]:
        out: List[MemoryObservation] = []
        for index, record in enumerate(records):
            meta = dict(record.get("meta", {})) if isinstance(record.get("meta", {}), dict) else {}
            out.append(
                MemoryObservation(
                    memory_id=str(record.get("id", f"{source_type}-{index}")),
                    text=str(record.get("text", "")),
                    normalized_text=normalize_text(str(record.get("text", ""))),
                    source_type=source_type,
                    source_name=source_name,
                    storage_kind=str(meta.get("storage_kind", meta.get("channel", ""))),
                    speaker=str(meta.get("speaker", "")),
                    timestamp=str(meta.get("timestamp", "")),
                    session_id=str(meta.get("session_id", meta.get("session", ""))),
                    score=float(record.get("score", 0.0) or 0.0),
                    meta=meta,
                    raw_payload_ref=str(meta.get("raw_payload_ref", "")),
                )
            )
        return out

    def _observation_to_record(self, observation: MemoryObservation) -> Dict[str, Any]:
        return {
            "id": observation.memory_id,
            "text": observation.text,
            "score": observation.score,
            "meta": {
                **dict(observation.meta),
                "source_type": observation.source_type,
                "source_name": observation.source_name,
                "storage_kind": observation.storage_kind,
                "speaker": observation.speaker,
                "timestamp": observation.timestamp,
                "session_id": observation.session_id,
            },
        }

    def _merge_records(self, primary: List[Dict[str, Any]], secondary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_norm: set[str] = set()
        out: List[Dict[str, Any]] = []
        for bucket in (primary, secondary):
            for record in bucket:
                norm = normalize_text(str(record.get("text", "")))[:800]
                if norm and norm in seen_norm:
                    continue
                if norm:
                    seen_norm.add(norm)
                out.append(record)
        return out

    def _merge_observations(self, primary: List[MemoryObservation], secondary: List[MemoryObservation]) -> List[MemoryObservation]:
        seen_norm: set[str] = set()
        out: List[MemoryObservation] = []
        for bucket in (primary, secondary):
            for observation in bucket:
                norm = observation.normalized_text[:800]
                if norm and norm in seen_norm:
                    continue
                if norm:
                    seen_norm.add(norm)
                out.append(observation)
        return out

    def _build_candidate_groups(self, observations: List[MemoryObservation], evidence_spec: EvidenceSpec) -> List[CandidateGroup]:
        groups: List[CandidateGroup] = []
        for observation in observations:
            groups.append(
                CandidateGroup(
                    group_id=f"cg-single-{observation.memory_id}",
                    member_ids=[observation.memory_id],
                    group_type="single_record",
                    aggregated_text=observation.text,
                    supporting_slots=list(evidence_spec.must_have_constraints[:3]),
                    source_breakdown=[observation.source_name or observation.source_type],
                    confidence_hint=observation.score,
                )
            )
        top = [obs for obs in observations if obs.text][:3]
        if len(top) >= 2:
            groups.append(
                CandidateGroup(
                    group_id="cg-composed-top3",
                    member_ids=[obs.memory_id for obs in top],
                    group_type="cross_record_composition",
                    aggregated_text="\n".join(obs.text for obs in top),
                    supporting_slots=list(evidence_spec.must_have_constraints[:5]),
                    source_breakdown=[obs.source_name or obs.source_type for obs in top],
                    confidence_hint=max(float(obs.score) for obs in top),
                )
            )
        return groups

    def _encoding_merge_top_k(self, cfg: Optional[EvaluatorConfig], top_k: Optional[int]) -> int:
        base = int(cfg.encoding_native_retrieval_top_k) if cfg else 20
        if top_k is not None:
            return max(base, int(top_k))
        return base

    def _fallback_find_records(self, question: str, f_key: List[str], memory_corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for memory in memory_corpus:
            text = str(memory.get("text", ""))
            q_hit = text_match(question, text)
            f_hit = any(text_match(fact, text) for fact in f_key if fact)
            if q_hit or f_hit:
                candidates.append(memory)
        return candidates

    def _fact_match(self, fact: str, text: str) -> bool:
        norm_fact = normalize_text(fact)
        norm_text = normalize_text(text)
        if not norm_fact or not norm_text:
            return False
        if len(norm_fact) <= 3:
            tokens = re.findall(r"[\w\u4e00-\u9fff]+", norm_text)
            return norm_fact in tokens
        return norm_fact == norm_text or norm_fact in norm_text

    def _compose_f_key_query(self, f_key: List[str]) -> str:
        chunks = [str(item).strip() for item in f_key if str(item).strip()]
        if not chunks:
            return ""
        return " ; ".join(chunks[:5])
