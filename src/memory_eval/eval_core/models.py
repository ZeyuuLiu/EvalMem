from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# -----------------------------
# Core states / 缺陷状态定义
# -----------------------------
ENC_STATES = {"EXIST", "MISS", "CORRUPT_AMBIG", "CORRUPT_WRONG", "DIRTY"}
RET_STATES = {"HIT", "MISS", "NOISE"}
GEN_STATES = {"PASS", "FAIL"}

DEFECT_ORDER = ["EM", "EA", "EW", "DMP", "RF", "LATE", "NOI", "NIR", "GH", "GF", "GRF"]


@dataclass(frozen=True)
class EvalSample:
    """
    Unified evaluation sample contract.
    统一评估样本契约，评估层只依赖这里的字段。
    """

    sample_id: str
    question_id: str
    question: str
    answer_gold: str
    task_type: str  # POS | NEG
    f_key: List[str]
    oracle_context: str
    category: int = 0  # LoCoMo category: 1=multi-hop, 2=temporal, 3=open, 4=single-hop, 5=adversarial
    evidence_ids: List[str] = field(default_factory=list)
    evidence_texts: List[str] = field(default_factory=list)
    evidence_with_time: List[str] = field(default_factory=list)
    construction_evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievedItem:
    """
    Adapter-normalized retrieval item.
    适配器标准化后的检索条目。
    """

    id: str
    text: str
    score: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterTrace:
    """
    Inputs from adapter layer to evaluator.
    适配器层提供给评估层的统一观测输入。
    """

    memory_view: List[Dict[str, Any]]
    retrieved_items: List[RetrievedItem]
    answer_online: str = ""
    answer_oracle: str = ""
    raw_trace: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceSpec:
    query: str
    task_type: str
    question_id: str = ""
    sample_id: str = ""
    f_key: List[str] = field(default_factory=list)
    evidence_texts: List[str] = field(default_factory=list)
    evidence_with_time: List[str] = field(default_factory=list)
    oracle_context: str = ""
    fact_units: List[Dict[str, Any]] = field(default_factory=list)
    must_have_constraints: List[str] = field(default_factory=list)
    soft_constraints: List[str] = field(default_factory=list)
    negative_constraints: List[str] = field(default_factory=list)
    evidence_priority: List[str] = field(default_factory=list)
    normalization_notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class MemoryObservation:
    memory_id: str
    text: str
    normalized_text: str = ""
    source_type: str = ""
    source_name: str = ""
    storage_kind: str = ""
    speaker: str = ""
    timestamp: str = ""
    session_id: str = ""
    score: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)
    raw_payload_ref: str = ""


@dataclass(frozen=True)
class CandidateGroup:
    group_id: str
    member_ids: List[str]
    group_type: str
    aggregated_text: str
    supporting_slots: List[str] = field(default_factory=list)
    source_breakdown: List[str] = field(default_factory=list)
    confidence_hint: float = 0.0


@dataclass(frozen=True)
class MemoryObservationBundle:
    full_memory_view: List[MemoryObservation] = field(default_factory=list)
    native_candidate_view: List[MemoryObservation] = field(default_factory=list)
    framework_candidate_view: List[MemoryObservation] = field(default_factory=list)
    native_retrieval_shadow: List[MemoryObservation] = field(default_factory=list)
    combined_candidates: List[MemoryObservation] = field(default_factory=list)
    candidate_groups: List[CandidateGroup] = field(default_factory=list)
    adapter_manifest: Dict[str, Any] = field(default_factory=dict)
    observability_notes: List[str] = field(default_factory=list)
    coverage_report: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EncodingAssessment:
    state: str
    defects: List[str]
    confidence: float = 0.0
    matched_ids: List[str] = field(default_factory=list)
    supporting_snippets: List[str] = field(default_factory=list)
    contradicting_snippets: List[str] = field(default_factory=list)
    missing_fact_units: List[str] = field(default_factory=list)
    ambiguous_fact_units: List[str] = field(default_factory=list)
    coverage_report: Dict[str, Any] = field(default_factory=dict)
    reasoning_chain: List[str] = field(default_factory=list)
    evidence_found_by: str = "not_found"
    risk_flags: List[str] = field(default_factory=list)
    debug_payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AttributionAssessment:
    primary_cause: str
    secondary_causes: List[str] = field(default_factory=list)
    decision_trace: List[str] = field(default_factory=list)
    summary: str = ""
    llm_payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProbeResult:
    """
    Generic probe result.
    通用探针输出：状态 + 缺陷 + 证据 + 属性。
    """

    probe: str  # enc | ret | gen
    state: str
    defects: List[str]
    evidence: Dict[str, Any] = field(default_factory=dict)
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AttributionResult:
    """
    Final per-sample attribution output.
    单样本最终归因输出，包含三层证据。
    """

    question_id: str
    sample_id: str
    task_type: str
    states: Dict[str, str]
    defects: List[str]
    probe_results: Dict[str, ProbeResult]
    attribution_evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "sample_id": self.sample_id,
            "task_type": self.task_type,
            "states": dict(self.states),
            "defects": list(self.defects),
            "probe_results": {k: asdict(v) for k, v in self.probe_results.items()},
            "attribution_evidence": dict(self.attribution_evidence),
        }


@dataclass(frozen=True)
class EvaluatorConfig:
    """
    Rule-mode evaluator configuration.
    规则模式评估器配置。
    """

    tau_rank: int = 5
    tau_snr: float = 0.2
    neg_noise_score_threshold: float = 0.15
    max_workers: int = 3  # parallel probes / 三探针并行线程数
    retrieval_gate_on_encoding: bool = True
    # Run encoding before final retrieval judgement so RF is gated by S_enc.
    # 编码状态先行用于约束 RF 归因；生成探针仍可并行执行。
    # LLM-assisted probe judgments are enabled by default.
    # 默认评测采用 LLM 主判与严格模式。
    use_llm_assist: bool = True
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    correctness_use_llm_judge: bool = False
    correctness_require_llm_judge: bool = False
    # Strict execution policy is enabled by default.
    # 默认要求 LLM 判定成功，不回退到规则。
    require_llm_judgement: bool = True
    strict_adapter_call: bool = True
    disable_rule_fallback: bool = True
    require_online_answer: bool = False
    # Encoding: merge native retrieval into LLM candidate set (observation alignment).
    encoding_merge_native_retrieval: bool = True
    encoding_native_retrieval_top_k: int = 20
