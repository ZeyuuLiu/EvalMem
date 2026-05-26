from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any, Dict, List, Optional

from memory_eval.eval_core.adapter_protocol import EncodingAdapterProtocol, RetrievalAdapterProtocol
from memory_eval.eval_core.models import EvalSample, EvaluatorConfig, ProbeResult


@dataclass(frozen=True)
class EncodingProbeInput:
    question: str
    memory_corpus: List[Dict[str, Any]]
    f_key: List[str]
    task_type: str
    evidence_texts: List[str] = field(default_factory=list)
    evidence_with_time: List[str] = field(default_factory=list)


def evaluate_encoding_probe_with_adapter(
    sample: EvalSample,
    adapter: EncodingAdapterProtocol,
    run_ctx: Any,
    cfg: Optional[EvaluatorConfig] = None,
    *,
    retrieval_adapter: Optional[RetrievalAdapterProtocol] = None,
    top_k: Optional[int] = None,
) -> ProbeResult:
    from memory_eval.eval_core.encoding_agent import EncodingAgent

    return EncodingAgent().evaluate_with_adapter(
        sample=sample,
        adapter=adapter,
        run_ctx=run_ctx,
        cfg=cfg,
        retrieval_adapter=retrieval_adapter,
        top_k=top_k,
    )


def evaluate_encoding_probe(
    inp: EncodingProbeInput,
    candidate_records: Optional[List[Dict[str, Any]]] = None,
    cfg: Optional[EvaluatorConfig] = None,
) -> ProbeResult:
    from memory_eval.eval_core.encoding_agent import EncodingAgent

    return EncodingAgent().evaluate(inp=inp, candidate_records=candidate_records, cfg=cfg)
