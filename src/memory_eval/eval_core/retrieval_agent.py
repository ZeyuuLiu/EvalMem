from __future__ import annotations

from typing import Any, Optional

from memory_eval.eval_core.adapter_protocol import RetrievalAdapterProtocol
from memory_eval.eval_core.models import EvalSample, EvaluatorConfig, ProbeResult
from memory_eval.eval_core.retrieval import RetrievalProbeInput, evaluate_retrieval_probe, evaluate_retrieval_probe_with_adapter


class RetrievalAgent:
    name = "RetrievalAgent"

    def evaluate_with_adapter(
        self,
        sample: EvalSample,
        adapter: RetrievalAdapterProtocol,
        run_ctx: Any,
        cfg: EvaluatorConfig,
        top_k: int,
        s_enc: Optional[str] = None,
    ) -> ProbeResult:
        return self._attach_agent_metadata(
            evaluate_retrieval_probe_with_adapter(
                sample=sample,
                adapter=adapter,
                run_ctx=run_ctx,
                cfg=cfg,
                top_k=top_k,
                s_enc=s_enc,
            )
        )

    def evaluate(self, inp: RetrievalProbeInput, cfg: EvaluatorConfig, s_enc: Optional[str] = None) -> ProbeResult:
        return self._attach_agent_metadata(evaluate_retrieval_probe(inp=inp, cfg=cfg, s_enc=s_enc))

    def _attach_agent_metadata(self, result: ProbeResult) -> ProbeResult:
        evidence = dict(result.evidence)
        evidence["agent_name"] = self.name
        return ProbeResult(
            probe=result.probe,
            state=result.state,
            defects=list(result.defects),
            evidence=evidence,
            attrs=dict(result.attrs),
        )
