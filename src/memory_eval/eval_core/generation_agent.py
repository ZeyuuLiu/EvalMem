from __future__ import annotations

from typing import Any

from memory_eval.eval_core.adapter_protocol import GenerationAdapterProtocol
from memory_eval.eval_core.generation import GenerationProbeInput, evaluate_generation_probe, evaluate_generation_probe_with_adapter
from memory_eval.eval_core.models import EvalSample, EvaluatorConfig, ProbeResult


class GenerationAgent:
    name = "GenerationAgent"

    def evaluate_with_adapter(
        self,
        sample: EvalSample,
        adapter: GenerationAdapterProtocol,
        run_ctx: Any,
        cfg: EvaluatorConfig,
    ) -> ProbeResult:
        return self._attach_agent_metadata(
            evaluate_generation_probe_with_adapter(
                sample=sample,
                adapter=adapter,
                run_ctx=run_ctx,
                cfg=cfg,
            )
        )

    def evaluate(self, inp: GenerationProbeInput, cfg: EvaluatorConfig) -> ProbeResult:
        return self._attach_agent_metadata(evaluate_generation_probe(inp=inp, cfg=cfg))

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
