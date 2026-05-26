from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from memory_eval.eval_core.adapter_protocol import EncodingAdapterProtocol, GenerationAdapterProtocol, RetrievalAdapterProtocol
from memory_eval.eval_core.attribution_agent import AttributionAgent
from memory_eval.eval_core.encoding import EncodingProbeInput
from memory_eval.eval_core.encoding_agent import EncodingAgent
from memory_eval.eval_core.generation import GenerationProbeInput
from memory_eval.eval_core.generation_agent import GenerationAgent
from memory_eval.eval_core.models import AdapterTrace, AttributionResult, EvalSample, EvaluatorConfig
from memory_eval.eval_core.retrieval import RetrievalProbeInput
from memory_eval.eval_core.retrieval_agent import RetrievalAgent


class ParallelThreeProbeEvaluator:
    """
    Three-probe evaluator with full parallel execution.
    三探针并行评估器：编码/检索/生成并行运行，提高吞吐效率。
    """

    def __init__(self, config: EvaluatorConfig | None = None):
        self.config = config or EvaluatorConfig()
        self.encoding_agent = EncodingAgent()
        self.retrieval_agent = RetrievalAgent()
        self.generation_agent = GenerationAgent()
        self.attribution_agent = AttributionAgent()

    def evaluate(self, sample: EvalSample, trace: AdapterTrace) -> AttributionResult:
        with ThreadPoolExecutor(max_workers=max(1, self.config.max_workers)) as pool:
            f_enc = pool.submit(
                self.encoding_agent.evaluate,
                EncodingProbeInput(
                    question=sample.question,
                    memory_corpus=trace.memory_view,
                    f_key=list(sample.f_key),
                    task_type=sample.task_type,
                    evidence_texts=list(sample.evidence_texts),
                    evidence_with_time=list(sample.evidence_with_time),
                ),
                trace.memory_view,
                self.config,
            )
            f_gen = pool.submit(
                self.generation_agent.evaluate,
                GenerationProbeInput(
                    question=sample.question,
                    oracle_context=sample.oracle_context,
                    answer_online=str(trace.answer_online or ""),
                    answer_oracle=str(trace.answer_oracle or ""),
                    answer_gold=sample.answer_gold,
                    task_type=sample.task_type,
                ),
                self.config,
            )
            enc = f_enc.result()
            f_ret = pool.submit(
                self.retrieval_agent.evaluate,
                RetrievalProbeInput(
                    question=sample.question,
                    retrieved_items=[{"id": item.id, "text": item.text, "score": item.score, "meta": dict(item.meta)} for item in trace.retrieved_items],
                    f_key=list(sample.f_key),
                    task_type=sample.task_type,
                    evidence_texts=list(sample.evidence_with_time or sample.evidence_texts),
                ),
                self.config,
                enc.state if self.config.retrieval_gate_on_encoding else None,
            )
            ret = f_ret.result()
            gen = f_gen.result()
        return self.attribution_agent.attribute(sample, enc, ret, gen, self.config)

    def evaluate_with_adapters(
        self,
        sample: EvalSample,
        run_ctx,
        encoding_adapter: EncodingAdapterProtocol,
        retrieval_adapter: RetrievalAdapterProtocol,
        generation_adapter: GenerationAdapterProtocol,
        top_k: int = 5,
    ) -> AttributionResult:
        with ThreadPoolExecutor(max_workers=max(1, self.config.max_workers)) as pool:
            f_enc = pool.submit(
                self.encoding_agent.evaluate_with_adapter,
                sample,
                encoding_adapter,
                run_ctx,
                self.config,
                retrieval_adapter=retrieval_adapter,
                top_k=top_k,
            )
            f_gen = pool.submit(self.generation_agent.evaluate_with_adapter, sample, generation_adapter, run_ctx, self.config)
            enc = f_enc.result()
            f_ret = pool.submit(
                self.retrieval_agent.evaluate_with_adapter,
                sample,
                retrieval_adapter,
                run_ctx,
                self.config,
                top_k,
                enc.state if self.config.retrieval_gate_on_encoding else None,
            )
            ret = f_ret.result()
            gen = f_gen.result()
        return self.attribution_agent.attribute(sample, enc, ret, gen, self.config)
