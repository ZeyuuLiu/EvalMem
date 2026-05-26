from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol

from memory_eval.adapters.registry import export_adapter_runtime_manifest
from memory_eval.dataset.locomo_builder import build_locomo_eval_samples
from memory_eval.eval_core.engine import ParallelThreeProbeEvaluator
from memory_eval.eval_core.models import AttributionResult, EvalSample, EvaluatorConfig
from memory_eval.memwiki.config import MemWikiConfig
from memory_eval.memwiki.injector import MemWikiInjector


def _redact_secrets(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for key, value in obj.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("api_key", "apikey", "token", "secret", "password")):
                out[key] = "***REDACTED***" if value else value
            else:
                out[key] = _redact_secrets(value)
        return out
    if isinstance(obj, list):
        return [_redact_secrets(x) for x in obj]
    return obj


class FullEvalAdapterProtocol(Protocol):
    """
    Full adapter contract used by end-to-end evaluation pipeline.
    端到端评估流水线所需的完整适配器协议。
    """

    def ingest_conversation(self, sample_id: str, conversation: List[Dict[str, Any]]) -> Any:
        ...

    def export_full_memory(self, run_ctx: Any) -> List[Dict[str, Any]]:
        ...

    def find_memory_records(
        self,
        run_ctx: Any,
        query: str,
        f_key: List[str],
        memory_corpus: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ...

    def retrieve_original(self, run_ctx: Any, query: str, top_k: int) -> List[Dict[str, Any]]:
        ...

    def generate_oracle_answer(self, run_ctx: Any, query: str, oracle_context: str) -> str:
        ...

    def export_build_artifact(self, run_ctx: Any) -> Dict[str, Any]:
        ...

    def load_build_artifact(self, manifest: Dict[str, Any]) -> Any:
        ...


@dataclass(frozen=True)
class PipelineConfig:
    dataset_path: str
    output_path: str
    top_k: int = 5
    limit: int | None = None
    f_key_mode: str = "rule"
    evaluator_config: EvaluatorConfig = EvaluatorConfig()
    build_manifest_path: str = ""
    use_memwiki: bool = False
    memwiki_output_root: str = ""
    memwiki_max_aux_records: int | None = None
    memwiki_disable_llm: bool = False
    memwiki_llm_record_limit: int | None = None
    memwiki_llm_synthesis_limit: int | None = None


def _load_episode_map(dataset_path: str) -> Dict[str, Dict[str, Any]]:
    path = Path(dataset_path)
    with path.open("r", encoding="utf-8") as f:
        episodes = json.load(f)
    out: Dict[str, Dict[str, Any]] = {}
    for ep in episodes:
        sid = str(ep.get("sample_id", "")).strip()
        if sid:
            out[sid] = ep
    return out


def _build_summary(results: List[AttributionResult]) -> Dict[str, Any]:
    defect_counts: Dict[str, int] = {}
    task_counts: Dict[str, int] = {"POS": 0, "NEG": 0}
    state_counts: Dict[str, Dict[str, int]] = {"enc": {}, "ret": {}, "gen": {}}

    for r in results:
        task_counts[r.task_type] = task_counts.get(r.task_type, 0) + 1
        for d in r.defects:
            defect_counts[d] = defect_counts.get(d, 0) + 1
        for probe in ("enc", "ret", "gen"):
            s = r.states.get(probe, "UNKNOWN")
            probe_map = state_counts[probe]
            probe_map[s] = probe_map.get(s, 0) + 1

    return {
        "total": len(results),
        "ok": len(results),
        "errors": 0,
        "task_counts": task_counts,
        "defect_counts": defect_counts,
        "state_counts": state_counts,
    }


def _safe_segment(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text or "").strip())
    return cleaned.strip("_") or "unknown"


def _resolve_output_layout(output_path: str) -> tuple[Path, Path]:
    out_path = Path(output_path)
    if out_path.suffix.lower() == ".json":
        return out_path, out_path.with_suffix("")
    return out_path / "result_bundle.json", out_path


def _load_build_artifact_map(path: str) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    artifacts = raw.get("artifacts", raw if isinstance(raw, list) else [])
    out: Dict[str, Dict[str, Any]] = {}
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        sample_id = str(item.get("sample_id", "")).strip()
        if sample_id:
            out[sample_id] = item
    return out


def _extract_generation_correctness(result: AttributionResult) -> Dict[str, Any]:
    gen = result.probe_results.get("gen")
    evidence = dict(gen.evidence) if gen else {}
    return {
        "online": dict(evidence.get("online_correctness", {})),
        "oracle": dict(evidence.get("oracle_correctness", {})),
        "answer_online": str(evidence.get("answer_online", "")),
        "answer_oracle": str(evidence.get("answer_oracle", "")),
    }


def _extract_artifact_refs(adapter: Any, run_ctx: Any) -> Dict[str, Any]:
    refs: Dict[str, Any] = {}
    export_fn = getattr(adapter, "export_build_artifact", None)
    if callable(export_fn):
        try:
            refs = dict(export_fn(run_ctx))
        except Exception:
            refs = {}
    if isinstance(run_ctx, dict):
        refs.setdefault("sample_id", str(run_ctx.get("sample_id", "")))
        refs.setdefault("run_id", str(run_ctx.get("run_id", "")))
        refs.setdefault("output_root", str(run_ctx.get("output_root", "")))
        refs.setdefault("raw_data_path", str(run_ctx.get("raw_data_path", "")))
        refs.setdefault("config_snapshot", dict(run_ctx.get("config_snapshot", {})))
        if run_ctx.get("memwiki_last_report"):
            refs["memwiki_last_report"] = dict(run_ctx.get("memwiki_last_report", {}))
        if run_ctx.get("memwiki_aux_record_count") is not None:
            refs["memwiki_aux_record_count"] = int(run_ctx.get("memwiki_aux_record_count", 0) or 0)
    return refs


def _build_question_record(sample: EvalSample, result: AttributionResult, run_ctx: Any, adapter: Any) -> Dict[str, Any]:
    generation_correctness = _extract_generation_correctness(result)
    final_attribution = dict(result.attribution_evidence.get("final_attribution", {}))
    return {
        "question_id": sample.question_id,
        "sample_id": sample.sample_id,
        "task_type": sample.task_type,
        "question": sample.question,
        "answer_gold": sample.answer_gold,
        "answer_online": generation_correctness.get("answer_online", ""),
        "answer_oracle": generation_correctness.get("answer_oracle", ""),
        "generation_correctness": {
            "online": generation_correctness.get("online", {}),
            "oracle": generation_correctness.get("oracle", {}),
        },
        "probe_states": dict(result.states),
        "probe_defects": {probe: list(probe_result.defects) for probe, probe_result in result.probe_results.items()},
        "probe_results": {probe: probe_result.evidence for probe, probe_result in result.probe_results.items()},
        "final_attribution": final_attribution,
        "decision_logic": list(final_attribution.get("decision_logic", [])),
        "artifact_refs": _extract_artifact_refs(adapter, run_ctx),
    }


def _write_question_record(run_dir: Path, record: Dict[str, Any]) -> str:
    sample_dir = run_dir / _safe_segment(str(record.get("sample_id", "")))
    sample_dir.mkdir(parents=True, exist_ok=True)
    rel_path = Path(_safe_segment(str(record.get("sample_id", "")))) / f"{_safe_segment(str(record.get('question_id', '')))}.json"
    target = run_dir / rel_path
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(rel_path)


def _augment_summary(summary: Dict[str, Any], question_index: List[Dict[str, Any]]) -> Dict[str, Any]:
    final_correct = 0
    pos_total = 0
    pos_correct = 0
    neg_total = 0
    neg_correct = 0
    for item in question_index:
        is_correct = bool(item.get("final_correct", False))
        final_correct += int(is_correct)
        if item.get("task_type") == "POS":
            pos_total += 1
            pos_correct += int(is_correct)
        elif item.get("task_type") == "NEG":
            neg_total += 1
            neg_correct += int(is_correct)
    summary["final_correct"] = final_correct
    summary["final_accuracy"] = (final_correct / len(question_index)) if question_index else 0.0
    summary["pos_final_accuracy"] = (pos_correct / pos_total) if pos_total else 0.0
    summary["neg_final_accuracy"] = (neg_correct / neg_total) if neg_total else 0.0
    return summary


def _build_error_record(sample: EvalSample, exc: Exception) -> Dict[str, Any]:
    return {
        "question_id": sample.question_id,
        "sample_id": sample.sample_id,
        "task_type": sample.task_type,
        "status": "EVAL_ERROR",
        "error_type": exc.__class__.__name__,
        "error_message": str(exc),
    }


def _conversation_to_turns(conversation: Any) -> List[Dict[str, Any]]:
    """
    Normalize dataset conversation payload into flat turns list.
    将数据集中 conversation 统一展开为 turn 列表，方便适配器 ingest。
    """
    if isinstance(conversation, list):
        out: List[Dict[str, Any]] = []
        for i, turn in enumerate(conversation):
            if not isinstance(turn, dict):
                continue
            out.append(
                {
                    "turn_index": int(turn.get("turn_index", i)),
                    "speaker": str(turn.get("speaker") or turn.get("role") or "").strip(),
                    "text": str(turn.get("text") or turn.get("content") or "").strip(),
                    "timestamp": str(turn.get("timestamp") or turn.get("time") or "").strip(),
                }
            )
        return [x for x in out if x.get("text")]

    if not isinstance(conversation, dict):
        return []

    out: List[Dict[str, Any]] = []
    session_dt: Dict[str, str] = {}
    for key, value in conversation.items():
        k = str(key)
        if k.startswith("session_") and k.endswith("_date_time"):
            session_dt[k.replace("_date_time", "")] = str(value).strip()

    turn_index = 0
    for key in sorted(conversation.keys()):
        k = str(key)
        if not k.startswith("session_") or k.endswith("_date_time"):
            continue
        turns = conversation.get(k)
        if not isinstance(turns, list):
            continue
        ts = session_dt.get(k, "")
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            text = str(turn.get("text", "")).strip()
            if not text:
                continue
            out.append(
                {
                    "turn_index": turn_index,
                    "speaker": str(turn.get("speaker", "")).strip(),
                    "text": text,
                    "timestamp": ts,
                }
            )
            turn_index += 1
    return out


def _inject_memwiki_if_enabled(cfg: PipelineConfig, adapter: Any, run_ctx: Any, sample_id: str) -> Dict[str, Any]:
    if not cfg.use_memwiki:
        return {}
    if isinstance(run_ctx, dict) and run_ctx.get("memwiki_aux_injected"):
        return dict(run_ctx.get("memwiki_last_report", {}))
    output_root = cfg.memwiki_output_root
    if not output_root:
        _, run_dir = _resolve_output_layout(cfg.output_path)
        output_root = str(run_dir / "memwiki")
    injector = MemWikiInjector(
        config=MemWikiConfig(
            output_root=output_root,
            lint_on_build=False,
            llm_wikify_record_limit=cfg.memwiki_llm_record_limit,
            llm_synthesis_page_limit=cfg.memwiki_llm_synthesis_limit,
        ),
        llm_cfg=None
        if cfg.memwiki_disable_llm
        else {
            "api_key": cfg.evaluator_config.llm_api_key,
            "base_url": cfg.evaluator_config.llm_base_url,
            "model": cfg.evaluator_config.llm_model or "gpt-4o-mini",
            "temperature": cfg.evaluator_config.llm_temperature,
        },
        max_aux_records=cfg.memwiki_max_aux_records,
    )
    report = injector.inject(adapter=adapter, run_ctx=run_ctx, sample_id=sample_id).to_dict()
    if isinstance(run_ctx, dict):
        run_ctx["memwiki_aux_injected"] = bool(report.get("injected", False))
        run_ctx["memwiki_last_report"] = report
    print(json.dumps({"event": "memwiki_aux_injection", **report}, ensure_ascii=False), flush=True)
    return report


class ThreeProbeEvaluationPipeline:
    """
    End-to-end pipeline:
    dataset -> adapter runtime ctx -> three-probe evaluator -> output report.
    端到端流水线：数据集 -> 适配器运行态 -> 三探针评估 -> 结果落盘。
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.evaluator = ParallelThreeProbeEvaluator(config=cfg.evaluator_config)

    def run(self, adapter: FullEvalAdapterProtocol) -> Dict[str, Any]:
        samples = build_locomo_eval_samples(
            dataset_path=self.cfg.dataset_path,
            limit=self.cfg.limit,
            f_key_mode=self.cfg.f_key_mode,
        )
        eval_samples: List[EvalSample] = [s.to_eval_sample() for s in samples]
        episode_map = _load_episode_map(self.cfg.dataset_path)
        run_ctx_cache: Dict[str, Any] = {}
        artifact_map = _load_build_artifact_map(self.cfg.build_manifest_path)
        results: List[AttributionResult] = []
        errors: List[Dict[str, Any]] = []
        question_index: List[Dict[str, Any]] = []
        bundle_path, run_dir = _resolve_output_layout(self.cfg.output_path)
        run_dir.mkdir(parents=True, exist_ok=True)

        for sample in eval_samples:
            if sample.sample_id not in run_ctx_cache:
                if sample.sample_id in artifact_map and callable(getattr(adapter, "load_build_artifact", None)):
                    run_ctx_cache[sample.sample_id] = adapter.load_build_artifact(artifact_map[sample.sample_id])
                else:
                    episode = episode_map.get(sample.sample_id, {})
                    conv = episode.get("conversation", {})
                    run_ctx_cache[sample.sample_id] = adapter.ingest_conversation(sample.sample_id, _conversation_to_turns(conv))

            run_ctx = run_ctx_cache[sample.sample_id]
            _inject_memwiki_if_enabled(self.cfg, adapter, run_ctx, sample.sample_id)
            try:
                print(
                    json.dumps(
                        {
                            "event": "eval_question_start",
                            "sample_id": sample.sample_id,
                            "question_id": sample.question_id,
                            "task_type": sample.task_type,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                result = self.evaluator.evaluate_with_adapters(
                    sample=sample,
                    run_ctx=run_ctx,
                    encoding_adapter=adapter,
                    retrieval_adapter=adapter,
                    generation_adapter=adapter,
                    top_k=self.cfg.top_k,
                )
                results.append(result)
                question_record = _build_question_record(sample, result, run_ctx, adapter)
                result_file = _write_question_record(run_dir, question_record)
                final_correct = bool((question_record.get("generation_correctness", {}).get("online", {}) or {}).get("final_correct", False))
                final_attribution = dict(question_record.get("final_attribution", {}))
                question_index.append(
                    {
                        "question_id": sample.question_id,
                        "sample_id": sample.sample_id,
                        "task_type": sample.task_type,
                        "final_correct": final_correct,
                        "primary_cause": str(final_attribution.get("primary_cause", "")),
                        "result_file": result_file,
                    }
                )
                print(
                    json.dumps(
                        {
                            "event": "eval_question_done",
                            "sample_id": sample.sample_id,
                            "question_id": sample.question_id,
                            "answer_online": question_record.get("answer_online", ""),
                            "final_correct": final_correct,
                            "primary_cause": final_attribution.get("primary_cause", ""),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            except Exception as exc:
                errors.append(_build_error_record(sample, exc))
                error_record = dict(errors[-1])
                error_record["question"] = sample.question
                error_record["artifact_refs"] = _extract_artifact_refs(adapter, run_ctx)
                result_file = _write_question_record(
                    run_dir,
                    {
                        "question_id": sample.question_id,
                        "sample_id": sample.sample_id,
                        "task_type": sample.task_type,
                        "question": sample.question,
                        "status": "EVAL_ERROR",
                        "error": error_record,
                    },
                )
                question_index.append(
                    {
                        "question_id": sample.question_id,
                        "sample_id": sample.sample_id,
                        "task_type": sample.task_type,
                        "final_correct": False,
                        "primary_cause": "eval_error",
                        "result_file": result_file,
                    }
                )
                print(json.dumps({"event": "eval_question_error", **errors[-1]}, ensure_ascii=False), flush=True)

        summary = _build_summary(results)
        summary["errors"] = len(errors)
        summary["total"] = len(results) + len(errors)
        summary["ok"] = len(results)
        summary = _augment_summary(summary, question_index)
        payload = {
            "config": {
                "dataset_path": self.cfg.dataset_path,
                "top_k": self.cfg.top_k,
                "limit": self.cfg.limit,
                "f_key_mode": self.cfg.f_key_mode,
                "build_manifest_path": self.cfg.build_manifest_path,
                "evaluator_config": _redact_secrets(asdict(self.cfg.evaluator_config)),
            },
            "adapter_manifest": export_adapter_runtime_manifest(adapter),
            "summary": summary,
            "question_index": question_index,
            "results": [r.to_dict() for r in results],
            "errors": errors,
            "artifacts": {
                "run_summary": "run_summary.json",
                "question_index": "question_index.json",
                "questions_root": ".",
            },
        }

        (run_dir / "run_summary.json").write_text(
            json.dumps(
                {
                    "config": payload["config"],
                    "adapter_manifest": payload["adapter_manifest"],
                    "summary": summary,
                    "errors": errors,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (run_dir / "question_index.json").write_text(json.dumps(question_index, ensure_ascii=False, indent=2), encoding="utf-8")
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        with bundle_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return payload
