from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_eval.adapters import create_adapter_by_system, export_adapter_runtime_manifest, load_runtime_credentials
from memory_eval.dataset.locomo_builder import build_locomo_eval_samples
from memory_eval.eval_core.agentic_rag.config import AgenticRAGConfig
from memory_eval.eval_core.agentic_rag.retriever import AgenticRAGRetriever
from memory_eval.eval_core.correctness_judge import judge_answer_correctness
from memory_eval.eval_core.models import EvaluatorConfig
from memory_eval.memwiki.config import MemWikiConfig
from memory_eval.memwiki.injector import MemWikiInjector
from memory_eval.pipeline.runner import PipelineConfig, ThreeProbeEvaluationPipeline, _conversation_to_turns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real memory-system reproduction or evaluation on LOCOMO.")
    parser.add_argument("--memory-system", required=True, help="Registered adapter key, e.g. o_mem_stable_eval or membox_stable_eval")
    parser.add_argument("--dataset", default="data/locomo10.json")
    parser.add_argument("--sample-id", default="", help="Optional LOCOMO sample_id filter")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mode", choices=["build", "baseline", "eval"], default="eval")
    parser.add_argument("--output", default="")
    parser.add_argument("--build-manifest", default="")
    parser.add_argument("--keys-path", default=str(PROJECT_ROOT / "configs" / "keys.local.json"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--embedding-model-path", default="")
    parser.add_argument("--omem-root", default="")
    parser.add_argument("--membox-root", default="")
    parser.add_argument("--use-real-omem", action="store_true")
    parser.add_argument("--allow-fallback-lightweight", action="store_true")
    parser.add_argument("--llm-assist", dest="llm_assist", action="store_true")
    parser.add_argument("--no-llm-assist", dest="llm_assist", action="store_false")
    parser.add_argument("--strict-judge", dest="strict_judge", action="store_true")
    parser.add_argument("--no-strict-judge", dest="strict_judge", action="store_false")
    parser.add_argument("--allow-correctness-rule-fallback", action="store_true")
    parser.add_argument("--eval-model", default="", help="Override evaluator/judge model")
    parser.add_argument("--system-model", default="", help="Override memory-system model")
    parser.add_argument("--request-timeout-sec", type=float, default=120.0)
    parser.add_argument("--use-agentic-rag", action="store_true", help="Use bounded encoding-layer memory RAG for encoding probe candidates")
    parser.add_argument("--agentic-max-iter", type=int, default=4)
    parser.add_argument("--agentic-max-pool-size", type=int, default=50)
    parser.add_argument("--agentic-round-jaccard-threshold", type=float, default=0.85)
    parser.add_argument("--agentic-epsilon-new-ratio", type=float, default=0.10)
    parser.add_argument("--agentic-stable-rounds", type=int, default=2)
    parser.add_argument("--agentic-disable-coverage-diagnostics", action="store_true")
    parser.add_argument("--use-memwiki", action="store_true", help="Build and inject MemWiki auxiliary index records before evaluation")
    parser.add_argument("--memwiki-output-root", default="")
    parser.add_argument("--memwiki-max-aux-records", type=int, default=None)
    parser.add_argument("--memwiki-disable-llm", action="store_true", help="Force deterministic MemWiki construction even when API credentials exist")
    parser.add_argument("--memwiki-llm-record-limit", type=int, default=None, help="Optional maximum source records wikified through LLM")
    parser.add_argument("--memwiki-llm-synthesis-limit", type=int, default=None, help="Optional maximum entity/topic/event pages synthesized through LLM")
    parser.set_defaults(llm_assist=True, strict_judge=True)
    return parser.parse_args()


def load_dataset(path: Path, sample_id: str) -> Path:
    if not sample_id:
        return path
    data = json.loads(path.read_text(encoding="utf-8"))
    filtered = [episode for episode in data if str(episode.get("sample_id", "")).strip() == sample_id]
    if not filtered:
        raise ValueError(f"dataset 中未找到 sample_id={sample_id}")
    temp_dir = tempfile.mkdtemp(prefix="memory_eval_locomo_")
    out = Path(temp_dir) / path.name
    out.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def build_adapter_config(args: argparse.Namespace) -> Dict[str, Any]:
    creds = load_runtime_credentials(args.keys_path, require_complete=False)
    cfg: Dict[str, Any] = {
        "api_key": creds.get("api_key", ""),
        "base_url": creds.get("base_url", ""),
        "llm_model": args.system_model or creds.get("system_model", "") or creds.get("model", "") or "gpt-4o-mini",
        "keys_path": args.keys_path,
    }
    key = str(args.memory_system).lower()
    if "o_mem" in key or "omem" in key:
        default_embedding = PROJECT_ROOT / "Qwen" / "Qwen3-Embedding-0.6B"
        cfg.update(
            {
                "use_real_omem": bool(args.use_real_omem or "stable_eval" in key),
                "allow_fallback_lightweight": bool(args.allow_fallback_lightweight),
            }
        )
        cfg["embedding_model_name"] = args.embedding_model_path or (str(default_embedding) if default_embedding.exists() else "all-MiniLM-L6-v2")
        if args.omem_root:
            cfg["omem_root"] = args.omem_root
    if "membox" in key and args.membox_root:
        cfg["membox_root"] = args.membox_root
    if "membox" in key:
        cfg["request_timeout_sec"] = float(args.request_timeout_sec)
    return cfg


def _load_build_manifest_by_sample(path: str) -> Dict[str, Dict[str, Any]]:
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


def _build_eval_cfg(args: argparse.Namespace, adapter: Any) -> EvaluatorConfig:
    keys_path = str(getattr(args, "keys_path", PROJECT_ROOT / "configs" / "keys.local.json"))
    creds = load_runtime_credentials(keys_path, require_complete=False)
    return EvaluatorConfig(
        use_llm_assist=bool(args.llm_assist),
        require_llm_judgement=bool(args.strict_judge),
        disable_rule_fallback=bool(args.strict_judge),
        strict_adapter_call=True,
        require_online_answer=False,
        llm_api_key=getattr(adapter.config, "api_key", ""),
        llm_base_url=getattr(adapter.config, "base_url", ""),
        llm_model=getattr(args, "eval_model", "") or creds.get("eval_model", "") or "gpt-4o-mini",
        correctness_use_llm_judge=True,
        correctness_require_llm_judge=not bool(args.allow_correctness_rule_fallback),
    )


def _attach_agentic_rag_if_enabled(args: argparse.Namespace, adapter: Any) -> None:
    if not bool(getattr(args, "use_agentic_rag", False)):
        return
    creds = load_runtime_credentials(args.keys_path, require_complete=False)
    cfg = AgenticRAGConfig(
        max_iter=int(args.agentic_max_iter),
        max_pool_size=int(args.agentic_max_pool_size),
        round_jaccard_threshold=float(args.agentic_round_jaccard_threshold),
        epsilon_new_ratio=float(args.agentic_epsilon_new_ratio),
        consecutive_stable_rounds=int(args.agentic_stable_rounds),
        enable_coverage_diagnostics=not bool(args.agentic_disable_coverage_diagnostics),
        llm_api_key=getattr(adapter.config, "api_key", "") or creds.get("api_key", ""),
        llm_base_url=getattr(adapter.config, "base_url", "") or creds.get("base_url", ""),
        llm_model=args.eval_model or creds.get("eval_model", "") or "gpt-4o-mini",
    )
    adapter.set_external_high_recall_retriever(AgenticRAGRetriever(config=cfg))


def _inject_memwiki_if_enabled(args: argparse.Namespace, adapter: Any, run_ctx: Any, sample_id: str, output_path: Path) -> Dict[str, Any]:
    if not bool(getattr(args, "use_memwiki", False)):
        return {}
    if isinstance(run_ctx, dict) and run_ctx.get("memwiki_aux_injected"):
        return dict(run_ctx.get("memwiki_last_report", {}))
    if args.memwiki_output_root:
        output_root = args.memwiki_output_root
    else:
        _, run_dir = _resolve_output_layout(str(output_path))
        output_root = str(run_dir / "memwiki")
    injector = MemWikiInjector(
        config=MemWikiConfig(
            output_root=output_root,
            lint_on_build=False,
            llm_wikify_record_limit=args.memwiki_llm_record_limit,
            llm_synthesis_page_limit=args.memwiki_llm_synthesis_limit,
        ),
        llm_cfg=None if bool(getattr(args, "memwiki_disable_llm", False)) else _build_memwiki_llm_cfg(args, adapter),
        max_aux_records=args.memwiki_max_aux_records,
    )
    report = injector.inject(adapter=adapter, run_ctx=run_ctx, sample_id=sample_id).to_dict()
    if isinstance(run_ctx, dict):
        run_ctx["memwiki_aux_injected"] = bool(report.get("injected", False))
        run_ctx["memwiki_last_report"] = report
    print(json.dumps({"event": "memwiki_aux_injection", **report}, ensure_ascii=False), flush=True)
    return report


def _build_memwiki_llm_cfg(args: argparse.Namespace, adapter: Any) -> Dict[str, Any]:
    creds = load_runtime_credentials(args.keys_path, require_complete=False)
    return {
        "api_key": getattr(adapter.config, "api_key", "") or creds.get("api_key", ""),
        "base_url": getattr(adapter.config, "base_url", "") or creds.get("base_url", ""),
        "model": (
            getattr(args, "system_model", "")
            or creds.get("system_model", "")
            or creds.get("model", "")
            or creds.get("eval_model", "")
            or "gpt-4o-mini"
        ),
        "temperature": 0.0,
        "timeout_sec": int(float(getattr(args, "request_timeout_sec", 120.0) or 120.0)),
    }


def _safe_segment(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text or "").strip())
    return cleaned.strip("_") or "unknown"


def _resolve_output_layout(output_path: str) -> tuple[Path, Path]:
    out_path = Path(output_path)
    if out_path.suffix.lower() == ".json":
        return out_path, out_path.with_suffix("")
    return out_path / "result_bundle.json", out_path


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


def _write_question_record(run_dir: Path, sample_id: str, question_id: str, record: Dict[str, Any]) -> str:
    rel_path = Path(_safe_segment(sample_id)) / f"{_safe_segment(question_id)}.json"
    target = run_dir / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(rel_path)


def run_build(args: argparse.Namespace, dataset_path: Path, adapter: Any) -> Path:
    samples = build_locomo_eval_samples(str(dataset_path), limit=args.limit)
    episodes = json.loads(dataset_path.read_text(encoding="utf-8"))
    by_sample = {str(ep.get("sample_id", "")).strip(): ep for ep in episodes}
    artifacts: List[Dict[str, Any]] = []
    run_ctx_cache: Dict[str, Any] = {}
    export_fn = getattr(adapter, "export_build_artifact", None)
    if not callable(export_fn):
        raise RuntimeError(f"{adapter.__class__.__name__} does not support build artifact export")
    for sample in samples:
        sample_id = sample.sample_id
        if sample_id in run_ctx_cache:
            continue
        episode = by_sample.get(sample_id)
        if not episode:
            continue
        run_ctx = adapter.ingest_conversation(sample_id, _conversation_to_turns(episode.get("conversation", {})))
        run_ctx_cache[sample_id] = run_ctx
        artifact = dict(export_fn(run_ctx))
        artifact["sample_id"] = sample_id
        artifacts.append(artifact)
        print(json.dumps({"event": "build_done", "sample_id": sample_id, "run_id": artifact.get("run_id", "")}, ensure_ascii=False), flush=True)
    out_path = Path(args.output) if args.output else PROJECT_ROOT / "outputs" / f"{args.memory_system}_build_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "memory_system": args.memory_system,
                "mode": "build",
                "sample_filter": args.sample_id,
                "count": len(artifacts),
                "artifacts": artifacts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_path


def run_baseline(args: argparse.Namespace, dataset_path: Path, adapter: Any) -> Path:
    samples = build_locomo_eval_samples(str(dataset_path), limit=args.limit)
    episodes = json.loads(dataset_path.read_text(encoding="utf-8"))
    by_sample = {str(ep.get("sample_id", "")).strip(): ep for ep in episodes}
    build_manifest = _load_build_manifest_by_sample(args.build_manifest)
    cfg = _build_eval_cfg(args, adapter)
    bundle_path, run_dir = _resolve_output_layout(str(Path(args.output) if args.output else PROJECT_ROOT / "outputs" / f"{args.memory_system}_baseline.json"))
    run_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    question_index: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    correct = 0
    total = 0
    run_ctx_cache: Dict[str, Any] = {}
    for sample in samples:
        sample_id = sample.sample_id
        if sample_id not in run_ctx_cache:
            if sample_id in build_manifest and callable(getattr(adapter, "load_build_artifact", None)):
                run_ctx_cache[sample_id] = adapter.load_build_artifact(build_manifest[sample_id])
            else:
                episode = by_sample.get(sample_id)
                if not episode:
                    continue
                run_ctx_cache[sample_id] = adapter.ingest_conversation(sample_id, _conversation_to_turns(episode.get("conversation", {})))
        run_ctx = run_ctx_cache[sample_id]
        _inject_memwiki_if_enabled(args, adapter, run_ctx, sample_id, bundle_path)
        print(
            json.dumps(
                {"event": "baseline_question_start", "sample_id": sample.sample_id, "question_id": sample.question_id, "task_type": sample.task_type},
                ensure_ascii=False,
            ),
            flush=True,
        )
        try:
            answer_online = adapter.generate_online_answer(run_ctx, sample.question, args.top_k)
            retrieved_context = ""
            retrieve_fn = getattr(adapter, "retrieve_original", None)
            if callable(retrieve_fn):
                try:
                    retrieved_context = _render_retrieved_context(retrieve_fn(run_ctx, sample.question, args.top_k))
                except Exception as exc:
                    retrieved_context = ""
                    errors.append(
                        {
                            "sample_id": sample.sample_id,
                            "question_id": sample.question_id,
                            "stage": "baseline_retrieve_context",
                            "error": str(exc),
                        }
                    )
            judgement = judge_answer_correctness(
                task_type=sample.task_type,
                question=sample.question,
                answer_gold=sample.answer_gold,
                answer_pred=answer_online,
                cfg=cfg,
                judge_mode="online",
                retrieved_context=retrieved_context,
            )
            is_correct = bool(judgement.final_correct)
            total += 1
            correct += int(is_correct)
            row = {
                "sample_id": sample.sample_id,
                "question_id": sample.question_id,
                "task_type": sample.task_type,
                "question": sample.question,
                "answer_gold": sample.answer_gold,
                "answer_online": answer_online,
                "retrieved_context": retrieved_context,
                "rule_correct": judgement.rule_correct,
                "llm_correct": judgement.llm_correct,
                "final_correct": judgement.final_correct,
                "judge_label": judgement.judge_label,
                "judge_reason": judgement.judge_reason,
                "judge_payload": judgement.judge_payload,
                "artifact_refs": _extract_artifact_refs(adapter, run_ctx),
            }
            results.append(row)
            result_file = _write_question_record(run_dir, sample.sample_id, sample.question_id, row)
            question_index.append(
                {
                    "sample_id": sample.sample_id,
                    "question_id": sample.question_id,
                    "task_type": sample.task_type,
                    "final_correct": row["final_correct"],
                    "result_file": result_file,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "baseline_question_done",
                        "sample_id": sample.sample_id,
                        "question_id": sample.question_id,
                        "answer_online": answer_online,
                        "final_correct": row["final_correct"],
                        "judge_label": row["judge_label"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            error_record = {
                "sample_id": sample.sample_id,
                "question_id": sample.question_id,
                "task_type": sample.task_type,
                "question": sample.question,
                "stage": "baseline",
                "error": str(exc),
                "artifact_refs": _extract_artifact_refs(adapter, run_ctx),
            }
            errors.append(error_record)
            result_file = _write_question_record(run_dir, sample.sample_id, sample.question_id, {"status": "BASELINE_ERROR", **error_record})
            question_index.append(
                {
                    "sample_id": sample.sample_id,
                    "question_id": sample.question_id,
                    "task_type": sample.task_type,
                    "final_correct": False,
                    "result_file": result_file,
                }
            )
            print(json.dumps({"event": "baseline_question_error", **error_record}, ensure_ascii=False), flush=True)
    summary = {
        "memory_system": args.memory_system,
        "mode": "baseline",
        "sample_filter": args.sample_id,
        "count": total + len([e for e in errors if e.get("stage") == "baseline"]),
        "final_correct": correct,
        "final_accuracy": (correct / total) if total else 0.0,
        "errors": len([e for e in errors if e.get("stage") == "baseline"]),
        "adapter_manifest": export_adapter_runtime_manifest(adapter),
    }
    payload = {
        "summary": summary,
        "question_index": question_index,
        "results": results,
        "errors": errors,
        "artifacts": {
            "run_summary": "run_summary.json",
            "question_index": "question_index.json",
            "questions_root": ".",
        },
    }
    (run_dir / "run_summary.json").write_text(json.dumps({"summary": summary, "errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "question_index.json").write_text(json.dumps(question_index, ensure_ascii=False, indent=2), encoding="utf-8")
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return bundle_path


def run_eval(args: argparse.Namespace, dataset_path: Path, adapter: Any) -> Path:
    cfg = _build_eval_cfg(args, adapter)
    out_path = Path(args.output) if args.output else PROJECT_ROOT / "outputs" / f"{args.memory_system}_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline = ThreeProbeEvaluationPipeline(
        PipelineConfig(
            dataset_path=str(dataset_path),
            output_path=str(out_path),
            top_k=int(args.top_k),
            evaluator_config=cfg,
            limit=args.limit,
            build_manifest_path=args.build_manifest,
            use_memwiki=bool(getattr(args, "use_memwiki", False)),
            memwiki_output_root=str(getattr(args, "memwiki_output_root", "")),
            memwiki_max_aux_records=getattr(args, "memwiki_max_aux_records", None),
            memwiki_disable_llm=bool(getattr(args, "memwiki_disable_llm", False)),
            memwiki_llm_record_limit=getattr(args, "memwiki_llm_record_limit", None),
            memwiki_llm_synthesis_limit=getattr(args, "memwiki_llm_synthesis_limit", None),
        )
    )
    pipeline.run(adapter)
    return out_path


def main() -> None:
    args = parse_args()
    try:
        dataset_path = load_dataset((PROJECT_ROOT / args.dataset).resolve(), args.sample_id)
        adapter = create_adapter_by_system(args.memory_system, build_adapter_config(args))
        _attach_agentic_rag_if_enabled(args, adapter)
        if args.mode == "build":
            out_path = run_build(args, dataset_path, adapter)
        elif args.mode == "baseline":
            out_path = run_baseline(args, dataset_path, adapter)
        else:
            out_path = run_eval(args, dataset_path, adapter)
        print(json.dumps({"ok": True, "memory_system": args.memory_system, "mode": args.mode, "output": str(out_path)}, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"ok": False, "memory_system": args.memory_system, "mode": args.mode, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
