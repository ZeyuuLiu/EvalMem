"""
Membox: three-probe attribution for ALL questions of one LoCoMo sample.

Uses Membox_stableEval by default (path stability fix). Requires valid API in
configs/keys.local.json or --api-key / env MEMORY_EVAL_API_KEY.

Example:
  nohup conda run --no-capture-output -n omem-paper100 python -u scripts/run_membox_sample_attribution.py \\
      --sample-id conv-26 \\
      > outputs/logs/membox_attr_conv26.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from memory_eval.adapters import MemboxAdapter, MemboxAdapterConfig
from memory_eval.adapters.o_mem_adapter import load_runtime_credentials
from memory_eval.dataset.locomo_builder import build_locomo_eval_samples
from memory_eval.eval_core import EvaluatorConfig, ParallelThreeProbeEvaluator
from memory_eval.pipeline.runner import _conversation_to_turns

CATEGORY_NAMES = {1: "Multi-hop", 2: "Temporal", 3: "Open", 4: "Single-hop", 5: "Adversarial"}


def _load_episode_map(dataset_path: Path) -> Dict[str, Dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as f:
        episodes = json.load(f)
    return {str(ep.get("sample_id", "")).strip(): ep for ep in episodes if ep.get("sample_id")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Membox three-probe attribution on all questions of one sample.")
    parser.add_argument("--dataset", default="data/locomo10.json")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--memory-dir", default="")
    parser.add_argument("--membox-root", default="", help="Default: system/Membox_stableEval")
    parser.add_argument("--keys-path", default="configs/keys.local.json")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--eval-llm-model", default="")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--tau-rank", type=int, default=5)
    parser.add_argument("--tau-snr", type=float, default=0.2)
    args = parser.parse_args()

    dataset_path = (PROJECT_ROOT / args.dataset).resolve()
    sid = args.sample_id.strip()
    out_rel = args.output or f"outputs/membox_attr_{sid}_all.json"
    mem_rel = args.memory_dir or f"outputs/membox_attr_{sid}_all_memory"

    keys_path = Path(args.keys_path)
    if not keys_path.is_absolute():
        keys_path = PROJECT_ROOT / keys_path
    creds = load_runtime_credentials(str(keys_path), require_complete=False)
    api_key = args.api_key or creds.get("api_key", "")
    base_url = args.base_url or creds.get("base_url", "https://api.openai.com/v1")
    llm_model = args.llm_model or creds.get("model", "gpt-4o-mini")
    eval_llm_model = args.eval_llm_model or llm_model

    if not api_key:
        raise RuntimeError("api_key is required (configs/keys.local.json, --api-key, or MEMORY_EVAL_API_KEY)")

    all_rows = build_locomo_eval_samples(str(dataset_path), limit=None)
    questions = [s for s in all_rows if s.sample_id == sid]
    if not questions:
        raise RuntimeError(f"No questions for sample_id={sid}")

    eval_samples = [s.to_eval_sample() for s in questions]
    print(f"[INFO] Membox attribution {sid}: {len(eval_samples)} questions")
    sys.stdout.flush()

    membox_root = args.membox_root
    if not membox_root:
        membox_root = str(PROJECT_ROOT / "system" / "Membox_stableEval")
    else:
        membox_root = str(Path(membox_root).resolve())

    adapter = MemboxAdapter(
        MemboxAdapterConfig(
            api_key=api_key,
            base_url=base_url,
            llm_model=llm_model,
            embedding_model=str(args.embedding_model or "text-embedding-3-small"),
            membox_root=membox_root,
            memory_dir=str((PROJECT_ROOT / mem_rel).resolve()),
            run_id_prefix=f"attr_{sid}",
            answer_top_n=5,
            text_modes=["content_trace_event"],
        )
    )

    evaluator_cfg = EvaluatorConfig(
        tau_rank=args.tau_rank,
        tau_snr=args.tau_snr,
        max_workers=3,
        use_llm_assist=True,
        llm_model=eval_llm_model,
        llm_temperature=0.0,
        llm_api_key=api_key,
        llm_base_url=base_url,
        require_llm_judgement=True,
        strict_adapter_call=True,
        disable_rule_fallback=True,
        require_online_answer=True,
        encoding_merge_native_retrieval=True,
        encoding_native_retrieval_top_k=20,
    )
    evaluator = ParallelThreeProbeEvaluator(config=evaluator_cfg)

    episode_map = _load_episode_map(dataset_path)
    episode = episode_map.get(sid, {})
    conv = _conversation_to_turns(episode.get("conversation", {}))

    print(f"[INGEST] turns={len(conv)}")
    sys.stdout.flush()
    t0 = time.time()
    run_ctx = adapter.ingest_conversation(sid, conv)
    print(f"[INGEST] done in {time.time() - t0:.1f}s")
    sys.stdout.flush()

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    defect_counts: Dict[str, int] = defaultdict(int)
    state_counts: Dict[str, Dict[str, int]] = {"enc": defaultdict(int), "ret": defaultdict(int), "gen": defaultdict(int)}

    for qi, sample in enumerate(eval_samples):
        t1 = time.time()
        try:
            r = evaluator.evaluate_with_adapters(
                sample=sample,
                run_ctx=run_ctx,
                encoding_adapter=adapter,
                retrieval_adapter=adapter,
                generation_adapter=adapter,
                top_k=args.top_k,
            )
            rd = r.to_dict()
            rd["category"] = sample.category
            rd["category_name"] = CATEGORY_NAMES.get(sample.category, f"cat{sample.category}")
            results.append(rd)
            for d in r.defects:
                defect_counts[d] += 1
            for probe in ("enc", "ret", "gen"):
                state_counts[probe][r.states.get(probe, "UNKNOWN")] += 1
            status = f"enc={r.states.get('enc')} ret={r.states.get('ret')} gen={r.states.get('gen')}"
        except Exception as exc:
            errors.append(
                {
                    "question_id": sample.question_id,
                    "sample_id": sample.sample_id,
                    "category": sample.category,
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                }
            )
            status = f"ERROR {exc.__class__.__name__}"
        elapsed = time.time() - t1
        if (qi + 1) % 5 == 0 or qi == 0 or qi == len(eval_samples) - 1:
            print(f"  [Q{qi+1}/{len(eval_samples)}] {sample.question_id} {status} ({elapsed:.1f}s)")
            sys.stdout.flush()

    summary = {
        "total": len(eval_samples),
        "attributed": len(results),
        "errors": len(errors),
        "defect_counts": dict(defect_counts),
        "state_counts": {k: dict(v) for k, v in state_counts.items()},
    }

    report = {
        "run_config": {
            "memory_system": "membox_stable_eval",
            "dataset": str(dataset_path),
            "sample_id": sid,
            "membox_root": membox_root,
            "evaluator_config": asdict(evaluator_cfg),
            "models": {"membox_llm": llm_model, "eval_llm": eval_llm_model},
        },
        "summary": summary,
        "results": results,
        "errors": errors,
    }

    out_path = (PROJECT_ROOT / out_rel).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
