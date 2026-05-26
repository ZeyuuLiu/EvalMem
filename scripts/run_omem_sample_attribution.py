"""
O-Mem Sample-level Three-Probe Attribution
-------------------------------------------
Run the full three-probe attribution framework on ALL questions for a given
sample in locomo10.json.

Usage:
  nohup conda run -n omem-paper100 python scripts/run_omem_sample_attribution.py \
      --sample-id conv-26 \
      --api-key <KEY> --embedding-model-path <PATH> \
      > outputs/logs/omem_attr_conv26.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
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

_omem_root = str(PROJECT_ROOT / "system" / "O-Mem-StableEval")
if _omem_root not in sys.path:
    sys.path.insert(0, _omem_root)
try:
    from memory_chain._gpu_runtime import bootstrap_cuda_wheel_runtime
    bootstrap_cuda_wheel_runtime()
except ImportError:
    pass

from memory_eval.adapters import OMemAdapter, OMemAdapterConfig
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
    parser = argparse.ArgumentParser(description="Three-probe attribution on all questions of one LoCoMo sample.")
    parser.add_argument("--dataset", default="data/locomo10.json")
    parser.add_argument("--sample-id", required=True, help='Target sample, e.g. "conv-26".')
    parser.add_argument("--output", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--omem-llm-model", default="")
    parser.add_argument("--eval-llm-model", default="")
    parser.add_argument("--embedding-model-path", default="")
    parser.add_argument("--memory-dir", default="")
    parser.add_argument("--retrieval-pieces", type=int, default=15)
    parser.add_argument("--retrieval-drop-threshold", type=float, default=0.1)
    parser.add_argument("--working-memory-max-size", type=int, default=20)
    parser.add_argument("--episodic-memory-refresh-rate", type=int, default=5)
    parser.add_argument("--device", default="")
    parser.add_argument("--disable-auto-select-cuda", action="store_true")
    parser.add_argument("--omem-root", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--tau-rank", type=int, default=5)
    parser.add_argument("--tau-snr", type=float, default=0.2)
    args = parser.parse_args()

    dataset_path = (PROJECT_ROOT / args.dataset).resolve()
    sid = args.sample_id.strip()
    output_path = args.output or f"outputs/omem_attr_{sid}.json"
    memory_dir = args.memory_dir or f"outputs/omem_attr_{sid}_memory"

    keys_path = str(PROJECT_ROOT / "configs" / "keys.local.json")
    creds = load_runtime_credentials(keys_path, require_complete=False)
    api_key = args.api_key or creds.get("api_key", "")
    base_url = args.base_url or creds.get("base_url", "https://api.openai.com/v1")
    llm_model = args.omem_llm_model or creds.get("model", "gpt-4o-mini")
    eval_llm_model = args.eval_llm_model or llm_model
    embedding_model = args.embedding_model_path or str(PROJECT_ROOT / "Qwen" / "Qwen3-Embedding-0.6B")
    omem_root = args.omem_root or str(PROJECT_ROOT / "system" / "O-Mem-StableEval")

    if not api_key:
        raise RuntimeError("api_key is required")

    all_samples = build_locomo_eval_samples(str(dataset_path), limit=None)
    questions = [s for s in all_samples if s.sample_id == sid]
    if not questions:
        raise RuntimeError(f"No questions found for sample_id={sid}")

    eval_samples = [s.to_eval_sample() for s in questions]
    print(f"[INFO] Sample {sid}: {len(eval_samples)} questions")
    print(f"[INFO] Category distribution: { {CATEGORY_NAMES.get(c, f'cat{c}'): sum(1 for s in eval_samples if s.category == c) for c in sorted(set(s.category for s in eval_samples))} }")
    sys.stdout.flush()

    adapter = OMemAdapter(
        config=OMemAdapterConfig(
            use_real_omem=True,
            allow_fallback_lightweight=False,
            api_key=api_key,
            base_url=base_url,
            llm_model=llm_model,
            embedding_model_name=embedding_model,
            memory_dir=memory_dir,
            retrieval_pieces=args.retrieval_pieces,
            retrieval_drop_threshold=args.retrieval_drop_threshold,
            working_memory_max_size=args.working_memory_max_size,
            episodic_memory_refresh_rate=args.episodic_memory_refresh_rate,
            device=args.device,
            auto_select_cuda=not args.disable_auto_select_cuda,
            omem_root=omem_root,
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

    print(f"[INGEST] Starting conversation ingest ({len(conv)} turns)...")
    sys.stdout.flush()
    t0 = time.time()
    run_ctx = adapter.ingest_conversation(sid, conv)
    print(f"[INGEST] Done in {time.time() - t0:.1f}s")
    sys.stdout.flush()

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    defect_counts: Dict[str, int] = defaultdict(int)
    state_counts: Dict[str, Dict[str, int]] = {"enc": defaultdict(int), "ret": defaultdict(int), "gen": defaultdict(int)}
    cat_defect_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

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
                cat_defect_counts[sample.category][d] += 1
            for probe in ("enc", "ret", "gen"):
                s = r.states.get(probe, "UNKNOWN")
                state_counts[probe][s] += 1

            elapsed = time.time() - t1
            defect_str = ",".join(r.defects) if r.defects else "none"
            status = f"enc={r.states.get('enc','?')} ret={r.states.get('ret','?')} gen={r.states.get('gen','?')} defects=[{defect_str}]"
        except Exception as exc:
            errors.append({
                "question_id": sample.question_id,
                "sample_id": sample.sample_id,
                "category": sample.category,
                "task_type": sample.task_type,
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
            })
            elapsed = time.time() - t1
            status = f"ERROR: {exc.__class__.__name__}: {exc}"

        if (qi + 1) % 5 == 0 or qi == 0 or qi == len(eval_samples) - 1:
            print(f"  [Q{qi+1}/{len(eval_samples)}] {sample.question_id} cat={sample.category} {status} ({elapsed:.1f}s)")
            sys.stdout.flush()

    summary = {
        "total": len(eval_samples),
        "attributed": len(results),
        "errors": len(errors),
        "defect_counts": dict(defect_counts),
        "state_counts": {k: dict(v) for k, v in state_counts.items()},
        "per_category_defects": {CATEGORY_NAMES.get(c, f"cat{c}"): dict(d) for c, d in sorted(cat_defect_counts.items())},
    }

    print(f"\n{'='*60}")
    print(f"ATTRIBUTION SUMMARY for {sid}")
    print(f"{'='*60}")
    print(f"  Total: {summary['total']}, Attributed: {summary['attributed']}, Errors: {summary['errors']}")
    print(f"  Defects: {dict(defect_counts)}")
    print(f"  Encoding states: {dict(state_counts['enc'])}")
    print(f"  Retrieval states: {dict(state_counts['ret'])}")
    print(f"  Generation states: {dict(state_counts['gen'])}")
    for c in sorted(cat_defect_counts.keys()):
        print(f"  Cat{c} ({CATEGORY_NAMES.get(c,'')}): {dict(cat_defect_counts[c])}")
    sys.stdout.flush()

    report = {
        "run_config": {
            "dataset": str(dataset_path),
            "sample_id": sid,
            "total_questions": len(eval_samples),
            "models": {"omem_llm_model": llm_model, "eval_llm_model": eval_llm_model, "embedding_model": embedding_model},
            "evaluator_config": asdict(evaluator_cfg),
            "omem_hparams": {
                "retrieval_pieces": args.retrieval_pieces,
                "retrieval_drop_threshold": args.retrieval_drop_threshold,
                "working_memory_max_size": args.working_memory_max_size,
                "episodic_memory_refresh_rate": args.episodic_memory_refresh_rate,
                "top_k": args.top_k,
                "omem_root": omem_root,
            },
        },
        "summary": summary,
        "results": results,
        "errors": errors,
    }

    out = (PROJECT_ROOT / output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] Output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
