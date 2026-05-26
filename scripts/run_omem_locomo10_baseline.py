"""
O-Mem LoCoMo-10 Baseline Reproduction
--------------------------------------
Reproduce O-Mem's online QA accuracy on locomo10.json, computing per-category
F1 and BLEU-1 metrics matching the LoCoMo paper format (Table 2).

Usage:
  # Full 10-sample run
  nohup conda run -n omem-paper100 python scripts/run_omem_locomo10_baseline.py \
      --api-key <KEY> --embedding-model-path <PATH> \
      > outputs/logs/omem_locomo10_baseline.log 2>&1 &

  # Single sample (sample0 = conv-26, all questions)
  conda run -n omem-paper100 python scripts/run_omem_locomo10_baseline.py \
      --sample-id conv-26 --api-key <KEY> --embedding-model-path <PATH>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Bootstrap CUDA before any torch import (through adapters)
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
from memory_eval.dataset.locomo_builder import build_locomo_eval_samples, LocomoEvalSample
from memory_eval.pipeline.runner import _conversation_to_turns

CATEGORY_NAMES = {1: "Multi-hop", 2: "Temporal", 3: "Open", 4: "Single-hop", 5: "Adversarial"}


def _tokens(text: str) -> List[str]:
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", str(text or "").lower())
    return [t for t in cleaned.split() if t]


def compute_f1(pred: str, gold: str) -> float:
    pred_tokens = _tokens(pred)
    gold_tokens = _tokens(gold)
    if not gold_tokens or not pred_tokens:
        return 0.0
    gold_counts: Dict[str, int] = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    overlap = 0
    for t in pred_tokens:
        if t in gold_counts and gold_counts[t] > 0:
            overlap += 1
            gold_counts[t] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_bleu1(pred: str, gold: str) -> float:
    pred_tokens = _tokens(pred)
    gold_tokens = _tokens(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    gold_counts: Dict[str, int] = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    clipped = 0
    for t in pred_tokens:
        if t in gold_counts and gold_counts[t] > 0:
            clipped += 1
            gold_counts[t] -= 1
    bp = min(1.0, len(pred_tokens) / max(len(gold_tokens), 1))
    precision = clipped / len(pred_tokens) if pred_tokens else 0.0
    return bp * precision if precision > 0 else 0.0


def _load_episode_map(dataset_path: Path) -> Dict[str, Dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as f:
        episodes = json.load(f)
    out: Dict[str, Dict[str, Any]] = {}
    for ep in episodes:
        sid = str(ep.get("sample_id", "")).strip()
        if sid:
            out[sid] = ep
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="O-Mem LoCoMo-10 baseline reproduction with per-category F1/BLEU-1.")
    parser.add_argument("--dataset", default="data/locomo10.json")
    parser.add_argument("--output", default="outputs/omem_locomo10_baseline.json")
    parser.add_argument("--sample-id", default="", help='Run only this sample, e.g. "conv-26". Empty = all samples.')
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--omem-llm-model", default="")
    parser.add_argument("--embedding-model-path", default="")
    parser.add_argument("--memory-dir", default="outputs/omem_locomo10_baseline_memory")
    parser.add_argument("--retrieval-pieces", type=int, default=15)
    parser.add_argument("--retrieval-drop-threshold", type=float, default=0.1)
    parser.add_argument("--working-memory-max-size", type=int, default=20)
    parser.add_argument("--episodic-memory-refresh-rate", type=int, default=5)
    parser.add_argument("--device", default="")
    parser.add_argument("--disable-auto-select-cuda", action="store_true")
    parser.add_argument("--omem-root", default="")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    dataset_path = (PROJECT_ROOT / args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    keys_path = str(PROJECT_ROOT / "configs" / "keys.local.json")
    creds = load_runtime_credentials(keys_path, require_complete=False)
    api_key = args.api_key or creds.get("api_key", "")
    base_url = args.base_url or creds.get("base_url", "https://api.openai.com/v1")
    llm_model = args.omem_llm_model or creds.get("model", "gpt-4o-mini")
    embedding_model = args.embedding_model_path or str(PROJECT_ROOT / "Qwen" / "Qwen3-Embedding-0.6B")

    if not api_key:
        raise RuntimeError("api_key is required (--api-key or configs/keys.local.json)")

    omem_root = args.omem_root or str(PROJECT_ROOT / "system" / "O-Mem-StableEval")

    all_samples = build_locomo_eval_samples(str(dataset_path), limit=None)
    episode_map = _load_episode_map(dataset_path)

    if args.sample_id:
        target_sid = args.sample_id.strip()
        all_samples = [s for s in all_samples if s.sample_id == target_sid]
        if not all_samples:
            raise RuntimeError(f"No questions found for sample_id={target_sid}")
        print(f"[INFO] Filtered to sample_id={target_sid}: {len(all_samples)} questions")

    sample_ids_ordered = []
    seen = set()
    for s in all_samples:
        if s.sample_id not in seen:
            sample_ids_ordered.append(s.sample_id)
            seen.add(s.sample_id)

    by_sample: Dict[str, List[LocomoEvalSample]] = defaultdict(list)
    for s in all_samples:
        by_sample[s.sample_id].append(s)

    print(f"[INFO] Samples to process: {sample_ids_ordered}")
    print(f"[INFO] Total questions: {len(all_samples)}")
    print(f"[INFO] Embedding model: {embedding_model}")
    print(f"[INFO] LLM model: {llm_model}")
    print(f"[INFO] O-Mem root: {omem_root}")
    sys.stdout.flush()

    all_rows: List[Dict[str, Any]] = []
    cat_metrics: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: {"f1": [], "bleu1": []})
    sample_summaries: Dict[str, Dict[str, Any]] = {}

    for si, sid in enumerate(sample_ids_ordered):
        questions = by_sample[sid]
        print(f"\n{'='*60}")
        print(f"[SAMPLE {si+1}/{len(sample_ids_ordered)}] {sid} — {len(questions)} questions")
        print(f"{'='*60}")
        sys.stdout.flush()

        memory_dir = str(Path(args.memory_dir) / sid)
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

        episode = episode_map.get(sid, {})
        conv = _conversation_to_turns(episode.get("conversation", {}))
        t0 = time.time()
        print(f"  [INGEST] Starting conversation ingest for {sid} ({len(conv)} turns)...")
        sys.stdout.flush()
        run_ctx = adapter.ingest_conversation(sid, conv)
        ingest_sec = time.time() - t0
        print(f"  [INGEST] Done in {ingest_sec:.1f}s")
        sys.stdout.flush()

        sample_cat_metrics: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: {"f1": [], "bleu1": []})

        for qi, q_sample in enumerate(questions):
            t1 = time.time()
            try:
                answer_online = adapter.generate_online_answer(run_ctx, q_sample.question, args.top_k)
            except Exception as exc:
                answer_online = f"[ERROR] {exc}"
                print(f"    [Q{qi}] ERROR generating answer: {exc}")

            f1 = compute_f1(answer_online, q_sample.answer_gold)
            b1 = compute_bleu1(answer_online, q_sample.answer_gold)
            cat = q_sample.category

            cat_metrics[cat]["f1"].append(f1)
            cat_metrics[cat]["bleu1"].append(b1)
            sample_cat_metrics[cat]["f1"].append(f1)
            sample_cat_metrics[cat]["bleu1"].append(b1)

            row = {
                "question_id": q_sample.question_id,
                "sample_id": sid,
                "category": cat,
                "category_name": CATEGORY_NAMES.get(cat, f"cat{cat}"),
                "task_type": q_sample.task_type,
                "question": q_sample.question,
                "answer_gold": q_sample.answer_gold,
                "answer_online": answer_online,
                "f1": round(f1, 4),
                "bleu1": round(b1, 4),
            }
            all_rows.append(row)

            elapsed = time.time() - t1
            if (qi + 1) % 10 == 0 or qi == 0 or qi == len(questions) - 1:
                print(f"    [Q{qi+1}/{len(questions)}] cat={cat} f1={f1:.3f} b1={b1:.3f} ({elapsed:.1f}s)")
                sys.stdout.flush()

        s_summary = {}
        for c in sorted(sample_cat_metrics.keys()):
            vals = sample_cat_metrics[c]
            n = len(vals["f1"])
            avg_f1 = sum(vals["f1"]) / n if n else 0
            avg_b1 = sum(vals["bleu1"]) / n if n else 0
            s_summary[f"cat{c}"] = {"name": CATEGORY_NAMES.get(c, f"cat{c}"), "count": n, "f1": round(avg_f1 * 100, 2), "bleu1": round(avg_b1 * 100, 2)}
        all_f1 = [r["f1"] for r in all_rows if r["sample_id"] == sid]
        all_b1 = [r["bleu1"] for r in all_rows if r["sample_id"] == sid]
        s_summary["average"] = {"count": len(all_f1), "f1": round(sum(all_f1) / len(all_f1) * 100, 2) if all_f1 else 0, "bleu1": round(sum(all_b1) / len(all_b1) * 100, 2) if all_b1 else 0}
        sample_summaries[sid] = s_summary
        print(f"  [SUMMARY] {sid}: {json.dumps(s_summary, ensure_ascii=False)}")
        sys.stdout.flush()

    global_summary: Dict[str, Any] = {}
    for c in sorted(cat_metrics.keys()):
        vals = cat_metrics[c]
        n = len(vals["f1"])
        avg_f1 = sum(vals["f1"]) / n if n else 0
        avg_b1 = sum(vals["bleu1"]) / n if n else 0
        global_summary[f"cat{c}"] = {"name": CATEGORY_NAMES.get(c, f"cat{c}"), "count": n, "f1": round(avg_f1 * 100, 2), "bleu1": round(avg_b1 * 100, 2)}
    total_f1 = [r["f1"] for r in all_rows]
    total_b1 = [r["bleu1"] for r in all_rows]
    global_summary["average"] = {"count": len(total_f1), "f1": round(sum(total_f1) / len(total_f1) * 100, 2) if total_f1 else 0, "bleu1": round(sum(total_b1) / len(total_b1) * 100, 2) if total_b1 else 0}

    print(f"\n{'='*60}")
    print("GLOBAL RESULTS (Table 2 format)")
    print(f"{'='*60}")
    header = f"{'Category':<15} {'F1':>8} {'B1':>8} {'Count':>8}"
    print(header)
    print("-" * len(header))
    for c in [1, 2, 3, 4, 5]:
        key = f"cat{c}"
        if key in global_summary:
            d = global_summary[key]
            print(f"{d['name']:<15} {d['f1']:>8.2f} {d['bleu1']:>8.2f} {d['count']:>8}")
    d = global_summary["average"]
    print(f"{'Average':<15} {d['f1']:>8.2f} {d['bleu1']:>8.2f} {d['count']:>8}")
    print()

    report = {
        "run_config": {
            "dataset": str(dataset_path),
            "sample_ids": sample_ids_ordered,
            "total_questions": len(all_rows),
            "models": {"omem_llm_model": llm_model, "embedding_model": embedding_model},
            "omem_hparams": {
                "retrieval_pieces": args.retrieval_pieces,
                "retrieval_drop_threshold": args.retrieval_drop_threshold,
                "working_memory_max_size": args.working_memory_max_size,
                "episodic_memory_refresh_rate": args.episodic_memory_refresh_rate,
                "top_k": args.top_k,
                "omem_root": omem_root,
            },
        },
        "global_summary": global_summary,
        "per_sample_summary": sample_summaries,
        "rows": all_rows,
    }

    out = (PROJECT_ROOT / args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Output written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
