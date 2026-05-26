from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from memory_eval.dataset.locomo_builder import build_locomo_eval_samples
from memory_eval.eval_core import EvaluatorConfig, ParallelThreeProbeEvaluator
from memory_eval.pipeline.runner import _conversation_to_turns


def _chat_json(base_url: str, api_key: str, model: str, temperature: float, prompt: str) -> Dict[str, Any] | None:
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
        obj = json.loads(raw)
        content = str(obj["choices"][0]["message"]["content"]).strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:].strip()
        return json.loads(content)
    except Exception:
        return None


def _judge_online_correct(
    *,
    base_url: str,
    api_key: str,
    judge_model: str,
    judge_temperature: float,
    question: str,
    answer_online: str,
    answer_gold: str,
    task_type: str,
) -> Tuple[bool, Dict[str, Any]]:
    prompt = (
        "You are a strict evaluator for memory QA correctness.\n"
        "Return strict JSON:\n"
        "{\"correct\": true|false, \"reason\": \"...\", \"task_type\": \"POS|NEG\"}\n"
        "Rules:\n"
        "- For POS, answer is correct if semantically equivalent to gold answer.\n"
        "- For NEG, answer is correct only when answer abstains/refuses consistently with gold.\n"
        f"TaskType: {task_type}\n"
        f"Question: {question}\n"
        f"OnlineAnswer: {answer_online}\n"
        f"GoldAnswer: {answer_gold}\n"
    )
    j = _chat_json(base_url, api_key, judge_model, judge_temperature, prompt)
    if not isinstance(j, dict) or "correct" not in j:
        raise RuntimeError("baseline correctness judge failed")
    return bool(j.get("correct", False)), j


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
    parser = argparse.ArgumentParser(description="Two-stage O-Mem eval: baseline accuracy -> attribution on wrong + sampled correct.")
    parser.add_argument("--dataset", default="data/locomo10.json")
    parser.add_argument("--output", default="outputs/omem_two_stage_eval_100.json")
    parser.add_argument("--limit-questions", type=int, default=100)
    parser.add_argument("--correct-sample-count", type=int, default=20)
    parser.add_argument("--baseline-only", action="store_true", help="Only reproduce O-Mem online QA accuracy; skip attribution phase.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--fkey-source", choices=["rule", "llm"], default="rule")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--tau-rank", type=int, default=5)
    parser.add_argument("--tau-snr", type=float, default=0.2)
    parser.add_argument("--neg-noise-threshold", type=float, default=0.15)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--omem-llm-model", default="gpt-4o-mini")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--eval-llm-model", default="gpt-4o-mini")
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--embedding-model-path", required=True)
    parser.add_argument("--memory-dir", default="outputs/omem_real_memory_100")
    parser.add_argument("--retrieval-pieces", type=int, default=15)
    parser.add_argument("--retrieval-drop-threshold", type=float, default=0.1)
    parser.add_argument("--working-memory-max-size", type=int, default=20)
    parser.add_argument("--episodic-memory-refresh-rate", type=int, default=5)
    parser.add_argument("--device", default="", help='Runtime device, e.g. "cuda:3" or "cpu". Empty means auto-select.')
    parser.add_argument("--disable-auto-select-cuda", action="store_true", help="Do not auto-select the freest CUDA device.")
    parser.add_argument("--omem-root", default="")
    args = parser.parse_args()

    dataset_path = (PROJECT_ROOT / args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    eval_samples = [s.to_eval_sample() for s in build_locomo_eval_samples(str(dataset_path), limit=args.limit_questions, f_key_mode=args.fkey_source)]
    if not eval_samples:
        raise RuntimeError("no samples built from dataset")

    adapter = OMemAdapter(
        config=OMemAdapterConfig(
            use_real_omem=True,
            allow_fallback_lightweight=False,
            api_key=args.api_key,
            base_url=args.base_url,
            llm_model=args.omem_llm_model,
            embedding_model_name=args.embedding_model_path,
            memory_dir=args.memory_dir,
            retrieval_pieces=args.retrieval_pieces,
            retrieval_drop_threshold=args.retrieval_drop_threshold,
            working_memory_max_size=args.working_memory_max_size,
            episodic_memory_refresh_rate=args.episodic_memory_refresh_rate,
            device=args.device,
            auto_select_cuda=not args.disable_auto_select_cuda,
            omem_root=args.omem_root,
        )
    )

    evaluator_cfg = EvaluatorConfig(
        tau_rank=args.tau_rank,
        tau_snr=args.tau_snr,
        neg_noise_score_threshold=args.neg_noise_threshold,
        max_workers=args.max_workers,
        use_llm_assist=True,
        llm_model=args.eval_llm_model,
        llm_temperature=args.llm_temperature,
        llm_api_key=args.api_key,
        llm_base_url=args.base_url,
        require_llm_judgement=True,
        strict_adapter_call=True,
        disable_rule_fallback=True,
        require_online_answer=True,
    )
    evaluator = ParallelThreeProbeEvaluator(config=evaluator_cfg)

    # Build runtime context cache by sample_id (ingest once per conversation).
    episode_map = _load_episode_map(dataset_path)
    run_ctx_cache: Dict[str, Any] = {}
    baseline_rows: List[Dict[str, Any]] = []
    incorrect_samples: List[Any] = []
    correct_samples: List[Any] = []

    for sample in eval_samples:
        if sample.sample_id not in run_ctx_cache:
            episode = episode_map.get(sample.sample_id, {})
            conv = _conversation_to_turns(episode.get("conversation", {}))
            run_ctx_cache[sample.sample_id] = adapter.ingest_conversation(sample.sample_id, conv)
        run_ctx = run_ctx_cache[sample.sample_id]

        answer_online = adapter.generate_online_answer(run_ctx, sample.question, args.top_k)
        is_correct, judge_payload = _judge_online_correct(
            base_url=args.base_url,
            api_key=args.api_key,
            judge_model=args.judge_model,
            judge_temperature=args.llm_temperature,
            question=sample.question,
            answer_online=answer_online,
            answer_gold=sample.answer_gold,
            task_type=sample.task_type,
        )
        row = {
            "question_id": sample.question_id,
            "sample_id": sample.sample_id,
            "task_type": sample.task_type,
            "question": sample.question,
            "answer_gold": sample.answer_gold,
            "answer_online": answer_online,
            "correct": bool(is_correct),
            "judge_payload": judge_payload,
        }
        baseline_rows.append(row)
        if is_correct:
            correct_samples.append(sample)
        else:
            incorrect_samples.append(sample)

    random.seed(args.random_seed)
    correct_pick_n = min(max(0, args.correct_sample_count), len(correct_samples))
    sampled_correct = random.sample(correct_samples, correct_pick_n) if correct_pick_n > 0 else []

    wrong_attribution_results: List[Dict[str, Any]] = []
    wrong_attribution_errors: List[Dict[str, Any]] = []
    if not args.baseline_only:
        for s in incorrect_samples:
            run_ctx = run_ctx_cache[s.sample_id]
            try:
                r = evaluator.evaluate_with_adapters(
                    sample=s,
                    run_ctx=run_ctx,
                    encoding_adapter=adapter,
                    retrieval_adapter=adapter,
                    generation_adapter=adapter,
                    top_k=args.top_k,
                )
                wrong_attribution_results.append(r.to_dict())
            except Exception as exc:
                wrong_attribution_errors.append(
                    {
                        "question_id": s.question_id,
                        "sample_id": s.sample_id,
                        "task_type": s.task_type,
                        "status": "EVAL_ERROR",
                        "error_type": exc.__class__.__name__,
                        "error_message": str(exc),
                    }
                )

    correct_attribution_results: List[Dict[str, Any]] = []
    correct_attribution_errors: List[Dict[str, Any]] = []
    if not args.baseline_only:
        for s in sampled_correct:
            run_ctx = run_ctx_cache[s.sample_id]
            try:
                r = evaluator.evaluate_with_adapters(
                    sample=s,
                    run_ctx=run_ctx,
                    encoding_adapter=adapter,
                    retrieval_adapter=adapter,
                    generation_adapter=adapter,
                    top_k=args.top_k,
                )
                correct_attribution_results.append(r.to_dict())
            except Exception as exc:
                correct_attribution_errors.append(
                    {
                        "question_id": s.question_id,
                        "sample_id": s.sample_id,
                        "task_type": s.task_type,
                        "status": "EVAL_ERROR",
                        "error_type": exc.__class__.__name__,
                        "error_message": str(exc),
                    }
                )

    total = len(baseline_rows)
    correct_n = sum(1 for x in baseline_rows if x.get("correct", False))
    acc = (correct_n / total) if total > 0 else 0.0

    report = {
        "run_config": {
            "dataset": str(dataset_path),
            "limit_questions": args.limit_questions,
            "correct_sample_count": args.correct_sample_count,
            "baseline_only": bool(args.baseline_only),
            "random_seed": args.random_seed,
            "fkey_source": args.fkey_source,
            "top_k": args.top_k,
            "models": {
                "omem_generation_model": args.omem_llm_model,
                "baseline_judge_model": args.judge_model,
                "eval_llm_model": args.eval_llm_model,
                "embedding_model": args.embedding_model_path,
            },
            "llm_runtime": {
                "base_url": args.base_url,
                "temperature": args.llm_temperature,
            },
            "omem_hparams": {
                "retrieval_pieces": args.retrieval_pieces,
                "retrieval_drop_threshold": args.retrieval_drop_threshold,
                "working_memory_max_size": args.working_memory_max_size,
                "episodic_memory_refresh_rate": args.episodic_memory_refresh_rate,
                "device": args.device,
                "auto_select_cuda": not args.disable_auto_select_cuda,
                "memory_dir": args.memory_dir,
                "omem_root": args.omem_root,
                "use_real_omem": True,
                "allow_fallback_lightweight": False,
            },
            "evaluator_hparams": asdict(evaluator_cfg),
        },
        "baseline_summary": {
            "total_questions": total,
            "correct": correct_n,
            "incorrect": total - correct_n,
            "accuracy": acc,
        },
        "baseline_rows": baseline_rows,
        "phase2_plan": {
            "enabled": not args.baseline_only,
            "incorrect_count": len(incorrect_samples),
            "sampled_correct_count": len(sampled_correct),
            "sampled_correct_question_ids": [s.question_id for s in sampled_correct],
        },
        "attribution_results": {
            "incorrect_subset": {
                "results": wrong_attribution_results,
                "errors": wrong_attribution_errors,
            },
            "sampled_correct_subset": {
                "results": correct_attribution_results,
                "errors": correct_attribution_errors,
            },
        },
    }

    out = (PROJECT_ROOT / args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Two-stage O-Mem evaluation finished.")
    print(f"Output: {out}")
    print(f"Baseline accuracy: {acc:.4f} ({correct_n}/{total})")
    if args.baseline_only:
        print("Attribution phase skipped (--baseline-only).")
    else:
        print(f"Incorrect subset attributed: {len(wrong_attribution_results)} ok, {len(wrong_attribution_errors)} errors")
        print(f"Sampled correct subset attributed: {len(correct_attribution_results)} ok, {len(correct_attribution_errors)} errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
