"""
Membox: baseline for one LoCoMo sample (all questions in that conversation).

Ingest the conversation once, then answer every QA row for the given sample_id.
Uses Membox_stableEval by default (path fix for trace stats).

Example:
  nohup conda run --no-capture-output -n omem-paper100 python -u scripts/run_membox_locomo_sample_baseline.py \\
      --sample-id conv-26 \\
      --membox-root system/Membox_stableEval \\
      > outputs/logs/membox_baseline_conv26.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from memory_eval.adapters import MemboxAdapter, MemboxAdapterConfig
from memory_eval.adapters.o_mem_adapter import load_runtime_credentials
from memory_eval.dataset.locomo_builder import build_locomo_eval_samples
from memory_eval.pipeline.runner import _conversation_to_turns

CATEGORY_NAMES = {1: "Multi-hop", 2: "Temporal", 3: "Open", 4: "Single-hop", 5: "Adversarial"}


def _ensure_nltk_punkt() -> None:
    try:
        import nltk

        nltk.data.find("tokenizers/punkt")
    except LookupError:
        import nltk

        nltk.download("punkt_tab", quiet=True)
        nltk.download("punkt", quiet=True)
    except Exception:
        pass


def _load_episode_map(dataset_path: Path) -> Dict[str, Dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as f:
        episodes = json.load(f)
    return {str(ep.get("sample_id", "")).strip(): ep for ep in episodes if ep.get("sample_id")}


def _chat_json(base_url: str, api_key: str, model: str, temperature: float, prompt: str) -> Dict[str, Any] | None:
    import urllib.request

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
        with urllib.request.urlopen(req, timeout=120) as resp:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Membox baseline: one LoCoMo sample, all questions.")
    parser.add_argument("--dataset", default="data/locomo10.json")
    parser.add_argument("--sample-id", required=True, help='e.g. conv-26')
    parser.add_argument("--output", default="")
    parser.add_argument("--memory-dir", default="")
    parser.add_argument(
        "--membox-root",
        default="system/Membox_stableEval",
        help="Default: stable-eval fork with TRACE_STATS_FILE fix.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--keys-path", default="configs/keys.local.json")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    args = parser.parse_args()

    _ensure_nltk_punkt()

    sid = args.sample_id.strip()
    out_rel = args.output or f"outputs/membox_baseline_{sid}_all.json"
    mem_rel = args.memory_dir or f"outputs/membox_baseline_{sid}_all_memory"

    keys_path = PROJECT_ROOT / args.keys_path if not Path(args.keys_path).is_absolute() else Path(args.keys_path)
    creds = load_runtime_credentials(str(keys_path), require_complete=False)
    api_key = args.api_key or creds.get("api_key", "")
    base_url = args.base_url or creds.get("base_url", "https://api.openai.com/v1")
    llm_model = args.llm_model or creds.get("model", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("api_key missing: configs/keys.local.json, --api-key, or MEMORY_EVAL_API_KEY")
    dataset_path = (PROJECT_ROOT / args.dataset).resolve()

    all_rows = build_locomo_eval_samples(str(dataset_path), limit=None, f_key_mode="rule")
    questions = [q for q in all_rows if q.sample_id == sid]
    if not questions:
        raise RuntimeError(f"No questions for sample_id={sid}")

    membox_root = (PROJECT_ROOT / args.membox_root).resolve() if not Path(args.membox_root).is_absolute() else Path(args.membox_root)
    memory_dir = str((PROJECT_ROOT / mem_rel).resolve())

    adapter = MemboxAdapter(
        MemboxAdapterConfig(
            api_key=api_key,
            base_url=base_url,
            llm_model=llm_model,
            embedding_model=str(args.embedding_model or "text-embedding-3-small"),
            membox_root=str(membox_root),
            memory_dir=memory_dir,
            run_id_prefix=f"baseline_{sid}",
            answer_top_n=5,
            text_modes=["content_trace_event"],
        )
    )

    episode_map = _load_episode_map(dataset_path)
    episode = episode_map.get(sid, {})
    conv = _conversation_to_turns(episode.get("conversation", {}))

    judge_model = str(args.judge_model or llm_model)

    print(f"[INGEST] {sid} turns={len(conv)} questions={len(questions)}")
    sys.stdout.flush()
    t0 = time.time()
    run_ctx = adapter.ingest_conversation(sid, conv)
    print(f"[INGEST] done in {time.time() - t0:.1f}s")
    sys.stdout.flush()

    rows_out: List[Dict[str, Any]] = []
    cat_metrics: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: {"f1": [], "bleu": []})

    ag = adapter._module.AnswerGenerator
    for qi, q_sample in enumerate(questions):
        ev = q_sample.to_eval_sample()
        t1 = time.time()
        answer_online = adapter.generate_online_answer(run_ctx, ev.question, args.top_k)
        f1 = ag._f1(answer_online, ev.answer_gold)
        bleu = ag._bleu(answer_online, ev.answer_gold)
        is_correct, judge_payload = _judge_online_correct(
            base_url=base_url,
            api_key=api_key,
            judge_model=judge_model,
            judge_temperature=0.0,
            question=ev.question,
            answer_online=answer_online,
            answer_gold=ev.answer_gold,
            task_type=ev.task_type,
        )
        cat = q_sample.category
        cat_metrics[cat]["f1"].append(f1)
        cat_metrics[cat]["bleu1"].append(bleu)

        rows_out.append(
            {
                "question_id": ev.question_id,
                "sample_id": sid,
                "category": cat,
                "category_name": CATEGORY_NAMES.get(cat, f"cat{cat}"),
                "task_type": ev.task_type,
                "question": ev.question,
                "answer_gold": ev.answer_gold,
                "answer_online": answer_online,
                "f1": round(f1, 4),
                "bleu1": round(bleu, 4),
                "llm_correct": is_correct,
                "judge_payload": judge_payload,
            }
        )
        elapsed = time.time() - t1
        if (qi + 1) % 10 == 0 or qi == 0 or qi == len(questions) - 1:
            print(f"  [Q{qi+1}/{len(questions)}] cat={cat} f1={f1:.3f} bleu={bleu:.3f} ok={is_correct} ({elapsed:.1f}s)")
            sys.stdout.flush()

    global_summary: Dict[str, Any] = {}
    for c in sorted(cat_metrics.keys()):
        vals = cat_metrics[c]
        n = len(vals["f1"])
        global_summary[f"cat{c}"] = {
            "name": CATEGORY_NAMES.get(c, f"cat{c}"),
            "count": n,
            "f1": round(sum(vals["f1"]) / n * 100, 2) if n else 0,
            "bleu1": round(sum(vals["bleu1"]) / n * 100, 2) if n else 0,
        }
    tf = [r["f1"] for r in rows_out]
    tb = [r["bleu1"] for r in rows_out]
    global_summary["average"] = {
        "count": len(tf),
        "f1": round(sum(tf) / len(tf) * 100, 2) if tf else 0,
        "bleu1": round(sum(tb) / len(tb) * 100, 2) if tb else 0,
    }

    report = {
        "run_config": {
            "sample_id": sid,
            "membox_root": str(membox_root),
            "memory_dir": memory_dir,
            "top_k": args.top_k,
            "total_questions": len(rows_out),
        },
        "global_summary": global_summary,
        "rows": rows_out,
    }

    out_path = (PROJECT_ROOT / out_rel).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
