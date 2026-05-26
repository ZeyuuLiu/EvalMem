from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

# Allow running the script directly without requiring PYTHONPATH setup.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from memory_eval.dataset import build_locomo_eval_samples, build_locomo_sample_registry


def _load_keys_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    try:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as e:
        return {"exists": True, "parse_ok": False, "error": str(e)}
    return {
        "exists": True,
        "parse_ok": True,
        "has_api_key": bool(obj.get("api_key")),
        "has_base_url": bool(obj.get("base_url")),
        "model": obj.get("model"),
        "temperature": obj.get("temperature", 0.0),
    }


def _extract_f_key_via_llm(
    evidence_texts: List[str],
    evidence_with_time: List[str],
    question: str,
    answer_gold: str,
    llm_cfg: Dict[str, Any],
) -> List[str]:
    api_key = str(llm_cfg.get("api_key", "")).strip()
    base_url = str(llm_cfg.get("base_url", "")).strip().rstrip("/")
    model = str(llm_cfg.get("model", "gpt-4o-mini")).strip()
    temperature = float(llm_cfg.get("temperature", 0.0) or 0.0)

    if not api_key or not base_url:
        return list(evidence_with_time) if evidence_with_time else list(evidence_texts)

    prompt = (
        "You are extracting key facts for memory evaluation.\n"
        "Return strict JSON only with schema: {\"f_key\": [\"...\"]}.\n"
        "Each item must preserve time information if present.\n"
        "Keep concise and factual. Do not invent.\n\n"
        f"Question: {question}\n"
        f"Gold answer: {answer_gold}\n"
        "Evidence with time:\n"
        + "\n".join(evidence_with_time)
        + "\n\nOutput JSON:"
    )
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": "You output strict JSON only."},
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        url=f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except Exception:
        return list(evidence_with_time) if evidence_with_time else list(evidence_texts)

    try:
        out = json.loads(raw)
        content = out["choices"][0]["message"]["content"]
        obj = json.loads(content)
        f_key = obj.get("f_key", [])
        if isinstance(f_key, list):
            cleaned = [str(x).strip() for x in f_key if str(x).strip()]
            if cleaned:
                return cleaned
    except Exception:
        pass
    return list(evidence_with_time) if evidence_with_time else list(evidence_texts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LOCOMO evaluation sample demo")
    parser.add_argument("--dataset", default="../data/locomo10.json", help="Path to locomo10.json")
    parser.add_argument("--limit", type=int, default=5, help="Max number of generated samples")
    parser.add_argument("--out", default="outputs/demo_locomo_samples.json", help="Output demo json path")
    parser.add_argument("--keys", default="../configs/keys.local.json", help="Path to keys.local.json")
    parser.add_argument(
        "--fkey-source",
        choices=["rule", "llm"],
        default="rule",
        help="How to construct f_key: rule uses evidence directly; llm calls model extraction.",
    )
    parser.add_argument("--query", default=None, help="Optional query string to resolve one sample from registry.")
    parser.add_argument("--question-id", default=None, help="Optional question_id to resolve one sample from registry.")
    parser.add_argument("--sample-id", default=None, help="Optional sample_id disambiguation for --query lookup.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    dataset_path = (script_dir / args.dataset).resolve()
    keys_path = (script_dir / args.keys).resolve()
    out_path = (script_dir.parent / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keys_info = _load_keys_file(keys_path)
    llm_cfg = {}
    if keys_info.get("parse_ok"):
        try:
            raw_keys = json.loads(keys_path.read_text(encoding="utf-8-sig"))
            llm_cfg = {
                "api_key": raw_keys.get("api_key"),
                "base_url": raw_keys.get("base_url"),
                "model": raw_keys.get("model"),
                "temperature": raw_keys.get("temperature", 0.0),
            }
        except Exception:
            llm_cfg = {}
    if args.fkey_source == "llm":
        samples = build_locomo_eval_samples(
            str(dataset_path),
            limit=args.limit,
            f_key_mode="llm",
            f_key_extractor=lambda ev, ev_t, q, a: _extract_f_key_via_llm(ev, ev_t, q, a, llm_cfg),
        )
    else:
        samples = build_locomo_eval_samples(str(dataset_path), limit=args.limit, f_key_mode="rule")

    # Registry lookup is kept rule-based to avoid accidental full-dataset LLM extraction calls.
    registry = build_locomo_sample_registry(str(dataset_path), f_key_mode="rule")
    resolved = None
    if args.question_id:
        hit = registry.get_by_question_id(args.question_id)
        resolved = asdict(hit) if hit else None
    elif args.query:
        hit = registry.find_by_query(args.query, sample_id=args.sample_id)
        resolved = asdict(hit) if hit else None

    payload = {
        "summary": {
            "dataset_path": str(dataset_path),
            "total_samples_built": len(samples),
            "f_key_source": args.fkey_source,
            "keys_file_check": keys_info,
        },
        "samples": [asdict(s) for s in samples],
        "resolved_sample": resolved,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Demo completed.")
    print(f"Output: {out_path}")
    print(f"Samples built: {len(samples)}")
    print(f"Keys file exists: {keys_info.get('exists')}")
    print(f"Keys parse ok: {keys_info.get('parse_ok', False)}")
    if args.query or args.question_id:
        print(f"Resolved sample found: {bool(resolved)}")


if __name__ == "__main__":
    main()
