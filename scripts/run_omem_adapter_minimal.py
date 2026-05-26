from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from memory_eval.adapters import OMemAdapter, OMemAdapterConfig, load_runtime_credentials
from memory_eval.dataset import build_locomo_eval_samples, build_locomo_sample_registry
from memory_eval.eval_core import EvaluatorConfig, ParallelThreeProbeEvaluator


def _flatten_conversation(conv: Dict[str, Any]) -> List[Dict[str, Any]]:
    session_dt: Dict[int, str] = {}
    for key, value in conv.items():
        k = str(key)
        if k.startswith("session_") and k.endswith("_date_time"):
            seg = k.removeprefix("session_").removesuffix("_date_time")
            if seg.isdigit():
                session_dt[int(seg)] = str(value or "").strip()
    turns: List[Dict[str, Any]] = []
    ordered: List[tuple[int, List[Dict[str, Any]]]] = []
    for key, value in conv.items():
        k = str(key)
        if k.startswith("session_") and isinstance(value, list):
            seg = k.removeprefix("session_")
            if seg.isdigit():
                ordered.append((int(seg), value))
    ordered.sort(key=lambda x: x[0])
    turn_index = 0
    for session_idx, session in ordered:
        for item in session:
            turns.append(
                {
                    "turn_index": turn_index,
                    "speaker": str(item.get("speaker", "")).strip(),
                    "text": str(item.get("text", "")).strip(),
                    "timestamp": session_dt.get(session_idx, ""),
                }
            )
            turn_index += 1
    return turns


def _resolve_sample(dataset_path: Path, question_id: Optional[str], query: Optional[str], sample_id: Optional[str]):
    registry = build_locomo_sample_registry(str(dataset_path), f_key_mode="rule")
    if question_id:
        hit = registry.get_by_question_id(question_id)
        if hit is None:
            raise ValueError(f"question_id 未找到: {question_id}")
        return hit
    if query:
        hit = registry.find_by_query(query, sample_id=sample_id)
        if hit is None:
            raise ValueError(f"query 未找到样本: {query}")
        return hit
    all_samples = build_locomo_eval_samples(str(dataset_path), limit=1, f_key_mode="rule")
    if not all_samples:
        raise ValueError("数据集为空，无法构建样本。")
    return all_samples[0]


def _resolve_conversation(dataset_path: Path, target_sample_id: str) -> List[Dict[str, Any]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    for episode in data:
        if str(episode.get("sample_id", "")).strip() == str(target_sample_id).strip():
            return _flatten_conversation(episode.get("conversation", {}))
    raise ValueError(f"未找到 sample_id 对应会话: {target_sample_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal O-Mem adapter validation")
    parser.add_argument("--dataset", default="data/locomo10.json")
    parser.add_argument("--question-id", default=None)
    parser.add_argument("--query", default=None)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--keys", default="configs/keys.local.json")
    parser.add_argument("--use-real-omem", action="store_true")
    parser.add_argument("--use-llm-assist", action="store_true")
    parser.add_argument("--out", default="outputs/omem_adapter_minimal_result.json")
    args = parser.parse_args()

    dataset_path = (PROJECT_ROOT / args.dataset).resolve()
    keys_path = (PROJECT_ROOT / args.keys).resolve()
    out_path = (PROJECT_ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sample = _resolve_sample(dataset_path, args.question_id, args.query, args.sample_id)
    conversation = _resolve_conversation(dataset_path, sample.sample_id)

    require_creds = bool(args.use_real_omem or args.use_llm_assist)
    creds = load_runtime_credentials(str(keys_path), require_complete=require_creds)

    adapter = OMemAdapter(
        OMemAdapterConfig(
            use_real_omem=bool(args.use_real_omem),
            api_key=creds.get("api_key", ""),
            base_url=creds.get("base_url", ""),
            llm_model=creds.get("model", "") or "gpt-4o-mini",
            memory_dir=str((PROJECT_ROOT / "outputs" / "omem_memory").resolve()),
            retrieval_pieces=max(args.top_k, 10),
        )
    )
    run_ctx = adapter.ingest_conversation(sample.sample_id, conversation)
    trace = adapter.build_trace_for_query(
        run_ctx=run_ctx,
        query=sample.question,
        oracle_context=sample.oracle_context,
        top_k=args.top_k,
    )

    cfg = EvaluatorConfig(
        use_llm_assist=bool(args.use_llm_assist),
        llm_api_key=creds.get("api_key", ""),
        llm_base_url=creds.get("base_url", ""),
        llm_model=creds.get("model", "") or "gpt-4o-mini",
    )
    evaluator = ParallelThreeProbeEvaluator(cfg)
    result = evaluator.evaluate(sample.to_eval_sample(), trace)

    payload = {
        "summary": {
            "dataset": str(dataset_path),
            "question_id": sample.question_id,
            "sample_id": sample.sample_id,
            "use_real_omem": bool(args.use_real_omem),
            "use_llm_assist": bool(args.use_llm_assist),
            "mode": str(run_ctx.get("mode", "")) if isinstance(run_ctx, dict) else "",
        },
        "trace_preview": {
            "memory_count": len(trace.memory_view),
            "retrieved_count": len(trace.retrieved_items),
            "answer_oracle": trace.answer_oracle,
            "raw_trace": dict(trace.raw_trace),
        },
        "attribution_result": result.to_dict(),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"输出文件: {out_path}")


if __name__ == "__main__":
    main()
