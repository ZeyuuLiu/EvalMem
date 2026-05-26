from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize per-question baseline/eval output directories into JSON and CSV tables."
    )
    parser.add_argument("--baseline-dir", default="", help="Directory containing per-question baseline JSON files.")
    parser.add_argument("--eval-dir", default="", help="Directory containing per-question eval JSON files.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "question_output_summaries"),
        help="Directory to write summary artifacts.",
    )
    return parser.parse_args()


def _normalize_path(raw: str) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("path is empty")
    # Allow pasted linux paths that use backslashes.
    normalized = text.replace("\\", "/")
    return Path(normalized).expanduser().resolve()


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no", ""}:
            return False
    return bool(value)


def _counter_to_dict(counter: Counter[str]) -> Dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _sort_records(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("sample_id", "")),
            str(row.get("question_id", "")),
        ),
    )


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean_row = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(clean_row)


def _iter_question_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.json")):
        name = path.name
        if name in {"run_summary.json", "question_index.json", "result_bundle.json"}:
            continue
        yield path


def _load_records(root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for path in _iter_question_files(root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append({"file": str(path), "error": f"json_load_failed: {exc}"})
            continue
        if not isinstance(payload, dict):
            errors.append({"file": str(path), "error": "record is not a JSON object"})
            continue
        payload["_source_file"] = str(path)
        records.append(payload)
    return records, errors


def _task_accuracy_rows(counter_total: Counter[str], counter_correct: Counter[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    all_tasks = sorted(set(counter_total) | set(counter_correct))
    for task_type in all_tasks:
        total = int(counter_total.get(task_type, 0))
        correct = int(counter_correct.get(task_type, 0))
        rows.append(
            {
                "task_type": task_type,
                "count": total,
                "final_correct": correct,
                "final_accuracy": (correct / total) if total else 0.0,
            }
        )
    return rows


def summarize_baseline(root: Path) -> Dict[str, Any]:
    records, load_errors = _load_records(root)
    question_rows: List[Dict[str, Any]] = []
    sample_totals: Dict[str, Counter[str]] = defaultdict(Counter)
    task_total = Counter()
    task_correct = Counter()
    judge_label_counts = Counter()

    for record in records:
        sample_id = str(record.get("sample_id", "")).strip()
        question_id = str(record.get("question_id", "")).strip()
        task_type = str(record.get("task_type", "")).strip()
        final_correct = _safe_bool(record.get("final_correct", False))
        rule_correct = _safe_bool(record.get("rule_correct", False))
        llm_correct = _safe_bool(record.get("llm_correct", False))
        judge_label = str(record.get("judge_label", "")).strip() or "UNKNOWN"
        status = str(record.get("status", "")).strip()
        error_message = str(record.get("error", "")).strip()

        question_rows.append(
            {
                "sample_id": sample_id,
                "question_id": question_id,
                "task_type": task_type,
                "final_correct": final_correct,
                "rule_correct": rule_correct,
                "llm_correct": llm_correct,
                "judge_label": judge_label,
                "judge_reason": str(record.get("judge_reason", "")).strip(),
                "status": status,
                "error": error_message,
                "source_file": record.get("_source_file", ""),
            }
        )

        task_total[task_type] += 1
        task_correct[task_type] += int(final_correct)
        judge_label_counts[judge_label] += 1

        sample_totals[sample_id]["count"] += 1
        sample_totals[sample_id]["final_correct"] += int(final_correct)
        sample_totals[sample_id]["rule_correct"] += int(rule_correct)
        sample_totals[sample_id]["llm_correct"] += int(llm_correct)
        if task_type:
            sample_totals[sample_id][f"task_{task_type}"] += 1
            sample_totals[sample_id][f"task_{task_type}_correct"] += int(final_correct)
        if status:
            sample_totals[sample_id][f"status_{status}"] += 1

    question_rows = _sort_records(question_rows)
    sample_rows: List[Dict[str, Any]] = []
    for sample_id in sorted(sample_totals):
        counter = sample_totals[sample_id]
        total = int(counter.get("count", 0))
        correct = int(counter.get("final_correct", 0))
        sample_rows.append(
            {
                "sample_id": sample_id,
                "count": total,
                "final_correct": correct,
                "final_accuracy": (correct / total) if total else 0.0,
                "rule_correct": int(counter.get("rule_correct", 0)),
                "llm_correct": int(counter.get("llm_correct", 0)),
                "pos_count": int(counter.get("task_POS", 0)),
                "pos_correct": int(counter.get("task_POS_correct", 0)),
                "neg_count": int(counter.get("task_NEG", 0)),
                "neg_correct": int(counter.get("task_NEG_correct", 0)),
            }
        )

    total = len(question_rows)
    final_correct = sum(int(row["final_correct"]) for row in question_rows)
    summary = {
        "mode": "baseline",
        "source_dir": str(root),
        "count": total,
        "final_correct": final_correct,
        "final_accuracy": (final_correct / total) if total else 0.0,
        "task_breakdown": _task_accuracy_rows(task_total, task_correct),
        "judge_label_counts": _counter_to_dict(judge_label_counts),
        "load_errors": load_errors,
        "load_error_count": len(load_errors),
    }
    return {
        "summary": summary,
        "question_rows": question_rows,
        "sample_rows": sample_rows,
    }


def summarize_eval(root: Path) -> Dict[str, Any]:
    records, load_errors = _load_records(root)
    question_rows: List[Dict[str, Any]] = []
    sample_totals: Dict[str, Counter[str]] = defaultdict(Counter)
    task_total = Counter()
    task_correct = Counter()
    primary_cause_counts = Counter()
    final_judgement_counts = Counter()
    overall_defect_counts = Counter()
    per_probe_defect_counts: Dict[str, Counter[str]] = {
        "enc": Counter(),
        "ret": Counter(),
        "gen": Counter(),
    }
    state_counts: Dict[str, Counter[str]] = {
        "enc": Counter(),
        "ret": Counter(),
        "gen": Counter(),
    }

    for record in records:
        sample_id = str(record.get("sample_id", "")).strip()
        question_id = str(record.get("question_id", "")).strip()
        task_type = str(record.get("task_type", "")).strip()
        status = str(record.get("status", "")).strip()

        generation_correctness = record.get("generation_correctness", {}) or {}
        online_correctness = generation_correctness.get("online", {}) or {}
        oracle_correctness = generation_correctness.get("oracle", {}) or {}
        online_final_correct = _safe_bool(online_correctness.get("final_correct", False))
        oracle_final_correct = _safe_bool(oracle_correctness.get("final_correct", False))

        probe_states = record.get("probe_states", {}) or {}
        probe_defects = record.get("probe_defects", {}) or {}
        final_attribution = record.get("final_attribution", {}) or {}
        primary_cause = str(final_attribution.get("primary_cause", "")).strip() or "UNKNOWN"
        final_judgement = str(final_attribution.get("final_judgement", "")).strip() or "UNKNOWN"

        flat_defects: List[str] = []
        for probe in ("enc", "ret", "gen"):
            state = str(probe_states.get(probe, "")).strip() or "UNKNOWN"
            state_counts[probe][state] += 1
            defects = probe_defects.get(probe, [])
            if not isinstance(defects, list):
                defects = [str(defects)]
            cleaned = [str(item).strip() for item in defects if str(item).strip()]
            for defect in cleaned:
                overall_defect_counts[defect] += 1
                per_probe_defect_counts[probe][defect] += 1
            flat_defects.extend(cleaned)

        question_rows.append(
            {
                "sample_id": sample_id,
                "question_id": question_id,
                "task_type": task_type,
                "online_final_correct": online_final_correct,
                "oracle_final_correct": oracle_final_correct,
                "enc_state": str(probe_states.get("enc", "")).strip(),
                "ret_state": str(probe_states.get("ret", "")).strip(),
                "gen_state": str(probe_states.get("gen", "")).strip(),
                "enc_defects": "|".join(str(x).strip() for x in (probe_defects.get("enc", []) or []) if str(x).strip()),
                "ret_defects": "|".join(str(x).strip() for x in (probe_defects.get("ret", []) or []) if str(x).strip()),
                "gen_defects": "|".join(str(x).strip() for x in (probe_defects.get("gen", []) or []) if str(x).strip()),
                "all_defects": "|".join(sorted(set(flat_defects))),
                "primary_cause": primary_cause,
                "final_judgement": final_judgement,
                "status": status,
                "error_type": str(record.get("error_type", "")).strip(),
                "error_message": str(record.get("error_message", "")).strip(),
                "source_file": record.get("_source_file", ""),
            }
        )

        task_total[task_type] += 1
        task_correct[task_type] += int(online_final_correct)
        primary_cause_counts[primary_cause] += 1
        final_judgement_counts[final_judgement] += 1

        sample_totals[sample_id]["count"] += 1
        sample_totals[sample_id]["online_final_correct"] += int(online_final_correct)
        sample_totals[sample_id]["oracle_final_correct"] += int(oracle_final_correct)
        if task_type:
            sample_totals[sample_id][f"task_{task_type}"] += 1
            sample_totals[sample_id][f"task_{task_type}_correct"] += int(online_final_correct)
        sample_totals[sample_id][f"cause_{primary_cause}"] += 1

    question_rows = _sort_records(question_rows)
    sample_rows: List[Dict[str, Any]] = []
    for sample_id in sorted(sample_totals):
        counter = sample_totals[sample_id]
        total = int(counter.get("count", 0))
        online_correct = int(counter.get("online_final_correct", 0))
        sample_rows.append(
            {
                "sample_id": sample_id,
                "count": total,
                "online_final_correct": online_correct,
                "oracle_final_correct": int(counter.get("oracle_final_correct", 0)),
                "online_final_accuracy": (online_correct / total) if total else 0.0,
                "pos_count": int(counter.get("task_POS", 0)),
                "pos_correct": int(counter.get("task_POS_correct", 0)),
                "neg_count": int(counter.get("task_NEG", 0)),
                "neg_correct": int(counter.get("task_NEG_correct", 0)),
                "retrieval_primary_count": int(counter.get("cause_retrieval", 0)),
                "encoding_primary_count": int(counter.get("cause_encoding", 0)),
                "generation_primary_count": int(counter.get("cause_generation", 0)),
            }
        )

    total = len(question_rows)
    online_final_correct = sum(int(row["online_final_correct"]) for row in question_rows)
    oracle_final_correct = sum(int(row["oracle_final_correct"]) for row in question_rows)
    summary = {
        "mode": "eval",
        "source_dir": str(root),
        "count": total,
        "online_final_correct": online_final_correct,
        "online_final_accuracy": (online_final_correct / total) if total else 0.0,
        "oracle_final_correct": oracle_final_correct,
        "oracle_final_accuracy": (oracle_final_correct / total) if total else 0.0,
        "task_breakdown": _task_accuracy_rows(task_total, task_correct),
        "state_counts": {probe: _counter_to_dict(counter) for probe, counter in state_counts.items()},
        "defect_counts": _counter_to_dict(overall_defect_counts),
        "probe_defect_counts": {probe: _counter_to_dict(counter) for probe, counter in per_probe_defect_counts.items()},
        "primary_cause_counts": _counter_to_dict(primary_cause_counts),
        "final_judgement_counts": _counter_to_dict(final_judgement_counts),
        "load_errors": load_errors,
        "load_error_count": len(load_errors),
    }
    return {
        "summary": summary,
        "question_rows": question_rows,
        "sample_rows": sample_rows,
    }


def build_comparison(
    baseline_rows: List[Dict[str, Any]],
    eval_rows: List[Dict[str, Any]],
    baseline_dir: Path,
    eval_dir: Path,
) -> Dict[str, Any]:
    baseline_map = {
        (str(row.get("sample_id", "")), str(row.get("question_id", ""))): row
        for row in baseline_rows
    }
    eval_map = {
        (str(row.get("sample_id", "")), str(row.get("question_id", ""))): row
        for row in eval_rows
    }
    keys = sorted(set(baseline_map) | set(eval_map))
    rows: List[Dict[str, Any]] = []
    baseline_correct = 0
    eval_correct = 0
    baseline_correct_on_paired = 0
    eval_correct_on_paired = 0
    improved = 0
    dropped = 0
    matched = 0
    baseline_only = 0
    eval_only = 0

    for key in keys:
        base = baseline_map.get(key, {})
        ev = eval_map.get(key, {})
        base_ok = _safe_bool(base.get("final_correct", False))
        eval_ok = _safe_bool(ev.get("online_final_correct", False))
        baseline_correct += int(base_ok)
        eval_correct += int(eval_ok)
        if base and ev:
            baseline_correct_on_paired += int(base_ok)
            eval_correct_on_paired += int(eval_ok)
            if eval_ok and not base_ok:
                improved += 1
            elif base_ok and not eval_ok:
                dropped += 1
            elif base_ok == eval_ok:
                matched += 1
        elif base:
            baseline_only += 1
        elif ev:
            eval_only += 1
        rows.append(
            {
                "sample_id": key[0],
                "question_id": key[1],
                "task_type": ev.get("task_type", "") or base.get("task_type", ""),
                "baseline_final_correct": base_ok if base else "",
                "eval_online_final_correct": eval_ok if ev else "",
                "eval_oracle_final_correct": _safe_bool(ev.get("oracle_final_correct", False)) if ev else "",
                "enc_state": ev.get("enc_state", ""),
                "ret_state": ev.get("ret_state", ""),
                "gen_state": ev.get("gen_state", ""),
                "all_defects": ev.get("all_defects", ""),
                "primary_cause": ev.get("primary_cause", ""),
                "final_judgement": ev.get("final_judgement", ""),
                "baseline_source_file": base.get("source_file", ""),
                "eval_source_file": ev.get("source_file", ""),
            }
        )

    rows = _sort_records(rows)
    paired_count = sum(1 for key in keys if key in baseline_map and key in eval_map)
    summary = {
        "baseline_dir": str(baseline_dir),
        "eval_dir": str(eval_dir),
        "baseline_question_count": len(baseline_rows),
        "eval_question_count": len(eval_rows),
        "paired_question_count": paired_count,
        "baseline_only_question_count": baseline_only,
        "eval_only_question_count": eval_only,
        "baseline_correct_in_union": baseline_correct,
        "eval_online_correct_in_union": eval_correct,
        "baseline_correct_on_paired": baseline_correct_on_paired,
        "eval_online_correct_on_paired": eval_correct_on_paired,
        "improved_vs_baseline": improved,
        "dropped_vs_baseline": dropped,
        "same_correctness_on_paired": matched,
    }
    return {"summary": summary, "rows": rows}


def write_mode_outputs(output_dir: Path, prefix: str, payload: Dict[str, Any]) -> None:
    question_rows = payload["question_rows"]
    sample_rows = payload["sample_rows"]
    summary_payload = {
        "summary": payload["summary"],
        "artifacts": {
            "question_csv": f"{prefix}_question_rows.csv",
            "sample_csv": f"{prefix}_sample_rows.csv",
        },
    }
    _write_json(output_dir / f"{prefix}_summary.json", summary_payload)
    if question_rows:
        _write_csv(output_dir / f"{prefix}_question_rows.csv", question_rows, list(question_rows[0].keys()))
    if sample_rows:
        _write_csv(output_dir / f"{prefix}_sample_rows.csv", sample_rows, list(sample_rows[0].keys()))


def main() -> None:
    args = parse_args()
    if not args.baseline_dir and not args.eval_dir:
        raise SystemExit("At least one of --baseline-dir or --eval-dir must be provided.")

    output_dir = _normalize_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_payload: Dict[str, Any] | None = None
    eval_payload: Dict[str, Any] | None = None

    if args.baseline_dir:
        baseline_dir = _normalize_path(args.baseline_dir)
        baseline_payload = summarize_baseline(baseline_dir)
        write_mode_outputs(output_dir, "baseline", baseline_payload)
        print(
            json.dumps(
                {
                    "event": "baseline_summary_done",
                    "source_dir": str(baseline_dir),
                    "count": baseline_payload["summary"]["count"],
                    "final_accuracy": baseline_payload["summary"]["final_accuracy"],
                },
                ensure_ascii=False,
            )
        )

    if args.eval_dir:
        eval_dir = _normalize_path(args.eval_dir)
        eval_payload = summarize_eval(eval_dir)
        write_mode_outputs(output_dir, "eval", eval_payload)
        print(
            json.dumps(
                {
                    "event": "eval_summary_done",
                    "source_dir": str(eval_dir),
                    "count": eval_payload["summary"]["count"],
                    "online_final_accuracy": eval_payload["summary"]["online_final_accuracy"],
                },
                ensure_ascii=False,
            )
        )

    if baseline_payload is not None and eval_payload is not None:
        baseline_dir = _normalize_path(args.baseline_dir)
        eval_dir = _normalize_path(args.eval_dir)
        comparison = build_comparison(
            baseline_payload["question_rows"],
            eval_payload["question_rows"],
            baseline_dir,
            eval_dir,
        )
        _write_json(
            output_dir / "comparison_summary.json",
            {
                "summary": comparison["summary"],
                "artifacts": {"comparison_csv": "comparison_question_rows.csv"},
            },
        )
        if comparison["rows"]:
            _write_csv(
                output_dir / "comparison_question_rows.csv",
                comparison["rows"],
                list(comparison["rows"][0].keys()),
            )
        print(json.dumps({"event": "comparison_summary_done", **comparison["summary"]}, ensure_ascii=False))

    print(json.dumps({"ok": True, "output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
