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
        description="Analyze eval outputs from per-question directories or result bundles and write JSON/CSV/Markdown reports."
    )
    parser.add_argument("--omem-input", required=True, help="O-Mem eval per-question directory or bundle json path.")
    parser.add_argument("--membox-input", required=True, help="MemBox eval per-question directory or bundle json path.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "conv26_eval_reports"),
        help="Directory for generated report artifacts.",
    )
    return parser.parse_args()


def _normalize_path(raw: str) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("path is empty")
    return Path(text.replace("\\", "/")).expanduser().resolve()


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


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _iter_record_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.json")):
        if path.name in {"run_summary.json", "question_index.json", "result_bundle.json"}:
            continue
        yield path


def _load_dir_records(root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    load_errors: List[Dict[str, Any]] = []
    for path in _iter_record_files(root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            load_errors.append({"file": str(path), "error": f"json_load_failed: {exc}"})
            continue
        if not isinstance(payload, dict):
            load_errors.append({"file": str(path), "error": "record is not a JSON object"})
            continue
        payload["_source_file"] = str(path)
        records.append(payload)
    return records, load_errors


def _bundle_result_to_record(item: Dict[str, Any], root: Path) -> Dict[str, Any]:
    generation = {
        "online": dict((item.get("generation_correctness", {}) or {}).get("online", {})),
        "oracle": dict((item.get("generation_correctness", {}) or {}).get("oracle", {})),
    }
    final_attr = dict(item.get("final_attribution", {}) or {})
    return {
        "question_id": str(item.get("question_id", "")).strip(),
        "sample_id": str(item.get("sample_id", "")).strip(),
        "task_type": str(item.get("task_type", "")).strip(),
        "question": str(item.get("question", "")).strip(),
        "answer_gold": str(item.get("answer_gold", "")).strip(),
        "answer_online": str(generation.get("online", {}).get("answer_online", item.get("answer_online", ""))).strip(),
        "answer_oracle": str(generation.get("oracle", {}).get("answer_oracle", item.get("answer_oracle", ""))).strip(),
        "generation_correctness": generation,
        "probe_states": dict(item.get("states", {})),
        "probe_defects": _group_defects_by_probe(item),
        "final_attribution": {
            "primary_cause": str(final_attr.get("primary_cause", item.get("primary_cause", ""))).strip(),
            "final_judgement": str(final_attr.get("final_judgement", item.get("final_judgement", ""))).strip(),
        },
        "_source_file": str(root),
    }


def _bundle_error_to_record(item: Dict[str, Any], root: Path) -> Dict[str, Any]:
    return {
        "question_id": str(item.get("question_id", "")).strip(),
        "sample_id": str(item.get("sample_id", "")).strip(),
        "task_type": str(item.get("task_type", "")).strip(),
        "question": str(item.get("question", "")).strip(),
        "status": str(item.get("status", "EVAL_ERROR")).strip() or "EVAL_ERROR",
        "error_type": str(item.get("error_type", "")).strip(),
        "error_message": str(item.get("error_message", item.get("error", ""))).strip(),
        "_source_file": str(root),
    }


def _group_defects_by_probe(item: Dict[str, Any]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {"enc": [], "ret": [], "gen": []}
    probe_results = item.get("probe_results", {}) or {}
    if isinstance(probe_results, dict):
        for probe in ("enc", "ret", "gen"):
            probe_payload = probe_results.get(probe, {}) or {}
            defects = probe_payload.get("defects", [])
            if isinstance(defects, list):
                grouped[probe] = [str(x).strip() for x in defects if str(x).strip()]
    if any(grouped.values()):
        return grouped
    fallback = item.get("defects", [])
    if isinstance(fallback, list):
        grouped["ret"] = [str(x).strip() for x in fallback if str(x).strip()]
    return grouped


def _load_bundle_records(path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    errors = payload.get("errors", [])
    records: List[Dict[str, Any]] = []
    load_errors: List[Dict[str, Any]] = []
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict):
                records.append(_bundle_result_to_record(item, path))
    if isinstance(errors, list):
        for item in errors:
            if isinstance(item, dict):
                records.append(_bundle_error_to_record(item, path))
    return records, load_errors


def load_records(path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if path.is_dir():
        return _load_dir_records(path)
    if path.is_file():
        return _load_bundle_records(path)
    raise FileNotFoundError(str(path))


def _sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda row: (str(row.get("sample_id", "")), str(row.get("question_id", ""))))


def _top_counter(counter: Counter[str], limit: int = 10) -> List[Dict[str, Any]]:
    return [{"label": key, "count": count} for key, count in counter.most_common(limit)]


def _truncate(text: str, max_len: int = 220) -> str:
    raw = str(text or "").strip().replace("\n", " ")
    if len(raw) <= max_len:
        return raw
    return raw[: max_len - 3] + "..."


def analyze_eval_records(name: str, source_path: Path, records: List[Dict[str, Any]], load_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
    question_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    task_total = Counter()
    task_correct = Counter()
    task_error = Counter()
    state_counts = {"enc": Counter(), "ret": Counter(), "gen": Counter()}
    defect_counts = Counter()
    probe_defect_counts = {"enc": Counter(), "ret": Counter(), "gen": Counter()}
    primary_cause_counts = Counter()
    final_judgement_counts = Counter()
    error_type_counts = Counter()
    error_message_counts = Counter()
    sample_buckets: Dict[str, Counter[str]] = defaultdict(Counter)
    examples = {
        "eval_errors": [],
        "retrieval_failures": [],
        "encoding_failures": [],
        "generation_failures": [],
    }

    for record in _sort_rows(records):
        sample_id = str(record.get("sample_id", "")).strip()
        question_id = str(record.get("question_id", "")).strip()
        task_type = str(record.get("task_type", "")).strip() or "UNKNOWN"
        status = str(record.get("status", "")).strip()
        is_error = status == "EVAL_ERROR"
        nested_error = record.get("error", {}) or {}
        if not isinstance(nested_error, dict):
            nested_error = {}

        generation = record.get("generation_correctness", {}) or {}
        online = generation.get("online", {}) or {}
        oracle = generation.get("oracle", {}) or {}
        online_final_correct = _safe_bool(online.get("final_correct", False))
        oracle_final_correct = _safe_bool(oracle.get("final_correct", False))

        probe_states = record.get("probe_states", {}) or {}
        probe_defects = record.get("probe_defects", {}) or {}
        final_attr = record.get("final_attribution", {}) or {}
        primary_cause = str(final_attr.get("primary_cause", "")).strip() or ("eval_error" if is_error else "UNKNOWN")
        final_judgement = str(final_attr.get("final_judgement", "")).strip() or ("EVAL_ERROR" if is_error else "UNKNOWN")
        error_type = str(record.get("error_type", nested_error.get("error_type", ""))).strip()
        error_message = str(record.get("error_message", nested_error.get("error_message", nested_error.get("error", "")))).strip()

        enc_state = str(probe_states.get("enc", "")).strip() or ("UNKNOWN" if not is_error else "")
        ret_state = str(probe_states.get("ret", "")).strip() or ("UNKNOWN" if not is_error else "")
        gen_state = str(probe_states.get("gen", "")).strip() or ("UNKNOWN" if not is_error else "")

        all_defects: List[str] = []
        for probe, state in (("enc", enc_state), ("ret", ret_state), ("gen", gen_state)):
            if state:
                state_counts[probe][state] += 1
            defects = probe_defects.get(probe, [])
            if not isinstance(defects, list):
                defects = [defects]
            cleaned = [str(x).strip() for x in defects if str(x).strip()]
            for defect in cleaned:
                defect_counts[defect] += 1
                probe_defect_counts[probe][defect] += 1
            all_defects.extend(cleaned)

        task_total[task_type] += 1
        task_correct[task_type] += int(online_final_correct)
        task_error[task_type] += int(is_error)
        primary_cause_counts[primary_cause] += 1
        final_judgement_counts[final_judgement] += 1

        if is_error:
            error_type_counts[error_type or "UNKNOWN"] += 1
            error_message_counts[_truncate(error_message, 140) or "UNKNOWN"] += 1
            if len(examples["eval_errors"]) < 5:
                examples["eval_errors"].append(
                    {
                        "question_id": question_id,
                        "question": str(record.get("question", "")).strip(),
                        "error_type": error_type,
                        "error_message": error_message,
                        "source_file": str(record.get("_source_file", "")),
                    }
                )

        if primary_cause == "retrieval" and len(examples["retrieval_failures"]) < 5:
            examples["retrieval_failures"].append(
                {
                    "question_id": question_id,
                    "question": str(record.get("question", "")).strip(),
                    "ret_state": ret_state,
                    "ret_defects": list(probe_defects.get("ret", []) or []),
                    "answer_online": str(record.get("answer_online", "")).strip(),
                }
            )
        if primary_cause == "encoding" and len(examples["encoding_failures"]) < 5:
            examples["encoding_failures"].append(
                {
                    "question_id": question_id,
                    "question": str(record.get("question", "")).strip(),
                    "enc_state": enc_state,
                    "enc_defects": list(probe_defects.get("enc", []) or []),
                    "answer_online": str(record.get("answer_online", "")).strip(),
                }
            )
        if primary_cause == "generation" and len(examples["generation_failures"]) < 5:
            examples["generation_failures"].append(
                {
                    "question_id": question_id,
                    "question": str(record.get("question", "")).strip(),
                    "gen_state": gen_state,
                    "gen_defects": list(probe_defects.get("gen", []) or []),
                    "answer_online": str(record.get("answer_online", "")).strip(),
                    "answer_oracle": str(record.get("answer_oracle", "")).strip(),
                }
            )

        sample_buckets[sample_id]["count"] += 1
        sample_buckets[sample_id]["online_final_correct"] += int(online_final_correct)
        sample_buckets[sample_id]["oracle_final_correct"] += int(oracle_final_correct)
        sample_buckets[sample_id]["errors"] += int(is_error)
        sample_buckets[sample_id][f"cause_{primary_cause}"] += 1

        question_rows.append(
            {
                "system": name,
                "sample_id": sample_id,
                "question_id": question_id,
                "task_type": task_type,
                "status": status or "OK",
                "online_final_correct": online_final_correct,
                "oracle_final_correct": oracle_final_correct,
                "enc_state": enc_state,
                "ret_state": ret_state,
                "gen_state": gen_state,
                "all_defects": "|".join(sorted(set(all_defects))),
                "primary_cause": primary_cause,
                "final_judgement": final_judgement,
                "error_type": error_type,
                "error_message": error_message,
                "question": str(record.get("question", "")).strip(),
                "answer_online": str(record.get("answer_online", "")).strip(),
                "answer_oracle": str(record.get("answer_oracle", "")).strip(),
                "source_file": str(record.get("_source_file", "")),
            }
        )

    for sample_id in sorted(sample_buckets):
        bucket = sample_buckets[sample_id]
        total = int(bucket.get("count", 0))
        online_correct = int(bucket.get("online_final_correct", 0))
        sample_rows.append(
            {
                "system": name,
                "sample_id": sample_id,
                "count": total,
                "online_final_correct": online_correct,
                "oracle_final_correct": int(bucket.get("oracle_final_correct", 0)),
                "errors": int(bucket.get("errors", 0)),
                "online_final_accuracy": (online_correct / total) if total else 0.0,
                "encoding_primary_count": int(bucket.get("cause_encoding", 0)),
                "retrieval_primary_count": int(bucket.get("cause_retrieval", 0)),
                "generation_primary_count": int(bucket.get("cause_generation", 0)),
                "eval_error_count": int(bucket.get("cause_eval_error", 0)),
            }
        )

    total = len(question_rows)
    online_final_correct = sum(int(row["online_final_correct"]) for row in question_rows)
    oracle_final_correct = sum(int(row["oracle_final_correct"]) for row in question_rows)
    error_count = sum(1 for row in question_rows if row["status"] == "EVAL_ERROR")

    summary = {
        "system": name,
        "source_path": str(source_path),
        "count": total,
        "online_final_correct": online_final_correct,
        "online_final_accuracy": (online_final_correct / total) if total else 0.0,
        "oracle_final_correct": oracle_final_correct,
        "oracle_final_accuracy": (oracle_final_correct / total) if total else 0.0,
        "error_count": error_count,
        "error_rate": (error_count / total) if total else 0.0,
        "task_breakdown": [
            {
                "task_type": task,
                "count": int(task_total[task]),
                "online_final_correct": int(task_correct[task]),
                "errors": int(task_error[task]),
                "online_final_accuracy": (task_correct[task] / task_total[task]) if task_total[task] else 0.0,
            }
            for task in sorted(task_total)
        ],
        "state_counts": {probe: dict(counter) for probe, counter in state_counts.items()},
        "defect_counts": dict(defect_counts),
        "probe_defect_counts": {probe: dict(counter) for probe, counter in probe_defect_counts.items()},
        "primary_cause_counts": dict(primary_cause_counts),
        "final_judgement_counts": dict(final_judgement_counts),
        "error_type_counts": dict(error_type_counts),
        "top_error_messages": _top_counter(error_message_counts, limit=5),
        "load_error_count": len(load_errors),
        "load_errors": load_errors,
    }
    return {
        "summary": summary,
        "question_rows": question_rows,
        "sample_rows": sample_rows,
        "examples": examples,
    }


def _format_counter_lines(counter_map: Dict[str, int], empty_text: str = "无") -> List[str]:
    if not counter_map:
        return [f"- {empty_text}"]
    return [f"- `{key}`: {value}" for key, value in sorted(counter_map.items(), key=lambda item: (-item[1], item[0]))]


def _format_example_block(title: str, rows: List[Dict[str, Any]]) -> List[str]:
    lines = [f"### {title}"]
    if not rows:
        lines.append("- 无")
        return lines
    for row in rows:
        qid = row.get("question_id", "")
        question = _truncate(row.get("question", ""), 120)
        detail_bits = []
        if row.get("error_type"):
            detail_bits.append(str(row["error_type"]))
        if row.get("ret_defects"):
            detail_bits.append("ret=" + ",".join(row["ret_defects"]))
        if row.get("enc_defects"):
            detail_bits.append("enc=" + ",".join(row["enc_defects"]))
        if row.get("gen_defects"):
            detail_bits.append("gen=" + ",".join(row["gen_defects"]))
        if row.get("error_message"):
            detail_bits.append(_truncate(row["error_message"], 140))
        lines.append(f"- `{qid}`: {question}" + (f" | {' ; '.join(detail_bits)}" if detail_bits else ""))
    return lines


def build_markdown_report(omem: Dict[str, Any], membox: Dict[str, Any]) -> str:
    osum = omem["summary"]
    msum = membox["summary"]
    same_coverage = osum["count"] == msum["count"]
    lines: List[str] = []
    lines.append("# conv-26 Eval 结果说明")
    lines.append("")
    lines.append("## 总览")
    lines.append(
        f"- O-Mem 共 {osum['count']} 题，online 正确 {osum['online_final_correct']} 题，online 准确率 {osum['online_final_accuracy']:.4f}，oracle 准确率 {osum['oracle_final_accuracy']:.4f}，EVAL_ERROR {osum['error_count']} 题。"
    )
    lines.append(
        f"- MemBox 共 {msum['count']} 题，online 正确 {msum['online_final_correct']} 题，online 准确率 {msum['online_final_accuracy']:.4f}，oracle 准确率 {msum['oracle_final_accuracy']:.4f}，EVAL_ERROR {msum['error_count']} 题。"
    )
    if same_coverage:
        lines.append("- 两边当前覆盖题数一致，可以直接对比错误结构。")
    else:
        lines.append(
            f"- 两边当前覆盖题数不一致：O-Mem 为 {osum['count']} 题，MemBox 为 {msum['count']} 题，因此更适合比较错误模式与归因结构，而不适合直接比较绝对正确题数。"
        )
    lines.append("")
    lines.append("## O-Mem")
    lines.append(f"- 来源：`{osum['source_path']}`")
    lines.append(f"- 结果判断：已经产出完整逐题文件，说明这轮 `conv-26` 评估并非空跑；当前更像是“部分题成功、部分题失败/中断后仍落盘”的状态。")
    lines.append(f"- 主归因分布：")
    lines.extend(_format_counter_lines(osum["primary_cause_counts"]))
    lines.append(f"- 三层状态分布：")
    for probe in ("enc", "ret", "gen"):
        counts = osum["state_counts"].get(probe, {})
        pretty = ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda item: (-item[1], item[0])))
        lines.append(f"- `{probe}`: {pretty or '无'}")
    lines.append(f"- 高频缺陷：")
    lines.extend(_format_counter_lines(osum["defect_counts"]))
    lines.append("")
    lines.append("## MemBox")
    lines.append(f"- 来源：`{msum['source_path']}`")
    lines.append(f"- 结果判断：本轮是“命令已结束并写出 bundle，但评估过程中出现大量 strict-mode 运行时错误”。")
    lines.append(
        f"- 直接证据：{msum['error_count']}/{msum['count']} 题为 `EVAL_ERROR`，online 准确率仅 {msum['online_final_accuracy']:.4f}。"
    )
    lines.append(f"- 主归因分布：")
    lines.extend(_format_counter_lines(msum["primary_cause_counts"]))
    lines.append(f"- 高频错误类型：")
    lines.extend(_format_counter_lines(msum["error_type_counts"]))
    lines.append("")
    lines.append("## 关键结论")
    lines.append(
        "- O-Mem：本轮 `conv-26` 已经有大量逐题结果落盘，可用于继续做统计分析；问题重点不再是“完全没跑出来”，而是要区分成功题、EVAL_ERROR 题和各层缺陷模式。"
    )
    lines.append(
        "- MemBox：当前主要问题不是记忆本身全 MISS，而是 generation strict mode 与 judge 输出契约不一致，导致大量题在归因前就被 RuntimeError 中断。"
    )
    lines.append(
        "- 从 MemBox 日志可见，多处报错为 `POS failure requires llm_judgement.substate in {GF, GRF}, got 'NONE'`；这说明 judge 返回了 `correct/substate=NONE` 的组合，而 generation 严格后处理不接受这个组合。"
    )
    if same_coverage:
        lines.append("- 两边都受到同一类 strict-mode 契约问题影响，因此当前更应将结果视为“评测链路稳定性诊断”，而不是系统能力的最终排名。")
    else:
        lines.append(
            f"- O-Mem 与 MemBox 此轮结果不宜直接横向比较绝对准确率，因为两边覆盖题数不同：O-Mem 当前为 {osum['count']} 题，MemBox 当前为 {msum['count']} 题。"
        )
    lines.append("")
    lines.extend(_format_example_block("O-Mem 代表性检索失败", omem["examples"]["retrieval_failures"]))
    lines.append("")
    lines.extend(_format_example_block("O-Mem 代表性生成失败", omem["examples"]["generation_failures"]))
    lines.append("")
    lines.extend(_format_example_block("MemBox Eval 错误样例", membox["examples"]["eval_errors"]))
    lines.append("")
    lines.append("## 后续建议")
    lines.append("- 先修 MemBox 的 generation strict-mode 子状态契约，再重跑不带 `limit=10` 的 conv-26 eval。")
    lines.append("- 对 O-Mem 基于本次逐题目录继续汇总：按 POS/NEG、primary_cause、RF/LATE/NOI/GF/GRF 分层统计，并抽取错误样例做论文分析。")
    lines.append("- 若你需要论文表格，我可以在这份自动报告基础上继续生成更精炼的“实验结果分析”版 Markdown。")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    omem_path = _normalize_path(args.omem_input)
    membox_path = _normalize_path(args.membox_input)
    output_dir = _normalize_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    omem_records, omem_load_errors = load_records(omem_path)
    membox_records, membox_load_errors = load_records(membox_path)

    omem_report = analyze_eval_records("o_mem", omem_path, omem_records, omem_load_errors)
    membox_report = analyze_eval_records("membox", membox_path, membox_records, membox_load_errors)

    _json_dump(output_dir / "omem_eval_summary.json", omem_report)
    _json_dump(output_dir / "membox_eval_summary.json", membox_report)
    _write_csv(output_dir / "omem_eval_questions.csv", omem_report["question_rows"])
    _write_csv(output_dir / "membox_eval_questions.csv", membox_report["question_rows"])
    _write_csv(output_dir / "omem_eval_samples.csv", omem_report["sample_rows"])
    _write_csv(output_dir / "membox_eval_samples.csv", membox_report["sample_rows"])

    markdown = build_markdown_report(omem_report, membox_report)
    (output_dir / "conv26_eval_report.md").write_text(markdown, encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": True,
                "output_dir": str(output_dir),
                "omem_count": omem_report["summary"]["count"],
                "membox_count": membox_report["summary"]["count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
