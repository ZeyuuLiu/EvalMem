#!/usr/bin/env python3
"""
数据集 inspection 脚本。

跑完 01_build_per_persona.py --all 后用此脚本审查全部 10 画像的产物。
输出：
1. 每画像的统计摘要（题量、状态变化、ask 分布）
2. 跨画像汇总（总题量、子类分布、6 维多样性）
3. 抽样展示典型 POS / NEG 题
4. 公平性预检：每 session 的询问数是否在合理区间
"""
from __future__ import annotations

import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

# 强制 stdout UTF-8（Windows console 兼容）
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from builder import load_config


def banner(text: str, ch: str = "=") -> None:
    print(f"\n{ch * 70}\n{text}\n{ch * 70}")


def inspect_persona(persona_id: str, p_dir: Path) -> dict:
    """对单画像产生一份 summary dict。"""
    summary = {"persona_id": persona_id, "files": {}, "stats": {}}

    # ---- persona.json ----
    persona_path = p_dir / "persona.json"
    persona = json.loads(persona_path.read_text(encoding="utf-8"))
    summary["persona"] = {
        "name": persona["name"],
        "age": persona["age"],
        "stage": persona["stage"],
        "comm_style": persona["communication_style"],
        "rho": persona["rho"],
        "neg_sensitivity": persona["neg_sensitivity"],
        "preferred_domains": persona["preferred_domains"],
    }

    # ---- state_schema.json ----
    schema = json.loads((p_dir / "state_schema.json").read_text(encoding="utf-8"))
    summary["schema"] = {
        "n_vars": len(schema["variables"]),
        "main_core": schema["main_core"],
        "aux_cores": [
            {"var": c["aux_var"], "lag": c["cascade_lag"]}
            for c in schema["auxiliary_cores"]
        ],
        "domains_covered": sorted(set(v["domain"] for v in schema["variables"])),
    }

    # ---- state_evolution.json ----
    evolution = json.loads((p_dir / "state_evolution.json").read_text(encoding="utf-8"))
    changes_per_session = []
    rollback_count = 0
    cascade_count = 0
    for snap in evolution["trajectory"]:
        changes_per_session.append(len(snap["changes_from_prev"]))
        if snap.get("is_rollback"):
            rollback_count += 1
        if snap.get("is_aux_cascade"):
            cascade_count += 1
    summary["evolution"] = {
        "total_changes": sum(changes_per_session),
        "sessions_with_changes": sum(1 for n in changes_per_session if n > 0),
        "has_rollback": rollback_count > 0,
        "has_cascade": cascade_count > 0,
        "I_t_size_at_10": len(evolution["interference_set"].get("10", [])),
    }

    # ---- exposure_plan.json ----
    plan = json.loads((p_dir / "exposure_plan.json").read_text(encoding="utf-8"))
    exposure_items_per_session = [len(s["must_expose"]) for s in plan["sessions"]]
    summary["exposure"] = {
        "n_sessions": len(plan["sessions"]),
        "total_must_expose": sum(exposure_items_per_session),
        "avg_per_session": round(mean(exposure_items_per_session), 2),
    }

    # ---- seed_utterances.json ----
    seeds = json.loads((p_dir / "seed_utterances.json").read_text(encoding="utf-8"))
    summary["seeds"] = {"n_seeds": len(seeds)}

    # ---- eval_questions.json ----
    questions = json.loads((p_dir / "eval_questions.json").read_text(encoding="utf-8"))
    pos = [q for q in questions if q["task_type"] == "POS"]
    neg = [q for q in questions if q["task_type"] == "NEG"]
    tracking = [q for q in pos if q.get("is_state_tracking")]

    asks_per_session: dict = defaultdict(int)
    for q in questions:
        for s in q["ask_at_sessions"]:
            asks_per_session[s] += 1

    summary["questions"] = {
        "n_total_unique": len(questions),
        "n_pos": len(pos),
        "n_pos_fresh": len(pos) - len(tracking),
        "n_pos_tracking": len(tracking),
        "n_neg": len(neg),
        "neg_subtype": {
            sub: sum(1 for q in neg if q.get("neg_subtype") == sub)
            for sub in ["NEG-A", "NEG-B", "NEG-C", "NEG-D"]
        },
        "total_asks": sum(asks_per_session.values()),
        "asks_per_session": dict(sorted(asks_per_session.items())),
        "asks_min": min(asks_per_session.values()) if asks_per_session else 0,
        "asks_max": max(asks_per_session.values()) if asks_per_session else 0,
        "asks_mean": round(mean(asks_per_session.values()), 1) if asks_per_session else 0,
    }

    # ---- oracle_dialogues.json ----
    od = json.loads((p_dir / "oracle_dialogues.json").read_text(encoding="utf-8"))
    od_turns = [len(turns) for turns in od.values()]
    summary["oracle"] = {
        "n_sessions": len(od),
        "total_turns": sum(od_turns),
        "avg_turns_per_session": round(mean(od_turns), 1) if od_turns else 0,
    }

    return summary


def print_persona_summary(summary: dict) -> None:
    p = summary["persona"]
    print(f"\n--- {summary['persona_id']} ({p['name']}, {p['age']}岁, {p['stage']}) ---")
    print(f"  comm_style={p['comm_style']}, rho={p['rho']}, neg={p['neg_sensitivity']}, domains={p['preferred_domains']}")

    s = summary["schema"]
    print(f"  Schema: {s['n_vars']} vars, main_core={s['main_core']}")
    for aux in s["aux_cores"]:
        print(f"    aux: {aux['var']} ({aux['lag']})")

    e = summary["evolution"]
    print(f"  Evolution: {e['total_changes']} changes, "
          f"rollback={e['has_rollback']}, cascade={e['has_cascade']}, I_t@10={e['I_t_size_at_10']}")

    ex = summary["exposure"]
    print(f"  Exposure: {ex['total_must_expose']} items across {ex['n_sessions']} sessions (avg {ex['avg_per_session']}/s)")

    q = summary["questions"]
    print(f"  Questions: {q['n_total_unique']} unique = {q['n_pos']} POS ({q['n_pos_fresh']} fresh + {q['n_pos_tracking']} tracking) + {q['n_neg']} NEG")
    print(f"    NEG subtype: {q['neg_subtype']}")
    print(f"    Total asks: {q['total_asks']}, per-session range [{q['asks_min']}, {q['asks_max']}] (mean {q['asks_mean']})")
    print(f"    Asks per session: {q['asks_per_session']}")

    o = summary["oracle"]
    print(f"  Oracle: {o['n_sessions']} sessions, {o['total_turns']} turns (avg {o['avg_turns_per_session']}/s)")


def print_global_summary(all_summaries: list[dict]) -> None:
    banner("全局汇总 (10 画像)")

    # 总题量
    total_unique = sum(s["questions"]["n_total_unique"] for s in all_summaries)
    total_pos = sum(s["questions"]["n_pos"] for s in all_summaries)
    total_neg = sum(s["questions"]["n_neg"] for s in all_summaries)
    total_asks = sum(s["questions"]["total_asks"] for s in all_summaries)
    total_tracking = sum(s["questions"]["n_pos_tracking"] for s in all_summaries)

    print(f"  Unique questions: {total_unique}")
    print(f"  POS / NEG : {total_pos} / {total_neg}")
    print(f"  POS-tracking: {total_tracking}")
    print(f"  Total asks across all sessions: {total_asks}")
    print(f"  Avg asks per (persona, session): {total_asks / (len(all_summaries) * 10):.1f}")

    # NEG 子类汇总
    neg_total: Counter = Counter()
    for s in all_summaries:
        for sub, n in s["questions"]["neg_subtype"].items():
            neg_total[sub] += n
    print(f"  NEG subtype totals: {dict(neg_total)}")

    # 多样性
    print("\n  Persona diversity:")
    for dim in ["stage", "comm_style", "rho", "neg_sensitivity"]:
        counter: Counter = Counter()
        for s in all_summaries:
            counter[s["persona"][dim]] += 1
        print(f"    {dim}: {dict(counter)}")

    # 公平性预检
    print("\n  公平性预检 (asks per session):")
    asks_extremes = [(s["persona_id"], s["questions"]["asks_min"], s["questions"]["asks_max"])
                     for s in all_summaries]
    for pid, lo, hi in asks_extremes:
        flag = " ⚠" if hi - lo > 15 else ""
        print(f"    {pid}: min={lo}, max={hi}, spread={hi - lo}{flag}")

    # State evolution diversity
    rollback_count = sum(1 for s in all_summaries if s["evolution"]["has_rollback"])
    cascade_count = sum(1 for s in all_summaries if s["evolution"]["has_cascade"])
    print(f"\n  Evolution diversity:")
    print(f"    Personas with rollback: {rollback_count}/10")
    print(f"    Personas with cascade : {cascade_count}/10")


def main():
    cfg = load_config()
    data_root = cfg.paths.data_root
    per_persona_dir = data_root / "per_persona"

    persona_dirs = sorted(per_persona_dir.glob("p_*"))
    if not persona_dirs:
        print(f"No persona directories found at {per_persona_dir}")
        sys.exit(1)

    banner(f"DynaMem-Bench 数据集审查（{len(persona_dirs)} 画像）")

    all_summaries = []
    for p_dir in persona_dirs:
        pid = p_dir.name
        if not (p_dir / "eval_questions.json").exists():
            print(f"\n--- {pid}: 数据未生成完整，跳过 ---")
            continue
        try:
            summary = inspect_persona(pid, p_dir)
            print_persona_summary(summary)
            all_summaries.append(summary)
        except Exception as e:
            print(f"\n--- {pid}: ERROR: {e} ---")

    if all_summaries:
        print_global_summary(all_summaries)

    # 落盘 audit 报告
    audit_path = data_root / "audit_report.json"
    audit_path.write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n\nAudit report written: {audit_path}")


if __name__ == "__main__":
    main()
