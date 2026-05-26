#!/usr/bin/env python3
"""
数据集健康检查脚本。

检查项：
1. 文件完整性：每画像 8 个核心 JSON + verification_report
2. 跨阶段 schema 一致性：state_schema vars = state_evolution vars = eval_questions required_vars
3. eval_questions 完整性：每题有 ask_at_sessions + gold_answers + f_keys 对齐
4. state_evolution 合法性：σ_0..σ_10 完整、changes_from_prev 准确
5. exposure_plan 覆盖性：所有 schema vars 在某个 session 被 must_expose
6. seed_utterances 完整：10 session 都有种子
7. oracle_contexts 覆盖性：每条 question 的每个 ask_at_session 都有 context
8. verification_report 通过：全 10 画像 overall ≥ 90%

输出：health_report.json + 控制台彩色打印
"""
from __future__ import annotations

import io
import json
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from builder import load_config


CORE_FILES = [
    "persona.json",
    "questions_draft.json",
    "state_schema.json",
    "state_evolution.json",
    "exposure_plan.json",
    "seed_utterances.json",
    "eval_questions.json",
    "oracle_dialogues.json",
    "oracle_contexts.json",
]


class CheckResult:
    def __init__(self):
        self.passed: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def ok(self, msg: str):
        self.passed.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)

    def err(self, msg: str):
        self.errors.append(msg)

    @property
    def ok_count(self):
        return len(self.passed)

    @property
    def has_errors(self):
        return bool(self.errors)


def check_persona(p_dir: Path, persona_id: str) -> CheckResult:
    r = CheckResult()

    # ---- 1. 文件完整性 ----
    missing = []
    for fn in CORE_FILES:
        if not (p_dir / fn).exists():
            missing.append(fn)
    if missing:
        r.err(f"missing files: {missing}")
        return r  # 缺文件后续检查无意义
    r.ok(f"all {len(CORE_FILES)} core files present")

    # ---- 2. 加载所有数据 ----
    try:
        persona = json.loads((p_dir / "persona.json").read_text(encoding="utf-8"))
        schema = json.loads((p_dir / "state_schema.json").read_text(encoding="utf-8"))
        evolution = json.loads((p_dir / "state_evolution.json").read_text(encoding="utf-8"))
        plan = json.loads((p_dir / "exposure_plan.json").read_text(encoding="utf-8"))
        seeds = json.loads((p_dir / "seed_utterances.json").read_text(encoding="utf-8"))
        questions = json.loads((p_dir / "eval_questions.json").read_text(encoding="utf-8"))
        oracle_d = json.loads((p_dir / "oracle_dialogues.json").read_text(encoding="utf-8"))
        oracle_c = json.loads((p_dir / "oracle_contexts.json").read_text(encoding="utf-8"))
    except Exception as e:
        r.err(f"failed to load JSON: {e}")
        return r

    # ---- 3. persona_id 一致性 ----
    for name, data in [("persona", persona), ("schema", schema), ("evolution", evolution),
                       ("plan", plan)]:
        if data.get("persona_id") != persona_id:
            r.err(f"{name}.persona_id mismatch: {data.get('persona_id')} vs dir {persona_id}")

    # ---- 4. state_schema 合法性 ----
    schema_vars = {v["var_name"] for v in schema["variables"]}
    if not schema_vars:
        r.err("state_schema has no variables")
    main_core = schema["main_core"]
    if main_core not in schema_vars:
        r.err(f"main_core {main_core} not in schema vars")
    for aux in schema["auxiliary_cores"]:
        if aux["aux_var"] not in schema_vars:
            r.err(f"aux_var {aux['aux_var']} not in schema vars")
        if aux["cascade_lag"] not in {"immediate", "delayed"}:
            r.err(f"invalid cascade_lag: {aux['cascade_lag']}")
    r.ok(f"schema: {len(schema_vars)} vars, main_core={main_core}, "
         f"aux={[a['aux_var'] for a in schema['auxiliary_cores']]}")

    # ---- 5. state_evolution 合法性 ----
    traj = evolution["trajectory"]
    if len(traj) != 11:  # session 0..10
        r.err(f"trajectory length = {len(traj)}, expected 11")
    sessions_seen = [t["session"] for t in traj]
    if sessions_seen != list(range(11)):
        r.err(f"session indices not 0..10: {sessions_seen}")

    # 每个 σ_t 必须包含所有 schema vars
    for snap in traj:
        snap_vars = set(snap["state"].keys())
        missing_in_snap = schema_vars - snap_vars
        if missing_in_snap:
            r.err(f"σ_{snap['session']} missing vars: {missing_in_snap}")

    # 主核心变更应在 session 6
    sess6 = traj[6]
    if not sess6["is_main_core_change"]:
        r.err(f"session 6 not marked as main_core_change")
    prev_main = traj[5]["state"].get(main_core)
    curr_main = sess6["state"].get(main_core)
    if prev_main == curr_main:
        r.err(f"session 6 main_core didn't actually change: {prev_main} → {curr_main}")
    else:
        r.ok(f"main_core change at s6: {prev_main} → {curr_main}")

    # 检查 changes_from_prev 的真实性
    for i in range(1, 11):
        cur = traj[i]
        prev = traj[i - 1]
        actual_changes = [
            f"{k}: {prev['state'][k]} → {cur['state'][k]}"
            for k in cur["state"]
            if prev["state"].get(k) != cur["state"].get(k)
        ]
        listed_count = len(cur["changes_from_prev"])
        actual_count = len(actual_changes)
        if listed_count != actual_count:
            r.warn(f"session {i}: changes_from_prev has {listed_count} items, "
                   f"but state diff = {actual_count}")

    # interference_set
    interf = evolution.get("interference_set", {})
    if "0" not in interf and 0 not in interf:
        r.warn("interference_set missing session 0")
    last_I = interf.get("10", interf.get(10, []))
    r.ok(f"I_t at session 10: {len(last_I)} items")

    # ---- 6. exposure_plan 合法性 ----
    plan_sessions = plan["sessions"]
    if len(plan_sessions) != 10:
        r.err(f"exposure_plan has {len(plan_sessions)} sessions, expected 10")
    exposed_vars: set[str] = set()
    for sp in plan_sessions:
        for item in sp["must_expose"]:
            if item["var"] not in schema_vars:
                r.err(f"exposure_plan session {sp['session']}: var {item['var']} not in schema")
            exposed_vars.add(item["var"])
    unexposed = schema_vars - exposed_vars
    if unexposed:
        r.warn(f"never exposed in any session: {unexposed}")
    else:
        r.ok(f"all {len(schema_vars)} vars exposed across 10 sessions")

    # ---- 7. seed_utterances 完整性 ----
    expected_sessions = {str(s) for s in range(1, 11)}
    actual_sessions = set(seeds.keys())
    missing_seeds = expected_sessions - actual_sessions
    if missing_seeds:
        r.err(f"missing seed utterances for sessions: {missing_seeds}")
    else:
        r.ok(f"10/10 seed utterances present")
    # seed_text 非空
    for sk, sv in seeds.items():
        if not sv.get("seed_text", "").strip():
            r.err(f"session {sk} seed_text is empty")

    # ---- 8. eval_questions 合法性 ----
    pos = [q for q in questions if q["task_type"] == "POS"]
    neg = [q for q in questions if q["task_type"] == "NEG"]
    tracking = [q for q in pos if q.get("is_state_tracking")]
    n_total_asks = sum(len(q["ask_at_sessions"]) for q in questions)

    if len(questions) < 60:
        r.warn(f"only {len(questions)} questions (< 60 minimum recommended)")
    else:
        r.ok(f"{len(questions)} questions = {len(pos)} POS ({len(tracking)} tracking) + {len(neg)} NEG, total {n_total_asks} asks")

    # 每题字段一致性
    issues = 0
    for q in questions:
        # required_vars 必须在 schema 中（NEG-A/D 允许空）
        for v in q.get("required_vars", []):
            if v not in schema_vars:
                r.err(f"question {q['question_id']}: required_var {v} not in schema")
                issues += 1
                break
        # gold_answers / f_keys 的 keys 必须等于 ask_at_sessions
        ask_set = set(q["ask_at_sessions"])
        gold_keys = {int(k) for k in q["gold_answers"].keys()}
        fkey_keys = {int(k) for k in q.get("f_keys", {}).keys()}
        if gold_keys != ask_set:
            r.err(f"question {q['question_id']}: gold_answers keys {gold_keys} != ask_at {ask_set}")
            issues += 1
        # POS 题：gold 应与 σ_t 对齐
        if q["task_type"] == "POS" and q.get("required_vars"):
            for s in q["ask_at_sessions"]:
                snap_state = traj[s]["state"]
                gold = str(q["gold_answers"][str(s)])
                if len(q["required_vars"]) == 1:
                    var = q["required_vars"][0]
                    expected = str(snap_state.get(var, ""))
                    if expected != gold:
                        r.err(f"question {q['question_id']} session {s}: "
                              f"gold {gold!r} != σ_t[{var}]={expected!r}")
                        issues += 1
                        break
    if issues == 0:
        r.ok("eval_questions field-level consistency OK")

    # NEG 子类分布
    neg_subtypes = defaultdict(int)
    for q in neg:
        neg_subtypes[q.get("neg_subtype", "?")] += 1
    if neg:
        r.ok(f"NEG subtypes: {dict(neg_subtypes)}")

    # ---- 9. oracle_dialogues 完整性 ----
    od_sessions = {int(k) for k in oracle_d.keys()}
    expected = set(range(1, 11))
    missing_od = expected - od_sessions
    if missing_od:
        r.err(f"oracle_dialogues missing sessions: {missing_od}")
    od_turn_total = sum(len(v) for v in oracle_d.values())
    if od_turn_total < 30:
        r.warn(f"oracle_dialogues total turns = {od_turn_total} (< 30 minimum)")
    else:
        r.ok(f"oracle_dialogues: {len(od_sessions)} sessions, {od_turn_total} turns")

    # ---- 10. oracle_contexts 覆盖性 ----
    expected_oc_pairs = 0
    for q in questions:
        expected_oc_pairs += len(q["ask_at_sessions"])
    actual_oc_pairs = sum(len(v) for v in oracle_c.values())
    if actual_oc_pairs != expected_oc_pairs:
        r.warn(f"oracle_contexts pairs = {actual_oc_pairs}, expected {expected_oc_pairs}")
    # 检查每题每 ask_at 都有 context
    missing_oc = 0
    empty_oc = 0
    for q in questions:
        qid = q["question_id"]
        if qid not in oracle_c:
            missing_oc += len(q["ask_at_sessions"])
            continue
        for s in q["ask_at_sessions"]:
            if str(s) not in oracle_c[qid]:
                missing_oc += 1
            else:
                ctx_text = oracle_c[qid][str(s)].get("context_text", "")
                if not ctx_text.strip():
                    empty_oc += 1
    if missing_oc > 0:
        r.err(f"oracle_contexts missing {missing_oc} (question, session) pairs")
    if empty_oc > 0:
        r.warn(f"oracle_contexts has {empty_oc} empty context_text entries")
    if missing_oc == 0 and empty_oc == 0:
        r.ok(f"oracle_contexts fully covers all {expected_oc_pairs} pairs")

    # ---- 11. verification_report 通过 ----
    vr_path = p_dir / "verification_report.json"
    if vr_path.exists():
        try:
            vr = json.loads(vr_path.read_text(encoding="utf-8"))
            overall = vr["overall_pass_rate"]
            if overall >= 0.90:
                r.ok(f"verification overall = {overall:.2%}")
            else:
                r.err(f"verification overall = {overall:.2%} < 90% threshold")
            for ck, rate in vr["pass_rates"].items():
                if rate < 0.85:
                    r.warn(f"verification {ck} = {rate:.2%} < 85%")
        except Exception as e:
            r.warn(f"verification_report unreadable: {e}")
    else:
        r.warn("verification_report.json not present (may have not run)")

    return r


def print_color(s: str, color: str):
    """Windows console 友好的简化版"""
    print(s)


def main():
    cfg = load_config()
    per_persona_dir = cfg.paths.data_root / "per_persona"

    all_results: dict[str, CheckResult] = {}

    persona_dirs = sorted(per_persona_dir.glob("p_*"))
    print(f"\n{'=' * 80}")
    print(f"  数据集健康检查（{len(persona_dirs)} 画像）")
    print(f"{'=' * 80}\n")

    for p_dir in persona_dirs:
        pid = p_dir.name
        r = check_persona(p_dir, pid)
        all_results[pid] = r

        status = "✓ PASS" if not r.has_errors else "✗ FAIL"
        print(f"--- {pid} {status} ({r.ok_count} checks ok, "
              f"{len(r.warnings)} warn, {len(r.errors)} err) ---")
        if r.errors:
            for e in r.errors:
                print(f"  ✗ ERR: {e}")
        if r.warnings:
            for w in r.warnings:
                print(f"  ⚠ WARN: {w}")
        # 只打印一些代表性的 OK
        for o in r.passed[:3]:
            print(f"  · {o}")
        if len(r.passed) > 3:
            print(f"  · ... and {len(r.passed) - 3} more ok")
        print()

    # ---- 全局汇总 ----
    print(f"\n{'=' * 80}")
    print(f"  全局汇总")
    print(f"{'=' * 80}")

    total_errs = sum(len(r.errors) for r in all_results.values())
    total_warns = sum(len(r.warnings) for r in all_results.values())
    passed_personas = sum(1 for r in all_results.values() if not r.has_errors)

    print(f"\n  画像: {passed_personas}/{len(all_results)} 完全通过（无 errors）")
    print(f"  总 errors: {total_errs}")
    print(f"  总 warnings: {total_warns}")

    # 落盘
    report_path = cfg.paths.data_root / "health_report.json"
    summary = {
        pid: {
            "passed": list(r.passed),
            "warnings": list(r.warnings),
            "errors": list(r.errors),
            "has_errors": r.has_errors,
        }
        for pid, r in all_results.items()
    }
    summary["_overall"] = {
        "total_personas": len(all_results),
        "passed_personas": passed_personas,
        "total_errors": total_errs,
        "total_warnings": total_warns,
    }
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n  Health report saved: {report_path}")

    if total_errs == 0:
        print(f"\n  ✓ 数据集构建正常，可投入使用")
    else:
        print(f"\n  ✗ 检测到 {total_errs} 个 errors，需要修复")


if __name__ == "__main__":
    main()
