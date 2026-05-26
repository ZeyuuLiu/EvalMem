#!/usr/bin/env python3
"""
阶段 8 异源验证 orchestrator。

用法：
    python scripts/03_run_verification.py --persona-id p_01
    python scripts/03_run_verification.py --all
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from builder import build_client, load_config
from builder.verifier import verify_persona


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona-id", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    cfg = load_config()
    client = build_client(cfg, cache_subdir="llm_responses_verifier")

    if args.persona_id:
        targets = [args.persona_id]
    elif args.all:
        per_persona_dir = cfg.paths.data_root / "per_persona"
        targets = sorted([d.name for d in per_persona_dir.glob("p_*")
                          if (d / "eval_questions.json").exists()])
    else:
        print("Usage: --persona-id p_01  OR  --all")
        sys.exit(1)

    all_reports = []
    for pid in targets:
        print(f"\n========== verifying {pid} ==========")
        try:
            report = verify_persona(pid, cfg, client)
            all_reports.append(report)
            status = "✓" if report["all_thresholds_met"] else "✗"
            print(f"{status} {pid}: overall={report['overall_pass_rate']:.2%}")
            for ck, rate in report["pass_rates"].items():
                marker = "✓" if rate >= 0.85 else "✗"
                print(f"  {marker} {ck}: {rate:.2%}")
        except Exception as e:
            print(f"ERROR for {pid}: {e}")
            import traceback
            traceback.print_exc()

    # 汇总
    summary_path = cfg.paths.data_root / "verification_summary.json"
    summary = [
        {"persona_id": r["persona_id"],
         "overall_pass_rate": r["overall_pass_rate"],
         "pass_rates": r["pass_rates"],
         "all_thresholds_met": r["all_thresholds_met"]}
        for r in all_reports
    ]
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n\nVerification summary: {summary_path}")


if __name__ == "__main__":
    main()
