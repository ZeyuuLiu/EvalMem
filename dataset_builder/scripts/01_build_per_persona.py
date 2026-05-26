#!/usr/bin/env python3
"""
单画像构建 orchestrator。

执行阶段 0-7（不含异源验证 + 人工 κ）：
1. 加载 persona
2. 采样问题草稿
3. 构造 state schema + 双核心
4. 状态演化（纯算法）
5. 暴露计划 + 种子话术
6. 评测问题（含 state-tracking + NEG 4 类）
7. Oracle dialogues + contexts

用法：
    python scripts/01_build_per_persona.py --persona-id p_01
    python scripts/01_build_per_persona.py --all
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

# 把项目根加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from builder import (
    DatasetConfig,
    LLMClient,
    Persona,
    build_client,
    load_config,
    load_state_var_library,
)
from builder.persona_pool import build_persona_pool, load_persona_pool
from builder.question_sampler import build_questions_for_persona
from builder.state_schema import build_state_schema
from builder.state_evolution import build_state_evolution
from builder.exposure_plan import build_exposure_plan
from builder.seed_utterance import build_seed_utterances
from builder.eval_questions import build_eval_questions
from builder.oracle_context import build_oracle_dialogues, build_oracle_contexts


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_one_persona(persona: Persona, cfg: DatasetConfig, client: LLMClient) -> dict:
    """对单个画像执行 stage 1-7。返回各阶段的统计。"""
    logger = logging.getLogger("orchestrator")
    stats = {"persona_id": persona.persona_id, "stages": {}}
    lib = load_state_var_library()

    # 确保 per_persona/<id>/persona.json 存在（即便 pool 已构建）
    from builder.config import persona_data_dir
    pdir = persona_data_dir(cfg, persona.persona_id)
    persona_path = pdir / "persona.json"
    if not persona_path.exists():
        import json
        persona_path.write_text(
            json.dumps(persona.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"[{persona.persona_id}] wrote persona.json")

    t0 = time.time()

    # === Stage 1：问题草稿 ===
    logger.info(f"[{persona.persona_id}] Stage 1: question sampling...")
    t = time.time()
    questions_draft = build_questions_for_persona(persona, lib, cfg, client)
    stats["stages"]["1_questions_draft"] = {
        "n_questions": len(questions_draft),
        "elapsed_sec": round(time.time() - t, 1),
    }

    # === Stage 2：state schema ===
    logger.info(f"[{persona.persona_id}] Stage 2: state schema...")
    t = time.time()
    schema = build_state_schema(persona, questions_draft, lib, cfg, client)
    stats["stages"]["2_state_schema"] = {
        "n_vars": len(schema.variables),
        "main_core": schema.main_core,
        "n_aux_cores": len(schema.auxiliary_cores),
        "elapsed_sec": round(time.time() - t, 1),
    }

    # === Stage 3：state evolution（纯算法）===
    logger.info(f"[{persona.persona_id}] Stage 3: state evolution...")
    t = time.time()
    evolution = build_state_evolution(persona, schema, cfg)
    n_changes = sum(1 for s in evolution.trajectory if s.changes_from_prev)
    stats["stages"]["3_state_evolution"] = {
        "n_change_sessions": n_changes,
        "interference_size_at_t10": len(evolution.interference_set.get(10, [])),
        "elapsed_sec": round(time.time() - t, 1),
    }

    # === Stage 4：exposure plan ===
    logger.info(f"[{persona.persona_id}] Stage 4: exposure plan...")
    t = time.time()
    plan = build_exposure_plan(persona, schema, evolution, cfg, client)
    n_total_must_expose = sum(len(s.must_expose) for s in plan.sessions)
    stats["stages"]["4_exposure_plan"] = {
        "n_sessions": len(plan.sessions),
        "total_must_expose_items": n_total_must_expose,
        "elapsed_sec": round(time.time() - t, 1),
    }

    # === Stage 5：seed utterances ===
    logger.info(f"[{persona.persona_id}] Stage 5: seed utterances...")
    t = time.time()
    seeds = build_seed_utterances(persona, schema, plan, cfg, client)
    stats["stages"]["5_seed_utterances"] = {
        "n_seeds": len(seeds),
        "elapsed_sec": round(time.time() - t, 1),
    }

    # === Stage 6：eval questions ===
    logger.info(f"[{persona.persona_id}] Stage 6: eval questions...")
    t = time.time()
    eval_qs = build_eval_questions(
        persona, schema, evolution, plan, questions_draft, cfg, client,
    )
    pos_n = sum(1 for q in eval_qs if q.task_type == "POS")
    neg_n = sum(1 for q in eval_qs if q.task_type == "NEG")
    tracking_n = sum(1 for q in eval_qs if q.is_state_tracking)
    total_asks = sum(len(q.ask_at_sessions) for q in eval_qs)
    stats["stages"]["6_eval_questions"] = {
        "n_total": len(eval_qs),
        "n_pos": pos_n,
        "n_neg": neg_n,
        "n_state_tracking": tracking_n,
        "total_asks_across_sessions": total_asks,
        "elapsed_sec": round(time.time() - t, 1),
    }

    # === Stage 7：oracle dialogues + contexts ===
    logger.info(f"[{persona.persona_id}] Stage 7: oracle dialogues + contexts...")
    t = time.time()
    oracle_dialogues = build_oracle_dialogues(persona, schema, evolution, plan, cfg, client)
    oracle_contexts = build_oracle_contexts(persona, eval_qs, oracle_dialogues, cfg)
    stats["stages"]["7_oracle"] = {
        "n_oracle_sessions": len(oracle_dialogues),
        "n_oracle_contexts": sum(len(v) for v in oracle_contexts.values()),
        "elapsed_sec": round(time.time() - t, 1),
    }

    stats["total_elapsed_sec"] = round(time.time() - t0, 1)
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="Build DynaMem-Bench per persona.")
    parser.add_argument(
        "--persona-id", type=str, default=None,
        help="构建单个画像（如 p_01）；若不指定则需配合 --all"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="构建全部 10 画像（先确保 personas pool 已构建）"
    )
    parser.add_argument(
        "--build-pool", action="store_true",
        help="先构建 persona pool（首次运行必须）"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("orchestrator")

    cfg = load_config()
    client = build_client(cfg)

    # 0. （可选）构建 persona pool
    if args.build_pool:
        logger.info("=== Stage 0: building persona pool ===")
        build_persona_pool(cfg, client)

    # 加载已有 pool
    try:
        personas = load_persona_pool(cfg)
        logger.info(f"loaded {len(personas)} personas from pool")
    except FileNotFoundError:
        logger.info("persona pool not found; building it now...")
        personas = build_persona_pool(cfg, client)

    # 选定要构建的画像
    if args.persona_id:
        target = [p for p in personas if p.persona_id == args.persona_id]
        if not target:
            logger.error(f"persona {args.persona_id} not in pool")
            sys.exit(1)
        targets = target
    elif args.all:
        targets = personas
    else:
        logger.error("must specify --persona-id or --all (and optionally --build-pool)")
        sys.exit(1)

    all_stats = []
    for persona in targets:
        try:
            logger.info(f"========== building {persona.persona_id} ({persona.name}) ==========")
            stats = build_one_persona(persona, cfg, client)
            all_stats.append(stats)
            logger.info(f"=== {persona.persona_id} DONE in {stats['total_elapsed_sec']}s ===")
        except Exception as e:
            logger.error(f"persona {persona.persona_id} build FAILED: {e}")
            traceback.print_exc()
            all_stats.append({
                "persona_id": persona.persona_id,
                "error": str(e),
            })

    # 落盘汇总
    summary_path = cfg.paths.data_root / "build_stats.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    summary_path.write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"build stats: {summary_path}")


if __name__ == "__main__":
    main()
