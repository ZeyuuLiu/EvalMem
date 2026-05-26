#!/usr/bin/env python3
"""
Runtime smoke test：验证 user_simulator + session_runner 能跑。

不接真实记忆系统——用一个 dummy adapter（呼应"hello"），跑 p_01 session 1 的
20 轮对话，检查：
- 种子话术正确触发
- 用户模拟器生成连贯回复
- exposure_audit 全部 True
- 无公平性违规
"""
from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from builder import load_config, build_client
from builder.exposure_plan import load_exposure_plan
from builder.persona_pool import load_persona_pool
from builder.seed_utterance import load_seed_utterances
from builder.state_evolution import load_state_evolution
from runtime import MemorySystemAdapter, run_session_with_system


class DummyEchoAdapter:
    """最小被测系统：每次回复 echo + 助手风格短回复。"""
    name = "dummy_echo"

    def __init__(self):
        self.memory: list[tuple[str, str]] = []

    def reset(self) -> None:
        self.memory.clear()

    def respond(self, user_msg: str) -> str:
        # 简单 echo 助手：模拟一个"听到了，问个跟进问题"的助手
        reply = f"我听到你说的了。能再具体讲讲吗？"
        self.memory.append(("user", user_msg))
        self.memory.append(("assistant", reply))
        return reply

    def answer_in_readonly(self, question: str) -> str:
        return "（dummy: 无答案）"


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config()
    client = build_client(cfg, cache_subdir="runtime_smoke")

    # 用 p_01 + session 1 做最小测试
    pid = "p_01"
    personas = {p.persona_id: p for p in load_persona_pool(cfg)}
    persona = personas[pid]

    plan = load_exposure_plan(cfg, pid)
    seeds = load_seed_utterances(cfg, pid)
    evolution = load_state_evolution(cfg, pid)

    session_t = 1
    sess_plan = plan.sessions[session_t - 1]
    seed = seeds[session_t]
    state_t = evolution.trajectory[session_t].state

    adapter = DummyEchoAdapter()
    adapter.reset()

    print(f"\n=== Runtime smoke test: {pid} session {session_t} ===")
    print(f"Persona: {persona.name}, comm_style={persona.communication_style}")
    print(f"Seed: {seed.seed_text}")
    print(f"Must expose ({len(sess_plan.must_expose)} items):")
    for item in sess_plan.must_expose:
        print(f"  - {item.var} = {item.value} (hint: {item.tone_hint})")

    print(f"\n--- Running 20-turn dialogue ---")
    dialogue = run_session_with_system(
        persona=persona,
        session_t=session_t,
        plan=sess_plan,
        seed=seed,
        state_at_t=state_t,
        system=adapter,
        cfg=cfg,
        client=client,
    )

    print(f"\n=== Result ===")
    print(f"Turns: {len(dialogue.turns)}")
    print(f"Fairness violation: {dialogue.fairness_violation}")
    print(f"Exposure audit: {dialogue.exposure_audit}")
    print(f"\n--- First 4 turns ---")
    for t in dialogue.turns[:4]:
        force_marker = " [FORCE]" if t.is_force_expose else ""
        print(f"\n  Turn {t.turn_index}{force_marker} (exposed: {t.exposed_in_this_msg})")
        print(f"    USER: {t.user_message}")
        print(f"    ASST: {t.assistant_reply}")

    print(f"\n--- Last 2 turns ---")
    for t in dialogue.turns[-2:]:
        force_marker = " [FORCE]" if t.is_force_expose else ""
        print(f"\n  Turn {t.turn_index}{force_marker} (exposed: {t.exposed_in_this_msg})")
        print(f"    USER: {t.user_message}")
        print(f"    ASST: {t.assistant_reply}")

    # 落盘
    out_dir = cfg.paths.data_root / "runtime_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pid}_session_{session_t:02d}_{adapter.name}.json"
    out_path.write_text(
        json.dumps(dialogue.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nDialogue saved: {out_path}")


if __name__ == "__main__":
    main()
