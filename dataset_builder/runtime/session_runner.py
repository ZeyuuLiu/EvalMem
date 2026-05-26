"""
Runtime 阶段 10：单 session 在线 on-policy 交互执行器。

输入：
- 离线生成的 (persona, schema, evolution, plan, seeds, eval_questions, oracle_contexts)
- 被测系统 adapter（实现 respond / answer / reset 接口）

输出：
- SessionDialogue：本 session 的 20 轮对话 + exposure_audit + 公平性违规标记
- per-question evaluation results

每个被测系统在每个画像上独立运行 1 次完整的 10-session 评测。
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Protocol

from builder.config import DatasetConfig
from builder.llm_client import LLMClient
from builder.schemas import (
    DialogueTurn,
    ExposureItem,
    ExposurePlan,
    Persona,
    SeedUtterance,
    SessionDialogue,
    SessionExposurePlan,
    StateEvolution,
)
from .user_simulator import (
    UserSimResponse,
    call_force_expose,
    call_user_simulator,
    mark_exposed,
)

logger = logging.getLogger(__name__)


# =====================================================================
# 被测系统适配器 Protocol（runtime 期望接口）
# =====================================================================

class MemorySystemAdapter(Protocol):
    """被测系统必须实现的最小接口。"""

    name: str

    def reset(self) -> None:
        """重置系统记忆库（每画像开始时调用）。"""
        ...

    def respond(self, user_msg: str) -> str:
        """正常对话路径，会更新系统记忆。"""
        ...

    def answer_in_readonly(self, question: str) -> str:
        """评测探针路径，必须保证不污染记忆。"""
        ...


# =====================================================================
# 单 session runner
# =====================================================================

def run_session_with_system(
    persona: Persona,
    session_t: int,
    plan: SessionExposurePlan,
    seed: SeedUtterance,
    state_at_t: dict,
    system: MemorySystemAdapter,
    cfg: DatasetConfig,
    client: LLMClient,
) -> SessionDialogue:
    """
    执行单 session 的 20 轮 on-policy 交互。
    必须保证 plan.must_expose 全部暴露完毕。
    """
    exposed_remaining = list(plan.must_expose)
    exposure_audit = {item.var: False for item in plan.must_expose}
    dialogue: list[DialogueTurn] = []

    n_turns = cfg.scale.num_turns_per_session

    for turn_k in range(n_turns):
        is_force = False

        # ===== 用户消息生成 =====
        if turn_k == 0:
            # turn 0：固定种子话术（所有系统这一轮一致）
            user_msg = seed.seed_text
            self_reported = list(seed.expects_to_expose)
        elif turn_k == n_turns - 1 and len(exposed_remaining) > 0:
            # 最后一轮强制兜底
            response = call_force_expose(persona, exposed_remaining, cfg, client)
            user_msg = response.text
            self_reported = response.exposed_in_this_msg
            is_force = True
        else:
            response = call_user_simulator(
                persona=persona,
                current_state=state_at_t,
                exposed_remaining=exposed_remaining,
                dialogue_history=dialogue,
                last_assistant_reply=dialogue[-1].assistant_reply if dialogue else "",
                sentiment=plan.expose_sentiment,
                session_theme=plan.session_theme,
                cfg=cfg,
                client=client,
            )
            user_msg = response.text
            self_reported = response.exposed_in_this_msg

        # ===== 校验：哪些信息真的被暴露 =====
        confirmed = mark_exposed(user_msg, self_reported, exposed_remaining, cfg, client)
        for var in confirmed:
            exposure_audit[var] = True

        # ===== 调被测系统响应 =====
        try:
            assistant_reply = system.respond(user_msg)
        except Exception as e:
            logger.error(
                f"system {system.name} respond failed at "
                f"persona={persona.persona_id} session={session_t} turn={turn_k}: {e}"
            )
            assistant_reply = f"[ERROR: {e}]"

        dialogue.append(DialogueTurn(
            turn_index=turn_k,
            user_message=user_msg,
            assistant_reply=assistant_reply,
            exposed_in_this_msg=confirmed,
            is_force_expose=is_force,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))

    # ===== session 末公平性校验 =====
    fairness_violation = any(not v for v in exposure_audit.values())
    if fairness_violation:
        unexposed = [k for k, v in exposure_audit.items() if not v]
        logger.warning(
            f"FAIRNESS VIOLATION: persona={persona.persona_id} session={session_t} "
            f"system={system.name} unexposed={unexposed}"
        )

    return SessionDialogue(
        persona_id=persona.persona_id,
        session=session_t,
        system_name=system.name,
        turns=dialogue,
        exposure_audit=exposure_audit,
        fairness_violation=fairness_violation,
    )


# =====================================================================
# 完整 persona run
# =====================================================================

def run_persona_with_system(
    persona: Persona,
    plan: ExposurePlan,
    seeds: dict[int, SeedUtterance],
    evolution: StateEvolution,
    system: MemorySystemAdapter,
    cfg: DatasetConfig,
    client: LLMClient,
) -> list[SessionDialogue]:
    """
    对一个 (persona, system) 组合跑完整 10 session。
    系统在开始前 reset，跨 session 共享 system.memory。
    """
    system.reset()
    out: list[SessionDialogue] = []

    for sess_plan in plan.sessions:
        t = sess_plan.session
        state_t = evolution.trajectory[t].state
        seed = seeds.get(t)
        if seed is None:
            logger.error(f"missing seed for session {t}")
            continue

        t_start = time.time()
        dialogue = run_session_with_system(
            persona=persona,
            session_t=t,
            plan=sess_plan,
            seed=seed,
            state_at_t=state_t,
            system=system,
            cfg=cfg,
            client=client,
        )
        elapsed = time.time() - t_start
        logger.info(
            f"persona {persona.persona_id} session {t} system {system.name}: "
            f"{cfg.scale.num_turns_per_session} turns in {elapsed:.1f}s, "
            f"audit={dialogue.exposure_audit}"
        )
        out.append(dialogue)

    return out


__all__ = [
    "MemorySystemAdapter",
    "run_session_with_system",
    "run_persona_with_system",
]
