"""
阶段 3：状态演化轨迹生成（纯算法，无 LLM 调用）。

输入：Persona + StateSchema
输出：StateEvolution（含 σ_0..σ_10 + I_t）

算法（详见构建方案 §七）：
- session 0：σ_0 初始化（避免极端值）
- session 1-5：保持 σ_0
- session 6：触发主核心变更
- session 7-10：触发辅核心级联（按 cascade_lag）+ 30% 画像回滚 + ρ-单点更新

完全确定性：相同 (persona_id, seed) 产生相同轨迹。
"""
from __future__ import annotations

import json
import logging
import random
from typing import Optional

from .config import DatasetConfig, persona_data_dir
from .schemas import (
    InterferenceItem,
    Persona,
    StateAssignment,
    StateEvolution,
    StateSchema,
    StateSnapshot,
)

logger = logging.getLogger(__name__)


# =====================================================================
# 子算法
# =====================================================================

def sample_initial_state(
    schema: StateSchema,
    persona: Persona,
    rng: random.Random,
) -> StateAssignment:
    """σ_0 采样：每个变量从候选值中选一个（避开首尾极端，留演化空间）。"""
    state: StateAssignment = {}
    for var in schema.variables:
        # 与画像有约束的变量
        if var.var_name == "education":
            state[var.var_name] = persona.education
        elif var.var_name == "age" or var.var_name == "age_group":
            state[var.var_name] = str(persona.age)
        else:
            mid_values = var.values[1:-1] if len(var.values) > 2 else var.values
            state[var.var_name] = rng.choice(mid_values)
    return state


def pick_new_value(
    var_values: list[str],
    current_value: str,
    rng: random.Random,
) -> str:
    """从候选中挑一个不等于 current_value 的新值。"""
    alternatives = [v for v in var_values if v != current_value]
    if not alternatives:
        return current_value  # 没法变（只有 1 个候选值）
    return rng.choice(alternatives)


def generate_event_for_change(
    var_name: str, var_display: str, old: str, new: str, persona: Persona
) -> str:
    """对状态变更生成简短叙事事件（模板化）。"""
    templates = {
        "city": f"{persona.name}因家庭/工作原因从 {old} 搬到 {new}",
        "job_title": f"{persona.name}的职位从 {old} 转换为 {new}",
        "current_employment_status": f"{persona.name}的就业状态从 {old} 变为 {new}",
        "current_focus": f"{persona.name}的目标方向从 {old} 转为 {new}",
        "marital_status": f"{persona.name}的婚姻状态从 {old} 变为 {new}",
        "internship_status": f"{persona.name}的实习状态从 {old} 变为 {new}",
        "exercise_frequency": f"{persona.name}的锻炼频率从 {old} 调整为 {new}",
        "diet_restriction": f"{persona.name}的饮食限制从 {old} 改为 {new}",
        "savings_goal": f"{persona.name}的储蓄目标从 {old} 转向 {new}",
    }
    return templates.get(
        var_name, f"{persona.name}的 {var_display} 发生变化：{old} → {new}"
    )


def narrate_session_event(
    session: int, changes: list[str], persona: Persona, default_narrative: str
) -> str:
    """无变更 session 的叙事兜底。"""
    if changes:
        return "; ".join(changes[:3])
    return default_narrative


# =====================================================================
# 主算法
# =====================================================================

def build_state_evolution(
    persona: Persona,
    schema: StateSchema,
    cfg: DatasetConfig,
    write_file: bool = True,
) -> StateEvolution:
    """
    严格按规则推进 σ_0 → σ_10。
    无 LLM 调用，可重现性 100%（受 cfg.seed 与 persona.persona_id 决定）。
    """
    rng = random.Random(hash((persona.persona_id, cfg.seed)) & 0x7FFFFFFF)

    main_core = schema.main_core
    main_var = schema.get(main_core)
    aux_cores = schema.auxiliary_cores

    se_cfg = cfg.state_evolution
    main_change_session = se_cfg.main_core_change_session  # 6
    rollback_session = se_cfg.rollback_session  # 9
    is_rollback_persona = persona.persona_id in se_cfg.rollback_personas

    trajectory: list[StateSnapshot] = []
    interference: dict[int, list[InterferenceItem]] = {}

    # === session 0 初始化 ===
    sigma_0 = sample_initial_state(schema, persona, rng)
    trajectory.append(StateSnapshot(
        session=0, state=dict(sigma_0),
        event="初始状态", changes_from_prev=[],
    ))
    interference[0] = []
    prev_state = dict(sigma_0)

    # === session 1 至 (main_change_session-1)：保持 σ_0 ===
    for t in range(1, main_change_session):
        new_state = dict(prev_state)
        trajectory.append(StateSnapshot(
            session=t, state=new_state,
            event=f"session_{t}: 日常对话，补充人物细节",
            changes_from_prev=[],
        ))
        interference[t] = list(interference[t - 1])
        prev_state = new_state

    # === session main_change_session：触发主核心变更 ===
    t = main_change_session  # = 6
    old_main_value = prev_state[main_core]
    new_main_value = pick_new_value(main_var.values, old_main_value, rng)
    new_state = dict(prev_state)
    new_state[main_core] = new_main_value
    change_desc = f"{main_core}: {old_main_value} → {new_main_value}"

    trajectory.append(StateSnapshot(
        session=t, state=new_state,
        event=generate_event_for_change(
            main_core, main_var.display_name, old_main_value, new_main_value, persona
        ),
        changes_from_prev=[change_desc],
        is_main_core_change=True,
    ))
    new_interf = list(interference[t - 1])
    new_interf.append(InterferenceItem(
        var=main_core, old_value=old_main_value,
        expired_at_session=t - 1, label="stale",
    ))
    interference[t] = new_interf
    prev_state = new_state

    # === 计算辅核心级联触发 session ===
    aux_trigger_at: dict[int, list] = {s: [] for s in range(main_change_session, 11)}
    for cascade in aux_cores:
        if cascade.cascade_lag == "immediate":
            target_session = main_change_session + 1  # session 7
        else:  # delayed
            lo = main_change_session + cascade.delayed_session_min  # 8
            hi = main_change_session + cascade.delayed_session_max  # 10
            target_session = rng.randint(min(lo, 10), min(hi, 10))
        aux_trigger_at[target_session].append(cascade)

    # === session (main_change_session+1) 至 10：级联 + 回滚 + ρ-update ===
    non_core_vars = [
        v for v in schema.variables
        if v.var_name != main_core
        and v.var_name not in [c.aux_var for c in aux_cores]
    ]

    for t in range(main_change_session + 1, 11):  # 7..10
        new_state = dict(prev_state)
        changes: list[str] = []
        new_interf = list(interference[t - 1])

        # (a) 应用本 session 的级联变更
        for cascade in aux_trigger_at.get(t, []):
            aux_var = schema.get(cascade.aux_var)
            old_aux_value = prev_state[cascade.aux_var]
            new_aux_value = pick_new_value(aux_var.values, old_aux_value, rng)
            new_state[cascade.aux_var] = new_aux_value
            changes.append(
                f"{cascade.aux_var}: {old_aux_value} → {new_aux_value} "
                f"(cascaded from {main_core})"
            )
            new_interf.append(InterferenceItem(
                var=cascade.aux_var, old_value=old_aux_value,
                expired_at_session=t - 1, label="cascaded",
                cause_session=main_change_session,
            ))

        # (b) 回滚（仅对配置中的画像，t == rollback_session）
        is_rollback = False
        if is_rollback_persona and t == rollback_session:
            current_main = new_state[main_core]
            new_state[main_core] = old_main_value  # 复原到 σ_5 的旧值
            changes.append(
                f"{main_core}: {current_main} → {old_main_value} (ROLLBACK)"
            )
            new_interf.append(InterferenceItem(
                var=main_core, old_value=current_main,
                expired_at_session=t - 1, label="rollback_reactivated",
            ))
            is_rollback = True

        # (c) ρ-单点更新（非核心变量）
        for var in non_core_vars:
            if rng.random() < persona.rho:
                old_v = prev_state[var.var_name]
                new_v = pick_new_value(var.values, old_v, rng)
                if new_v != old_v:
                    new_state[var.var_name] = new_v
                    changes.append(f"{var.var_name}: {old_v} → {new_v} (ρ-update)")
                    new_interf.append(InterferenceItem(
                        var=var.var_name, old_value=old_v,
                        expired_at_session=t - 1, label="stale",
                    ))

        snapshot = StateSnapshot(
            session=t, state=new_state,
            event=narrate_session_event(
                t, changes, persona,
                default_narrative=f"session_{t}: 日常对话",
            ),
            changes_from_prev=changes,
            is_aux_cascade=any("cascaded" in c for c in changes),
            is_rollback=is_rollback,
        )
        trajectory.append(snapshot)
        interference[t] = new_interf
        prev_state = new_state

    evolution = StateEvolution(
        persona_id=persona.persona_id,
        trajectory=trajectory,
        interference_set=interference,
    )

    # 摘要日志
    logger.info(f"persona {persona.persona_id} state evolution:")
    for snap in trajectory:
        if snap.changes_from_prev:
            logger.info(f"  session {snap.session}: {snap.changes_from_prev}")

    if write_file:
        pdir = persona_data_dir(cfg, persona.persona_id)
        path = pdir / "state_evolution.json"
        path.write_text(
            json.dumps(evolution.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"wrote {path}")

    return evolution


def load_state_evolution(cfg: DatasetConfig, persona_id: str) -> StateEvolution:
    pdir = persona_data_dir(cfg, persona_id)
    path = pdir / "state_evolution.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return StateEvolution.from_dict(raw)


def first_appearance_session(var: str, evolution: StateEvolution) -> int:
    """返回某变量首次在 σ_t 中出现的 session（即 σ_0 中即有 → 0）。"""
    for snapshot in evolution.trajectory:
        if var in snapshot.state:
            return snapshot.session
    return -1


def value_change_sessions(var: str, evolution: StateEvolution) -> list[int]:
    """返回某变量发生取值变化的 session 列表（含 σ_0 视为 session 0 的"出现"）。"""
    out = []
    prev_value = None
    for snapshot in evolution.trajectory:
        cur = snapshot.state.get(var)
        if cur != prev_value:
            out.append(snapshot.session)
            prev_value = cur
    return out


__all__ = [
    "sample_initial_state",
    "build_state_evolution",
    "load_state_evolution",
    "first_appearance_session",
    "value_change_sessions",
]
