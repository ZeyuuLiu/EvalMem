"""
阶段 6：评测问题生成（含 state-tracking 算法 + NEG 4 子类）。

输入：
- questions_draft.json（阶段 1 草稿）
- state_schema.json
- state_evolution.json
- exposure_plan.json

输出：eval_questions.json（每画像约 80 道完整 EvalQuestion，分布到 10 session，
        每 session 末平均 ~10 询问 = 总约 100 询问）

题目分布（每画像 80 题）：
- POS-fresh ~40 题：在状态首次暴露的 session 询问 1 次
- POS-tracking ~20 题：在最多 K 次状态变化 session 末询问（is_state_tracking=True）
- NEG ~20 题：4 子类细分，分散到不同 session（不集中在 session 10）

总询问次数 ≈ 40 + 20×2 + 20 = 100，期望平均 10/session。
"""
from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from typing import Optional

from .config import DatasetConfig, persona_data_dir
from .llm_client import LLMClient
from .schemas import (
    EvalQuestion,
    ExposurePlan,
    Persona,
    StateEvolution,
    StateSchema,
)

logger = logging.getLogger(__name__)


# =====================================================================
# 工具：first_appearance / value_change
# =====================================================================

def _first_appearance_session(var: str, evolution: StateEvolution) -> int:
    for snapshot in evolution.trajectory:
        if var in snapshot.state and snapshot.state[var] != "":
            return snapshot.session
    return -1


def _all_change_sessions(var: str, evolution: StateEvolution) -> list[int]:
    """变量取值发生变化的 session 列表（不含 session 0 的"出现"）。"""
    out = []
    prev = None
    for snapshot in evolution.trajectory:
        cur = snapshot.state.get(var)
        if prev is not None and cur != prev:
            out.append(snapshot.session)
        prev = cur
    return out


def _exposed_at_session(var: str, plan: ExposurePlan) -> Optional[int]:
    """变量在哪个 session 首次被 must_expose？"""
    for sess_plan in plan.sessions:
        for item in sess_plan.must_expose:
            if item.var == var:
                return sess_plan.session
    return None


# =====================================================================
# Gold answer 计算
# =====================================================================

def _compute_gold(
    required_vars: list[str],
    state: dict,
    schema: StateSchema,
) -> str:
    if len(required_vars) == 1:
        return str(state.get(required_vars[0], ""))
    parts = []
    for var in required_vars:
        try:
            display = schema.get(var).display_name
        except KeyError:
            display = var
        parts.append(f"{display}={state.get(var, '')}")
    return "；".join(parts)


# =====================================================================
# ask_at_sessions 算法（POS）
# =====================================================================

def compute_ask_at_sessions_for_pos(
    question: dict,
    evolution: StateEvolution,
    plan: ExposurePlan,
    schema: StateSchema,
    is_state_tracking: bool,
    num_sessions: int,
    max_asks: int,
) -> tuple[list[int], dict[int, str], dict[int, list]]:
    """
    POS 题的询问 session + gold 计算。

    - POS-fresh（is_state_tracking=False）：仅在最早可问的 session 问 1 次
    - POS-tracking（is_state_tracking=True）：在每次状态变化的 session 末问，最多 max_asks 次
    """
    required_vars = question["required_vars"]

    earliest_askable = 0
    for var in required_vars:
        exp_at = _exposed_at_session(var, plan)
        if exp_at is None:
            exp_at = _first_appearance_session(var, evolution)
        if exp_at is None or exp_at == -1:
            return [], {}, {}
        earliest_askable = max(earliest_askable, exp_at)

    ask_set = {earliest_askable}

    if is_state_tracking:
        for var in required_vars:
            for s in _all_change_sessions(var, evolution):
                if s >= earliest_askable:
                    ask_set.add(s)

    ask_at = sorted(ask_set)
    # cap：保留 earliest 与最多 (max_asks - 1) 个变化 session
    if len(ask_at) > max_asks:
        # 保留 earliest + 均匀采样剩余
        kept = [ask_at[0]] + ask_at[1::max(1, (len(ask_at) - 1) // (max_asks - 1))][:max_asks - 1]
        ask_at = sorted(set(kept))

    gold_answers: dict[int, str] = {}
    f_keys: dict[int, list] = {}
    for s in ask_at:
        snap = evolution.trajectory[s]
        gold_answers[s] = _compute_gold(required_vars, snap.state, schema)
        f_keys[s] = [[var, str(snap.state.get(var, ""))] for var in required_vars]

    return ask_at, gold_answers, f_keys


# =====================================================================
# NEG 题生成 prompts
# =====================================================================

NEG_A_PROMPT = """画像背景：{backstory}
该画像的 state schema 涉及以下变量：{schema_vars}
（这些变量会在评测剧本中被暴露）

【任务】
生成 {n} 道关于该画像但**不依赖任何上述变量**的中文问题。即：这些问题应当问的是剧本中**从未会出现**的具体信息（如妹妹生日、宠物名字、童年朋友联系方式等）。

【示例】
- 已涉及变量：city, job_title
- 合法 NEG-A：你妹妹的生日是哪一天？你的宠物叫什么名字？
- 不合法：你现在住在哪？（依赖 city，剧本会暴露）

【输出格式（严格 JSON）】
{{"neg_a_questions": ["...", "..."]}}
"""


NEG_D_PROMPT = """画像背景：{backstory}
schema 变量：{schema_vars}

【任务】
生成 {n} 道与某个 schema 变量**表层相似但语义不同**的中文问题。即：问题表述与剧本中可能出现的 record 看起来很像，但实际指向另一个对象（与该画像无关）。

【示例】
- 已涉及：job_title 是用户自己的工作
- 合法 NEG-D：你哥哥的工作是什么？（剧本中讲了"用户自己的工作"，但用户哥哥的工作没讲）
- 不合法：你的工作是什么？（这是 POS 题）

【输出格式（严格 JSON）】
{{"neg_d_questions": ["...", "..."]}}
"""


POS_TOPUP_PROMPT = """你是为虚拟用户生成评测问题的助手。

【画像】
{backstory}
姓名：{name}
沟通风格：{communication_style}

【可用状态变量列表（你只能使用这些 var，不能用其他）】
{schema_vars_text}

【现已生成的问题列表（避免重复）】
{existing_questions_text}

【任务】
再补充生成 {n} 道**仅依赖上述 schema 变量**的中文 POS 问题。要求与现有问题语义不重复。

【输出格式（严格 JSON）】
{{
  "questions": [
    {{"question": "...", "required_vars": ["..."], "domain": "..."}},
    ...
  ]
}}
"""


def _generate_pos_topup(
    persona: Persona,
    schema: StateSchema,
    existing: list[dict],
    n: int,
    cfg: DatasetConfig,
    client: LLMClient,
) -> list[dict]:
    """当 POS 草稿不足时，调 LLM 补足专门匹配 schema 的题目。"""
    if n <= 0:
        return []
    schema_vars_text = "\n".join([
        f"- var_name: {v.var_name} ({v.display_name}) 取值: {v.values} 领域: {v.domain}"
        for v in schema.variables
    ])
    existing_text = "\n".join([f"- {q['question']}" for q in existing[:30]])
    prompt = POS_TOPUP_PROMPT.format(
        backstory=persona.backstory,
        name=persona.name,
        communication_style=persona.communication_style,
        schema_vars_text=schema_vars_text,
        existing_questions_text=existing_text,
        n=n,
    )
    try:
        response = client.chat_json(
            prompt, model=cfg.llm.builder_model,
            temperature=cfg.llm.builder_temperature,
            max_tokens=2000,
        )
        raw = response.get("questions", [])
        valid = []
        valid_var_names = {v.var_name for v in schema.variables}
        valid_domains = {v.domain for v in schema.variables}
        for q in raw[:n]:
            if not isinstance(q, dict):
                continue
            text = str(q.get("question", "")).strip()
            req = q.get("required_vars", [])
            if not text or not isinstance(req, list) or not (1 <= len(req) <= 2):
                continue
            if any(v not in valid_var_names for v in req):
                continue
            domain = q.get("domain", "")
            if domain not in valid_domains:
                # 兜底：取 required_vars[0] 所在的领域
                first_var = schema.get(req[0])
                domain = first_var.domain
            valid.append({
                "question": text,
                "required_vars": list(req),
                "domain": domain,
            })
        return valid
    except Exception as e:
        logger.warning(f"POS topup failed: {e}")
        return []


def _generate_neg_a(
    persona: Persona,
    schema: StateSchema,
    n: int,
    cfg: DatasetConfig,
    client: LLMClient,
) -> list[dict]:
    if n <= 0:
        return []
    schema_vars = "、".join([v.display_name for v in schema.variables])
    prompt = NEG_A_PROMPT.format(
        backstory=persona.backstory, schema_vars=schema_vars, n=n,
    )
    try:
        response = client.chat_json(
            prompt, model=cfg.llm.builder_model,
            temperature=cfg.llm.builder_temperature, max_tokens=600,
        )
        questions = response.get("neg_a_questions", [])[:n]
        return [{"question": str(q).strip(), "neg_subtype": "NEG-A", "domain": "人际"}
                for q in questions if str(q).strip()]
    except Exception as e:
        logger.warning(f"NEG-A gen failed: {e}; using fallbacks")
        fallbacks = [
            "你的宠物叫什么名字？", "你妹妹今年几岁？", "你大学室友姓什么？",
            "你小时候最喜欢的玩具是什么？", "你父亲生日是几月？",
            "你高中班主任叫什么？", "你最早一份工作的工资多少？", "你第一次出国去的哪里？",
            "你最喜欢的颜色是什么？", "你的初恋是谁？",
        ]
        return [{"question": q, "neg_subtype": "NEG-A", "domain": "人际"}
                for q in fallbacks[:n]]


def _generate_neg_b(
    persona: Persona,
    schema: StateSchema,
    evolution: StateEvolution,
    plan: ExposurePlan,
    n: int,
    num_sessions: int,
) -> list[dict]:
    """NEG-B 过期型：问 var 的当前值，trap 是 I_t 中旧值。"""
    out = []
    changed_vars = []
    seen = set()
    for items in evolution.interference_set.values():
        for item in items:
            if item.var not in seen:
                seen.add(item.var)
                changed_vars.append(item.var)

    for var in changed_vars[:n]:
        try:
            display = schema.get(var).display_name
        except KeyError:
            continue
        out.append({
            "question": f"我现在的{display}是什么？",
            "required_vars": [var],
            "neg_subtype": "NEG-B",
            "domain": _find_domain_for_var(var, schema),
        })
    return out[:n]


def _find_domain_for_var(var: str, schema: StateSchema) -> str:
    try:
        return schema.get(var).domain
    except KeyError:
        return "工作"


def _generate_neg_c(
    persona: Persona,
    schema: StateSchema,
    n: int,
) -> list[dict]:
    if n <= 0:
        return []
    main_var_meta = schema.get(schema.main_core)
    return [{
        "question": f"刚刚我们聊到的{main_var_meta.display_name}，你帮我确认下现在准确的取值是什么？",
        "required_vars": [schema.main_core],
        "neg_subtype": "NEG-C",
        "domain": main_var_meta.domain,
    }][:n]


def _generate_neg_d(
    persona: Persona,
    schema: StateSchema,
    n: int,
    cfg: DatasetConfig,
    client: LLMClient,
) -> list[dict]:
    if n <= 0:
        return []
    schema_vars = "、".join([v.display_name for v in schema.variables])
    prompt = NEG_D_PROMPT.format(
        backstory=persona.backstory, schema_vars=schema_vars, n=n,
    )
    try:
        response = client.chat_json(
            prompt, model=cfg.llm.builder_model,
            temperature=cfg.llm.builder_temperature, max_tokens=400,
        )
        questions = response.get("neg_d_questions", [])[:n]
        return [{"question": str(q).strip(), "neg_subtype": "NEG-D", "domain": "人际"}
                for q in questions if str(q).strip()]
    except Exception as e:
        logger.warning(f"NEG-D gen failed: {e}; using fallbacks")
        fallbacks = [
            "你哥哥现在住哪里？", "你姐姐的工作是什么？", "你父母的健康状况如何？",
            "你表妹在做什么职业？", "你叔叔最近搬家了吗？",
        ]
        return [{"question": q, "neg_subtype": "NEG-D", "domain": "人际"}
                for q in fallbacks[:n]]


def _allocate_neg_subtypes(persona: Persona, total_neg: int) -> dict[str, int]:
    if persona.neg_sensitivity == "高隐私敏感":
        ratios = {"NEG-A": 0.55, "NEG-B": 0.25, "NEG-C": 0.05, "NEG-D": 0.15}
    elif persona.neg_sensitivity == "经常修正过往":
        ratios = {"NEG-A": 0.30, "NEG-B": 0.40, "NEG-C": 0.15, "NEG-D": 0.15}
    else:
        ratios = {"NEG-A": 0.30, "NEG-B": 0.45, "NEG-C": 0.05, "NEG-D": 0.20}
    out = {k: max(0, int(round(total_neg * v))) for k, v in ratios.items()}
    diff = total_neg - sum(out.values())
    out["NEG-A"] += diff
    return out


# =====================================================================
# 把 NEG 题分散到不同 session
# =====================================================================

def _distribute_neg_to_sessions(
    neg_subtype: str,
    n_questions: int,
    num_sessions: int,
    rng: random.Random,
    earliest_session_for_b_c: int = 6,
) -> list[int]:
    """
    给 n 道某子类 NEG 题分配各自的询问 session。
    - NEG-A / NEG-D：对话从未提及 → 任意 session 都可询问 → uniform 分布
    - NEG-B 过期型 / NEG-C 冲突型：必须在状态发生变化之后 → session ≥ earliest（默认 6）
    """
    if n_questions <= 0:
        return []
    if neg_subtype in {"NEG-A", "NEG-D"}:
        # 在 session 1..num_sessions 中均匀分布
        sessions_pool = list(range(1, num_sessions + 1))
    else:  # NEG-B / NEG-C
        sessions_pool = list(range(earliest_session_for_b_c, num_sessions + 1))
    out = []
    for i in range(n_questions):
        out.append(sessions_pool[i % len(sessions_pool)])
    rng.shuffle(out)
    return out


# =====================================================================
# 主入口
# =====================================================================

def build_eval_questions(
    persona: Persona,
    schema: StateSchema,
    evolution: StateEvolution,
    plan: ExposurePlan,
    questions_draft: list[dict],
    cfg: DatasetConfig,
    client: LLMClient,
    write_file: bool = True,
) -> list[EvalQuestion]:
    """
    构造完整 ~80 题（60 POS + 20 NEG）。
    """
    rng = random.Random(hash((persona.persona_id, "eval_questions", cfg.seed)) & 0x7FFFFFFF)
    num_sessions = cfg.scale.num_sessions
    qd = cfg.question_distribution
    main_change_session = cfg.state_evolution.main_core_change_session

    # === Step A: 筛 + 补足 valid POS drafts ===
    schema_var_names = {v.var_name for v in schema.variables}

    def _is_valid(q: dict) -> bool:
        req = list(q.get("required_vars", []))
        if not (1 <= len(req) <= 2):
            return False
        return all(v in schema_var_names for v in req)

    valid_drafts = [q for q in questions_draft if _is_valid(q)]
    logger.info(f"persona {persona.persona_id}: {len(valid_drafts)} valid POS drafts (initial)")

    # 补足：若 valid POS 不到 pos_per_persona，调 LLM topup
    needed = qd.pos_per_persona - len(valid_drafts)
    if needed > 0:
        logger.info(f"persona {persona.persona_id}: POS topup needed: {needed}")
        topup = _generate_pos_topup(persona, schema, valid_drafts, needed, cfg, client)
        valid_drafts.extend(topup)
        logger.info(f"persona {persona.persona_id}: after topup: {len(valid_drafts)} drafts")

    # === Step B: 分类 tracking vs fresh ===
    pos_tracking_pool = []
    pos_fresh_pool = []
    for q in valid_drafts:
        required = q["required_vars"]
        has_change = any(_all_change_sessions(v, evolution) for v in required)
        if has_change:
            pos_tracking_pool.append(q)
        else:
            pos_fresh_pool.append(q)

    rng.shuffle(pos_tracking_pool)
    rng.shuffle(pos_fresh_pool)

    pos_tracking_picked = pos_tracking_pool[:qd.pos_tracking_target]
    pos_fresh_picked = pos_fresh_pool[:qd.pos_fresh_target]

    # 互相补足
    if len(pos_fresh_picked) < qd.pos_fresh_target:
        deficit = qd.pos_fresh_target - len(pos_fresh_picked)
        extra = pos_tracking_pool[qd.pos_tracking_target:qd.pos_tracking_target + deficit]
        pos_fresh_picked.extend(extra)
    if len(pos_tracking_picked) < qd.pos_tracking_target:
        deficit = qd.pos_tracking_target - len(pos_tracking_picked)
        extra = pos_fresh_pool[qd.pos_fresh_target:qd.pos_fresh_target + deficit]
        pos_tracking_picked.extend(extra)

    # === Step C: 转换为 EvalQuestion ===
    eval_questions: list[EvalQuestion] = []
    qid_counter = 1

    def _next_qid() -> str:
        nonlocal qid_counter
        qid = f"{persona.persona_id}_q_{qid_counter:03d}"
        qid_counter += 1
        return qid

    # POS-fresh
    # 收集所有 POS-fresh 的 earliest_askable，按"在 [earliest, num_sessions] 区间循环分布"打散
    fresh_earliest: list[tuple[dict, int]] = []
    for q in pos_fresh_picked:
        required = q["required_vars"]
        earliest = 0
        ok = True
        for var in required:
            exp_at = _exposed_at_session(var, plan)
            if exp_at is None:
                exp_at = _first_appearance_session(var, evolution)
            if exp_at is None or exp_at == -1:
                ok = False
                break
            earliest = max(earliest, exp_at)
        if ok:
            fresh_earliest.append((q, earliest))

    # 按 earliest 升序排，然后用 round-robin 把 ask_at 分到 [earliest, num_sessions]
    fresh_earliest.sort(key=lambda x: x[1])
    # 为每个 session 维护一个计数器，挑负载最低的 session 分配
    session_load = {s: 0 for s in range(1, num_sessions + 1)}
    for q, earliest in fresh_earliest:
        candidates = [s for s in range(earliest, num_sessions + 1)]
        # 在合法区间内挑当前负载最低的 session
        ask_s = min(candidates, key=lambda s: session_load[s])
        session_load[ask_s] += 1
        snap = evolution.trajectory[ask_s]
        gold_a = _compute_gold(q["required_vars"], snap.state, schema)
        fkeys = [[v, str(snap.state.get(v, ""))] for v in q["required_vars"]]
        eval_questions.append(EvalQuestion(
            question_id=_next_qid(),
            persona_id=persona.persona_id,
            question=q["question"],
            task_type="POS",
            domain=q["domain"],
            required_vars=list(q["required_vars"]),
            ask_at_sessions=[ask_s],
            gold_answers={ask_s: gold_a},
            f_keys={ask_s: fkeys},
            is_state_tracking=False,
        ))

    # POS-tracking
    for q in pos_tracking_picked:
        ask_at, gold, fkeys = compute_ask_at_sessions_for_pos(
            q, evolution, plan, schema,
            is_state_tracking=True, num_sessions=num_sessions,
            max_asks=qd.pos_tracking_max_asks,
        )
        if not ask_at:
            continue
        # 若实际只问 1 次，降级为 fresh
        is_tracking = len(ask_at) >= 2
        cascade_meta = None
        if is_tracking:
            for cascade in schema.auxiliary_cores:
                if cascade.aux_var in q["required_vars"]:
                    cascade_meta = {
                        "is_cascade_var": True,
                        "main_core": cascade.main_var,
                        "main_core_change_session": main_change_session,
                    }
                    break
        eval_questions.append(EvalQuestion(
            question_id=_next_qid(),
            persona_id=persona.persona_id,
            question=q["question"],
            task_type="POS",
            domain=q["domain"],
            required_vars=list(q["required_vars"]),
            ask_at_sessions=ask_at,
            gold_answers=gold,
            f_keys=fkeys,
            is_state_tracking=is_tracking,
            cascade_metadata=cascade_meta,
        ))

    # === Step D: NEG 题（分散到多 session）===
    neg_alloc = _allocate_neg_subtypes(persona, qd.neg_per_persona)
    logger.info(f"persona {persona.persona_id}: NEG allocation = {neg_alloc}")

    # NEG-A
    neg_a = _generate_neg_a(persona, schema, neg_alloc["NEG-A"], cfg, client)
    neg_a_sessions = _distribute_neg_to_sessions(
        "NEG-A", len(neg_a), num_sessions, rng,
    )
    for q, s in zip(neg_a, neg_a_sessions):
        eval_questions.append(EvalQuestion(
            question_id=_next_qid(),
            persona_id=persona.persona_id,
            question=q["question"],
            task_type="NEG",
            neg_subtype="NEG-A",
            domain=q.get("domain", "人际"),
            required_vars=[],
            ask_at_sessions=[s],
            gold_answers={s: "无法回答（剧本中未提及）"},
            f_keys={s: []},
        ))

    # NEG-B
    neg_b = _generate_neg_b(persona, schema, evolution, plan, neg_alloc["NEG-B"], num_sessions)
    neg_b_sessions = _distribute_neg_to_sessions(
        "NEG-B", len(neg_b), num_sessions, rng,
        earliest_session_for_b_c=main_change_session,
    )
    for q, s in zip(neg_b, neg_b_sessions):
        var = q["required_vars"][0]
        snap = evolution.trajectory[s]
        # 找出 s 之前最近一次该 var 的旧值
        prev_value = ""
        for prev_s in range(s - 1, -1, -1):
            ps = evolution.trajectory[prev_s]
            if ps.state.get(var) != snap.state.get(var):
                prev_value = str(ps.state.get(var, ""))
                break
        eval_questions.append(EvalQuestion(
            question_id=_next_qid(),
            persona_id=persona.persona_id,
            question=q["question"],
            task_type="NEG",
            neg_subtype="NEG-B",
            domain=q["domain"],
            required_vars=q["required_vars"],
            ask_at_sessions=[s],
            gold_answers={s: str(snap.state.get(var, ""))},
            f_keys={s: [[var, str(snap.state.get(var, ""))]]},
            trap_answers={s: f"{prev_value}（过期值）"} if prev_value else {},
        ))

    # NEG-C
    neg_c = _generate_neg_c(persona, schema, neg_alloc["NEG-C"])
    neg_c_sessions = _distribute_neg_to_sessions(
        "NEG-C", len(neg_c), num_sessions, rng,
        earliest_session_for_b_c=main_change_session,
    )
    for q, s in zip(neg_c, neg_c_sessions):
        var = q["required_vars"][0]
        snap = evolution.trajectory[s]
        eval_questions.append(EvalQuestion(
            question_id=_next_qid(),
            persona_id=persona.persona_id,
            question=q["question"],
            task_type="NEG",
            neg_subtype="NEG-C",
            domain=q["domain"],
            required_vars=q["required_vars"],
            ask_at_sessions=[s],
            gold_answers={s: str(snap.state.get(var, ""))},
            f_keys={s: [[var, str(snap.state.get(var, ""))]]},
        ))

    # NEG-D
    neg_d = _generate_neg_d(persona, schema, neg_alloc["NEG-D"], cfg, client)
    neg_d_sessions = _distribute_neg_to_sessions(
        "NEG-D", len(neg_d), num_sessions, rng,
    )
    for q, s in zip(neg_d, neg_d_sessions):
        eval_questions.append(EvalQuestion(
            question_id=_next_qid(),
            persona_id=persona.persona_id,
            question=q["question"],
            task_type="NEG",
            neg_subtype="NEG-D",
            domain=q.get("domain", "人际"),
            required_vars=[],
            ask_at_sessions=[s],
            gold_answers={s: "无法回答（剧本中未提及）"},
            f_keys={s: []},
        ))

    # === Step E: 统计 + 落盘 ===
    asks_per_session: dict = defaultdict(int)
    for eq in eval_questions:
        for s in eq.ask_at_sessions:
            asks_per_session[s] += 1
    logger.info(
        f"persona {persona.persona_id}: total {len(eval_questions)} questions, "
        f"asks per session = {dict(sorted(asks_per_session.items()))}, "
        f"sum asks = {sum(asks_per_session.values())}"
    )

    if write_file:
        pdir = persona_data_dir(cfg, persona.persona_id)
        path = pdir / "eval_questions.json"
        path.write_text(
            json.dumps(
                [eq.to_dict() for eq in eval_questions],
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(f"wrote {path}")

    return eval_questions


def load_eval_questions(cfg: DatasetConfig, persona_id: str) -> list[EvalQuestion]:
    pdir = persona_data_dir(cfg, persona_id)
    path = pdir / "eval_questions.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [EvalQuestion.from_dict(d) for d in raw]


__all__ = [
    "build_eval_questions",
    "load_eval_questions",
    "compute_ask_at_sessions_for_pos",
]
