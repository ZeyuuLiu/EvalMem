"""
DynaMem-Bench core dataclasses.

These structures carry input/output between the 8 offline pre-generation
stages (persona pool → state schema → state evolution → exposure plan →
seed utterances → eval questions → oracle contexts → verification).

Design principles:
- All fields are type-hinted.
- All classes support to_dict / from_dict for JSON serialization.
- __post_init__ performs basic structural validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# =====================================================================
# 一、Persona（画像）
# =====================================================================

@dataclass
class Persona:
    persona_id: str
    name: str
    age: int
    gender: str
    education: str
    stage: str
    communication_style: str
    change_propensity: str  # 低/中/高
    rho: float
    neg_sensitivity: str
    preferred_domains: list[str]
    backstory: str

    def __post_init__(self) -> None:
        assert 18 <= self.age <= 80, f"age out of range: {self.age}"
        assert 1 <= len(self.preferred_domains) <= 2
        rho_map = {"低": 0.05, "中": 0.15, "高": 0.30}
        assert abs(self.rho - rho_map[self.change_propensity]) < 1e-6

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Persona:
        return cls(**d)


# =====================================================================
# 二、StateSchema（状态模式）
# =====================================================================

@dataclass
class StateVariable:
    var_name: str
    display_name: str
    values: list[str]
    domain: str
    description: str
    semantic_dynamic_score: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> StateVariable:
        return cls(**d)


@dataclass
class CascadeRule:
    main_var: str
    aux_var: str
    cascade_lag: str  # immediate / delayed
    delayed_session_min: int = 2
    delayed_session_max: int = 4
    cascade_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> CascadeRule:
        return cls(**d)


@dataclass
class StateSchema:
    persona_id: str
    variables: list[StateVariable]
    main_core: str
    auxiliary_cores: list[CascadeRule]

    def get(self, var_name: str) -> StateVariable:
        for v in self.variables:
            if v.var_name == var_name:
                return v
        raise KeyError(f"variable not in schema: {var_name}")

    def to_dict(self) -> dict:
        return {
            "persona_id": self.persona_id,
            "variables": [v.to_dict() for v in self.variables],
            "main_core": self.main_core,
            "auxiliary_cores": [c.to_dict() for c in self.auxiliary_cores],
        }

    @classmethod
    def from_dict(cls, d: dict) -> StateSchema:
        return cls(
            persona_id=d["persona_id"],
            variables=[StateVariable.from_dict(v) for v in d["variables"]],
            main_core=d["main_core"],
            auxiliary_cores=[CascadeRule.from_dict(c) for c in d["auxiliary_cores"]],
        )


# =====================================================================
# 三、StateEvolution（状态演化）
# =====================================================================

StateAssignment = dict  # var_name -> value


@dataclass
class StateSnapshot:
    session: int  # 0..10
    state: StateAssignment
    event: str
    changes_from_prev: list[str]
    is_main_core_change: bool = False
    is_aux_cascade: bool = False
    is_rollback: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> StateSnapshot:
        return cls(**d)


@dataclass
class InterferenceItem:
    var: str
    old_value: str
    expired_at_session: int
    label: str  # stale / cascaded / rollback_reactivated
    cause_session: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> InterferenceItem:
        return cls(**d)


@dataclass
class StateEvolution:
    persona_id: str
    trajectory: list[StateSnapshot]  # len = 11 (session 0..10)
    interference_set: dict[int, list[InterferenceItem]]  # session_t → I_t

    def to_dict(self) -> dict:
        return {
            "persona_id": self.persona_id,
            "trajectory": [s.to_dict() for s in self.trajectory],
            "interference_set": {
                str(k): [i.to_dict() for i in v]
                for k, v in self.interference_set.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> StateEvolution:
        return cls(
            persona_id=d["persona_id"],
            trajectory=[StateSnapshot.from_dict(s) for s in d["trajectory"]],
            interference_set={
                int(k): [InterferenceItem.from_dict(i) for i in v]
                for k, v in d["interference_set"].items()
            },
        )


# =====================================================================
# 四、ExposurePlan（信息暴露计划）
# =====================================================================

@dataclass
class ExposureItem:
    var: str
    value: str
    tone_hint: str = ""
    is_state_change: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ExposureItem:
        return cls(**d)


@dataclass
class SessionExposurePlan:
    session: int
    must_expose: list[ExposureItem]
    expose_sentiment: str = "neutral"
    session_theme: str = ""

    def to_dict(self) -> dict:
        return {
            "session": self.session,
            "must_expose": [e.to_dict() for e in self.must_expose],
            "expose_sentiment": self.expose_sentiment,
            "session_theme": self.session_theme,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SessionExposurePlan:
        return cls(
            session=d["session"],
            must_expose=[ExposureItem.from_dict(e) for e in d["must_expose"]],
            expose_sentiment=d.get("expose_sentiment", "neutral"),
            session_theme=d.get("session_theme", ""),
        )


@dataclass
class ExposurePlan:
    persona_id: str
    sessions: list[SessionExposurePlan]  # len = 10

    def to_dict(self) -> dict:
        return {
            "persona_id": self.persona_id,
            "sessions": [s.to_dict() for s in self.sessions],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExposurePlan:
        return cls(
            persona_id=d["persona_id"],
            sessions=[SessionExposurePlan.from_dict(s) for s in d["sessions"]],
        )


# =====================================================================
# 五、SeedUtterance（种子话术）
# =====================================================================

@dataclass
class SeedUtterance:
    persona_id: str
    session: int
    seed_text: str
    expects_to_expose: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SeedUtterance:
        return cls(**d)


# =====================================================================
# 六、EvalQuestion（评测问题，最复杂的 schema）
# =====================================================================

@dataclass
class EvalQuestion:
    question_id: str
    persona_id: str
    question: str
    task_type: str  # POS / NEG
    domain: str
    required_vars: list[str]
    ask_at_sessions: list[int]
    gold_answers: dict[int, str]    # session → 标准答案
    f_keys: dict[int, list[list[str]]]  # session → [[var, value], ...]
    trap_answers: dict[int, str] = field(default_factory=dict)
    neg_subtype: Optional[str] = None  # NEG-A / NEG-B / NEG-C / NEG-D
    is_state_tracking: bool = False
    cascade_metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # JSON 不接受 int key，转成 str
        d["gold_answers"] = {str(k): v for k, v in self.gold_answers.items()}
        d["f_keys"] = {str(k): v for k, v in self.f_keys.items()}
        d["trap_answers"] = {str(k): v for k, v in self.trap_answers.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> EvalQuestion:
        return cls(
            question_id=d["question_id"],
            persona_id=d["persona_id"],
            question=d["question"],
            task_type=d["task_type"],
            domain=d["domain"],
            required_vars=list(d.get("required_vars", [])),
            ask_at_sessions=list(d["ask_at_sessions"]),
            gold_answers={int(k): v for k, v in d["gold_answers"].items()},
            f_keys={int(k): v for k, v in d.get("f_keys", {}).items()},
            trap_answers={int(k): v for k, v in d.get("trap_answers", {}).items()},
            neg_subtype=d.get("neg_subtype"),
            is_state_tracking=d.get("is_state_tracking", False),
            cascade_metadata=d.get("cascade_metadata"),
        )


# =====================================================================
# 七、OracleContext
# =====================================================================

@dataclass
class OracleDialogueTurn:
    user: str
    assistant: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OracleContext:
    question_id: str
    asked_at_session: int
    context_text: str
    source_pointers: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> OracleContext:
        return cls(**d)


# =====================================================================
# 八、SessionDialogue（runtime 产物，留给阶段 10 用）
# =====================================================================

@dataclass
class DialogueTurn:
    turn_index: int
    user_message: str
    assistant_reply: str
    exposed_in_this_msg: list[str]
    is_force_expose: bool = False
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SessionDialogue:
    persona_id: str
    session: int
    system_name: str
    turns: list[DialogueTurn]
    exposure_audit: dict[str, bool]
    fairness_violation: bool = False

    def to_dict(self) -> dict:
        return {
            "persona_id": self.persona_id,
            "session": self.session,
            "system_name": self.system_name,
            "turns": [t.to_dict() for t in self.turns],
            "exposure_audit": dict(self.exposure_audit),
            "fairness_violation": self.fairness_violation,
        }


__all__ = [
    "Persona",
    "StateVariable",
    "CascadeRule",
    "StateSchema",
    "StateAssignment",
    "StateSnapshot",
    "InterferenceItem",
    "StateEvolution",
    "ExposureItem",
    "SessionExposurePlan",
    "ExposurePlan",
    "SeedUtterance",
    "EvalQuestion",
    "OracleDialogueTurn",
    "OracleContext",
    "DialogueTurn",
    "SessionDialogue",
]
