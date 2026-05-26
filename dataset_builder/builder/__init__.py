"""DynaMem-Bench builder package — offline pre-generation stages 0..7."""
from __future__ import annotations

from .config import (
    DatasetConfig,
    StateVarLibrary,
    PersonaDimensionsConfig,
    load_config,
    load_persona_dimensions,
    load_state_var_library,
    persona_data_dir,
    cache_dir,
)
from .llm_client import LLMClient, LLMCallError, build_client
from .schemas import (
    Persona,
    StateVariable,
    CascadeRule,
    StateSchema,
    StateSnapshot,
    InterferenceItem,
    StateEvolution,
    ExposureItem,
    SessionExposurePlan,
    ExposurePlan,
    SeedUtterance,
    EvalQuestion,
    OracleContext,
)
from .verifier import verify_persona

__all__ = [
    "DatasetConfig",
    "StateVarLibrary",
    "PersonaDimensionsConfig",
    "load_config",
    "load_persona_dimensions",
    "load_state_var_library",
    "persona_data_dir",
    "cache_dir",
    "LLMClient",
    "LLMCallError",
    "build_client",
    "Persona",
    "StateVariable",
    "CascadeRule",
    "StateSchema",
    "StateSnapshot",
    "InterferenceItem",
    "StateEvolution",
    "ExposureItem",
    "SessionExposurePlan",
    "ExposurePlan",
    "SeedUtterance",
    "EvalQuestion",
    "OracleContext",
]
