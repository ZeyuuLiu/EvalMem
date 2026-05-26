"""
配置加载模块。负责从 yaml/json 加载所有配置，提供 typed 接口。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# =====================================================================
# 路径解析（项目根目录 = 本文件向上 2 级）
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"


def project_root() -> Path:
    return PROJECT_ROOT


# =====================================================================
# 全局配置 dataclass
# =====================================================================

@dataclass
class ScaleConfig:
    num_personas: int
    num_sessions: int
    num_turns_per_session: int
    num_questions_per_session: int


@dataclass
class StateEvolutionConfig:
    main_core_change_session: int
    rollback_personas: list[str]
    rollback_session: int
    cascade_immediate_personas: list[str]
    cascade_delayed_personas: list[str]


@dataclass
class QuestionDistributionConfig:
    pos_per_persona: int
    neg_per_persona: int
    pos_fresh_target: int
    pos_tracking_target: int
    pos_tracking_max_asks: int
    question_sampler_target: int
    pos_fresh_avg: int
    pos_tracking_avg: int
    neg_avg: int


@dataclass
class LLMConfig:
    builder_model: str
    builder_temperature: float
    schema_temperature: float
    user_sim_model: str
    user_sim_temperature: float
    exposure_check_model: str
    oracle_dialogue_model: str
    verifier_model: str
    verifier_temperature: float
    request_timeout_sec: int
    max_retries: int
    retry_backoff_sec: int

    api_key: str = ""
    base_url: str = ""


@dataclass
class PathsConfig:
    data_root: Path
    prompts_root: Path
    cache_root: Path


@dataclass
class QualityThresholds:
    verification_pass_rate_min: float
    human_kappa_min: float
    exposure_completion_rate_min: float


@dataclass
class DatasetConfig:
    dataset_version: str
    seed: int
    scale: ScaleConfig
    state_evolution: StateEvolutionConfig
    question_distribution: QuestionDistributionConfig
    llm: LLMConfig
    paths: PathsConfig
    quality_thresholds: QualityThresholds


# =====================================================================
# 加载函数
# =====================================================================

def _resolve_path(p: str) -> Path:
    """yaml 中的相对路径以 project_root 为基准。"""
    path = Path(p)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(config_path: Path | None = None, keys_path: Path | None = None) -> DatasetConfig:
    """加载主配置 + 注入 API key。"""
    if config_path is None:
        config_path = CONFIG_ROOT / "dataset_config.yaml"
    if keys_path is None:
        keys_path = CONFIG_ROOT / "keys.local.json"

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    with open(keys_path, encoding="utf-8") as f:
        keys = json.load(f)

    qd = raw["question_distribution"]
    return DatasetConfig(
        dataset_version=raw["dataset_version"],
        seed=int(raw["seed"]),
        scale=ScaleConfig(**raw["scale"]),
        state_evolution=StateEvolutionConfig(**raw["state_evolution"]),
        question_distribution=QuestionDistributionConfig(
            pos_per_persona=int(qd["pos_per_persona"]),
            neg_per_persona=int(qd["neg_per_persona"]),
            pos_fresh_target=int(qd["pos_fresh_target"]),
            pos_tracking_target=int(qd["pos_tracking_target"]),
            pos_tracking_max_asks=int(qd["pos_tracking_max_asks"]),
            question_sampler_target=int(qd["question_sampler_target"]),
            pos_fresh_avg=int(qd["per_session"]["pos_fresh_avg"]),
            pos_tracking_avg=int(qd["per_session"]["pos_tracking_avg"]),
            neg_avg=int(qd["per_session"]["neg_avg"]),
        ),
        llm=LLMConfig(
            **raw["llm"],
            api_key=str(keys.get("api_key", "")),
            base_url=str(keys.get("base_url", "")),
        ),
        paths=PathsConfig(
            data_root=_resolve_path(raw["paths"]["data_root"]),
            prompts_root=_resolve_path(raw["paths"]["prompts_root"]),
            cache_root=_resolve_path(raw["paths"]["cache_root"]),
        ),
        quality_thresholds=QualityThresholds(**raw["quality_thresholds"]),
    )


# =====================================================================
# 画像维度 + 状态变量库 配置加载
# =====================================================================

@dataclass
class PersonaDimensionsConfig:
    personas_predefined: list[dict]  # 10 条预设
    rho_map: dict[str, float]
    communication_style_detailed: dict[str, str]


def load_persona_dimensions() -> PersonaDimensionsConfig:
    path = CONFIG_ROOT / "persona_dimensions.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return PersonaDimensionsConfig(
        personas_predefined=raw["personas_predefined"],
        rho_map=raw["rho_map"],
        communication_style_detailed=raw["communication_style_detailed"],
    )


@dataclass
class StateVarLibrary:
    """按 domain 索引的状态变量候选库。"""
    by_domain: dict[str, list[dict]]

    def all_vars(self) -> list[dict]:
        out = []
        for vars_in_domain in self.by_domain.values():
            out.extend(vars_in_domain)
        return out

    def find(self, var_name: str) -> dict | None:
        for vars_in_domain in self.by_domain.values():
            for v in vars_in_domain:
                if v["var_name"] == var_name:
                    return v
        return None

    def get_vars_for_domain(self, domain: str) -> list[dict]:
        return self.by_domain.get(domain, [])


def load_state_var_library() -> StateVarLibrary:
    path = CONFIG_ROOT / "state_var_library.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return StateVarLibrary(by_domain=raw)


# =====================================================================
# 数据落盘路径辅助
# =====================================================================

def persona_data_dir(cfg: DatasetConfig, persona_id: str) -> Path:
    p = cfg.paths.data_root / "per_persona" / persona_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir(cfg: DatasetConfig, subdir: str = "") -> Path:
    p = cfg.paths.cache_root
    if subdir:
        p = p / subdir
    p.mkdir(parents=True, exist_ok=True)
    return p


__all__ = [
    "PROJECT_ROOT",
    "CONFIG_ROOT",
    "project_root",
    "ScaleConfig",
    "StateEvolutionConfig",
    "QuestionDistributionConfig",
    "LLMConfig",
    "PathsConfig",
    "QualityThresholds",
    "DatasetConfig",
    "load_config",
    "PersonaDimensionsConfig",
    "load_persona_dimensions",
    "StateVarLibrary",
    "load_state_var_library",
    "persona_data_dir",
    "cache_dir",
]
