"""
Runtime 阶段 10 模块入口。

Runtime 处理在线 on-policy 交互，提供：
- user_simulator：on-policy 用户模拟器（GPT-4o）+ force_expose + mark_exposed
- session_runner：单 session / 完整 persona 在线交互执行器
- MemorySystemAdapter：被测系统必须实现的最小接口

所有配置 / LLM 客户端 / 数据 schema 复用 builder 包，runtime 不重复定义。
"""
from __future__ import annotations

# 重导出 builder 中常用类型，方便 runtime 调用方一站式 import
from builder.config import DatasetConfig
from builder.llm_client import LLMClient, build_client
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

from .session_runner import (
    MemorySystemAdapter,
    run_persona_with_system,
    run_session_with_system,
)
from .user_simulator import (
    UserSimResponse,
    call_force_expose,
    call_user_simulator,
    llm_check_exposure,
    mark_exposed,
)

__all__ = [
    # builder re-exports
    "DatasetConfig",
    "LLMClient",
    "build_client",
    "DialogueTurn",
    "ExposureItem",
    "ExposurePlan",
    "Persona",
    "SeedUtterance",
    "SessionDialogue",
    "SessionExposurePlan",
    "StateEvolution",
    # runtime
    "MemorySystemAdapter",
    "run_persona_with_system",
    "run_session_with_system",
    "UserSimResponse",
    "call_force_expose",
    "call_user_simulator",
    "llm_check_exposure",
    "mark_exposed",
]
