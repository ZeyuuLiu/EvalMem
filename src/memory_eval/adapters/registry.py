from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict

from memory_eval.adapters.gam_adapter import GAMAdapter, GAMAdapterConfig
from memory_eval.adapters.generic_text_adapter import GenericTextAdapterConfig, GenericTextMemoryAdapter
from memory_eval.adapters.membox_adapter import MemboxAdapter, MemboxAdapterConfig
from memory_eval.adapters.memos_adapter import MemOSAdapter, MemOSAdapterConfig
from memory_eval.adapters.o_mem_adapter import OMemAdapter, OMemAdapterConfig
from memory_eval.adapters.timem_adapter import TimemAdapter, TimemAdapterConfig
from memory_eval.adapters.memoryos_adapter import MemoryOSAdapter, MemoryOSAdapterConfig
from memory_eval.adapters.evermemos_adapter import EverMemOSAdapter, EverMemOSAdapterConfig


AdapterBuilder = Callable[[Dict[str, Any]], Any]


def _redact_secrets(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for key, value in obj.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("api_key", "apikey", "token", "secret", "password")):
                out[key] = "***REDACTED***" if value else value
            else:
                out[key] = _redact_secrets(value)
        return out
    if isinstance(obj, list):
        return [_redact_secrets(x) for x in obj]
    return obj


def _build_membox(raw: Dict[str, Any]) -> MemboxAdapter:
    cfg = MemboxAdapterConfig(**raw)
    return MemboxAdapter(config=cfg)


def _build_membox_stable_eval(raw: Dict[str, Any]) -> MemboxAdapter:
    cfg_raw = dict(raw)
    if not cfg_raw.get("membox_root"):
        cfg_raw["membox_root"] = str(Path(__file__).resolve().parents[3] / "system" / "Membox_stableEval")
    cfg = MemboxAdapterConfig(**cfg_raw)
    return MemboxAdapter(config=cfg)


def _build_o_mem(raw: Dict[str, Any]) -> OMemAdapter:
    cfg = OMemAdapterConfig(**raw)
    return OMemAdapter(config=cfg)


def _build_o_mem_stable_eval(raw: Dict[str, Any]) -> OMemAdapter:
    cfg_raw = dict(raw)
    if not cfg_raw.get("omem_root"):
        cfg_raw["omem_root"] = str(Path(__file__).resolve().parents[3] / "system" / "O-Mem-StableEval")
    cfg = OMemAdapterConfig(**cfg_raw)
    return OMemAdapter(config=cfg)


def _build_gam(raw: Dict[str, Any]) -> GAMAdapter:
    cfg_raw = dict(raw)
    if not cfg_raw.get("gam_root"):
        cfg_raw["gam_root"] = str(Path(__file__).resolve().parents[3] / "system" / "general-agentic-memory-main")
    cfg = GAMAdapterConfig(**cfg_raw)
    return GAMAdapter(config=cfg)


def _build_memos(raw: Dict[str, Any]) -> MemOSAdapter:
    cfg_raw = dict(raw)
    if not cfg_raw.get("memos_root"):
        cfg_raw["memos_root"] = str(Path(__file__).resolve().parents[3] / "system" / "MemOS-main")
    cfg = MemOSAdapterConfig(**cfg_raw)
    return MemOSAdapter(config=cfg)


def _build_generic_text(family: str, source_dir_name: str, raw: Dict[str, Any]) -> GenericTextMemoryAdapter:
    cfg_raw = dict(raw)
    cfg_raw.setdefault("family", family)
    cfg_raw.setdefault("flavor", "generic_text_export")
    cfg_raw.setdefault("run_id_prefix", family)
    cfg_raw.setdefault(
        "source_system_dir",
        str(Path(__file__).resolve().parents[3] / "system" / source_dir_name),
    )
    return GenericTextMemoryAdapter(config=GenericTextAdapterConfig(**cfg_raw))


def _build_timem(raw: Dict[str, Any]) -> TimemAdapter:
    cfg_raw = dict(raw)
    if not cfg_raw.get("timem_root"):
        cfg_raw["timem_root"] = str(Path(__file__).resolve().parents[3] / "system" / "timem-main")
    # filter out unknown keys (this raw dict often has extra membox/o_mem fields)
    allowed = {f for f in TimemAdapterConfig.__dataclass_fields__.keys()}
    cfg = TimemAdapterConfig(**{k: v for k, v in cfg_raw.items() if k in allowed})
    return TimemAdapter(config=cfg)


def _build_memoryos(raw: Dict[str, Any]) -> MemoryOSAdapter:
    cfg_raw = dict(raw)
    if not cfg_raw.get("memoryos_root"):
        cfg_raw["memoryos_root"] = str(Path(__file__).resolve().parents[3] / "system" / "MemoryOS-main")
    allowed = {f for f in MemoryOSAdapterConfig.__dataclass_fields__.keys()}
    cfg = MemoryOSAdapterConfig(**{k: v for k, v in cfg_raw.items() if k in allowed})
    return MemoryOSAdapter(config=cfg)


def _build_everos(raw: Dict[str, Any]) -> EverMemOSAdapter:
    cfg_raw = dict(raw)
    if not cfg_raw.get("everos_root"):
        cfg_raw["everos_root"] = str(Path(__file__).resolve().parents[3] / "system" / "EverOS-main")
    allowed = {f for f in EverMemOSAdapterConfig.__dataclass_fields__.keys()}
    cfg = EverMemOSAdapterConfig(**{k: v for k, v in cfg_raw.items() if k in allowed})
    return EverMemOSAdapter(config=cfg)


def _build_naive_rag(raw: Dict[str, Any]) -> GenericTextMemoryAdapter:
    return _build_generic_text("naive_rag", "", raw)


# One memory system = one dedicated adapter implementation module.
# 一套记忆系统 = 一份独立适配器实现模块。
_ADAPTER_BUILDERS: Dict[str, AdapterBuilder] = {
    "membox": _build_membox,
    "membox_stable_eval": _build_membox_stable_eval,
    "membox:stable_eval": _build_membox_stable_eval,
    "o_mem": _build_o_mem,
    "o_mem_stable_eval": _build_o_mem_stable_eval,
    "o_mem:stable_eval": _build_o_mem_stable_eval,
    "omem": _build_o_mem,
    "omem:stable_eval": _build_o_mem_stable_eval,
    "gam": _build_gam,
    "gam_stable_eval": _build_gam,
    "general_agentic_memory": _build_gam,
    "memos": _build_memos,
    "memos_stable_eval": _build_memos,
    "memos_api": _build_memos,
    "memos-api": _build_memos,
    "timem": _build_timem,
    "timem_stable_eval": _build_timem,
    "time_memory": _build_timem,
    "memoryos": _build_memoryos,
    "memory_os": _build_memoryos,
    "memoryos_stable_eval": _build_memoryos,
    "everos": _build_everos,
    "ever_os": _build_everos,
    "everos_stable_eval": _build_everos,
    "naive_rag": _build_naive_rag,
    "text_rag": _build_naive_rag,
}


def list_supported_memory_systems() -> Dict[str, str]:
    return {k: "registered" for k in sorted(_ADAPTER_BUILDERS.keys())}


def create_adapter_by_system(memory_system: str, config: Dict[str, Any] | None = None) -> Any:
    key = str(memory_system or "").strip().lower()
    if key not in _ADAPTER_BUILDERS:
        supported = ", ".join(sorted(_ADAPTER_BUILDERS.keys())) or "(none)"
        raise ValueError(f"unsupported memory system: {memory_system}. supported: {supported}")
    cfg = dict(config or {})
    return _ADAPTER_BUILDERS[key](cfg)


def export_adapter_runtime_manifest(adapter: Any) -> Dict[str, Any]:
    if hasattr(adapter, "runtime_manifest") and callable(getattr(adapter, "runtime_manifest")):
        out = dict(adapter.runtime_manifest())
        out["adapter_class"] = adapter.__class__.__name__
    else:
        out = {"adapter_class": adapter.__class__.__name__}
    cfg = getattr(adapter, "config", None)
    if cfg is not None:
        if is_dataclass(cfg):
            out["adapter_config"] = _redact_secrets(asdict(cfg))
        elif isinstance(cfg, dict):
            out["adapter_config"] = _redact_secrets(dict(cfg))
        else:
            out["adapter_config"] = {"repr": repr(cfg)}
    return out
