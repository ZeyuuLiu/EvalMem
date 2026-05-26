from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from memory_eval.eval_core.high_recall import EncodingHighRecallRetriever

from memory_eval.eval_core.utils import normalize_text
from memory_eval.memwiki.aux_record import normalize_aux_records


MEMWIKI_AUX_RECORDS_KEY = "memwiki_aux_records"


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_keys_path() -> Path:
    return project_root() / "configs" / "keys.local.json"


def load_runtime_credentials(keys_path: Optional[str] = None, require_complete: bool = False) -> Dict[str, str]:
    path_api_key = ""
    path_base_url = ""
    path_model = ""
    path_system_model = ""
    path_eval_model = ""
    candidate = Path(keys_path).resolve() if keys_path else default_keys_path()
    if candidate.exists():
        raw = json.loads(candidate.read_text(encoding="utf-8-sig"))
        path_api_key = str(raw.get("api_key", "")).strip()
        path_base_url = str(raw.get("base_url", "")).strip()
        path_model = str(raw.get("model", "")).strip()
        path_system_model = str(raw.get("system_model", "")).strip()
        path_eval_model = str(raw.get("eval_model", "")).strip()

    api_key = os.getenv("MEMORY_EVAL_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip() or path_api_key
    base_url = os.getenv("MEMORY_EVAL_BASE_URL", "").strip() or os.getenv("OPENAI_BASE_URL", "").strip() or path_base_url
    system_model = (
        os.getenv("MEMORY_SYSTEM_MODEL", "").strip()
        or os.getenv("MEMORY_EVAL_SYSTEM_MODEL", "").strip()
        or path_system_model
        or path_model
    )
    eval_model = (
        os.getenv("MEMORY_EVAL_LLM_MODEL", "").strip()
        or os.getenv("MEMORY_EVAL_MODEL", "").strip()
        or path_eval_model
        or system_model
    )
    model = system_model
    if require_complete and (not api_key or not base_url):
        raise ValueError(
            "缺少 API 凭据：请设置 MEMORY_EVAL_API_KEY/MEMORY_EVAL_BASE_URL（或 OPENAI_API_KEY/OPENAI_BASE_URL），"
            "或提供本地 keys 文件。"
        )
    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "system_model": system_model,
        "eval_model": eval_model,
        "keys_path": str(candidate),
    }


class BaseMemoryAdapter:
    family: str = "unknown"
    flavor: str = "default"

    def __init__(self) -> None:
        self._project_root = project_root()
        self._external_high_recall_retriever: EncodingHighRecallRetriever | None = None

    def capabilities(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "flavor": self.flavor,
            "supports_full_memory_export": True,
            "supports_native_retrieval": True,
            "supports_oracle_generation": True,
            "supports_online_generation": True,
            "supports_high_recall_candidates": True,
            "supports_memwiki_aux_ingest": True,
            "memwiki_aux_strategy": "append_to_runtime_context",
        }

    def runtime_manifest(self) -> Dict[str, Any]:
        caps = self.capabilities()
        return {
            "family": caps.get("family", self.family),
            "flavor": caps.get("flavor", self.flavor),
            "capabilities": caps,
        }

    def set_external_high_recall_retriever(self, retriever: EncodingHighRecallRetriever | None) -> None:
        self._external_high_recall_retriever = retriever

    def get_external_high_recall_retriever(self) -> EncodingHighRecallRetriever | None:
        return self._external_high_recall_retriever

    def ingest_aux_records(self, run_ctx: Any, aux_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Inject retrieval-friendly auxiliary records into the adapter runtime context.

        Adapters with a native writable index can override this method to push
        the same records into their own store. The default keeps records in
        run_ctx and export_full_memory() implementations append them.
        """
        normalized = normalize_aux_records(aux_records)
        if not isinstance(run_ctx, dict):
            raise TypeError("default ingest_aux_records requires dict run_ctx")
        existing = normalize_aux_records(list(run_ctx.get(MEMWIKI_AUX_RECORDS_KEY, [])))
        merged = self._merge_aux_records(existing, normalized)
        run_ctx[MEMWIKI_AUX_RECORDS_KEY] = merged
        run_ctx["memwiki_aux_record_count"] = len(merged)
        return merged

    def export_aux_records(self, run_ctx: Any) -> List[Dict[str, Any]]:
        """Return MemWiki auxiliary records stored in the runtime context."""
        if not isinstance(run_ctx, dict):
            return []
        return normalize_aux_records(list(run_ctx.get(MEMWIKI_AUX_RECORDS_KEY, [])))

    def append_aux_records(self, memory_records: List[Dict[str, Any]], run_ctx: Any) -> List[Dict[str, Any]]:
        """Append aux records to an exported memory view with id/text dedup."""
        return self._merge_aux_records(list(memory_records or []), self.export_aux_records(run_ctx))

    def _merge_aux_records(self, primary: List[Dict[str, Any]], secondary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: set[str] = set()
        out: List[Dict[str, Any]] = []
        for bucket in (primary, secondary):
            for record in bucket:
                rid = str(record.get("id", "")).strip()
                text = str(record.get("text", "")).strip()
                if not text:
                    continue
                key = rid or normalize_text(text)[:400]
                if key in seen:
                    continue
                seen.add(key)
                out.append(record)
        return out

    def merge_runtime_credentials(
        self,
        *,
        api_key: str = "",
        base_url: str = "",
        model: str = "",
        keys_path: str = "",
        require_complete: bool = False,
    ) -> Dict[str, str]:
        creds = load_runtime_credentials(keys_path or None, require_complete=require_complete)
        return {
            "api_key": str(api_key or creds.get("api_key", "")).strip(),
            "base_url": str(base_url or creds.get("base_url", "")).strip(),
            "model": str(model or creds.get("model", "")).strip(),
            "keys_path": str(keys_path or creds.get("keys_path", "")),
        }

    def normalize_turns(self, conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for idx, turn in enumerate(conversation):
            text = str(turn.get("text") or turn.get("content") or "").strip()
            if not text:
                continue
            out.append(
                {
                    "turn_index": int(turn.get("turn_index", idx)),
                    "speaker": str(turn.get("speaker") or turn.get("role") or "UNKNOWN").strip(),
                    "text": text,
                    "timestamp": str(turn.get("timestamp") or turn.get("time") or "").strip(),
                }
            )
        return out

    def guess_user_name(self, turns: List[Dict[str, Any]]) -> str:
        counts: Dict[str, int] = {}
        for turn in turns:
            speaker = str(turn.get("speaker", "")).strip()
            if not speaker:
                continue
            counts[speaker] = counts.get(speaker, 0) + 1
        if not counts:
            return "User"
        return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]

    def guess_agent_name(self, turns: List[Dict[str, Any]], user_name: str) -> str:
        speakers = [str(turn.get("speaker", "")).strip() for turn in turns if str(turn.get("speaker", "")).strip()]
        for speaker in speakers:
            if speaker != user_name:
                return speaker
        return "Assistant"

    def build_run_id(self, prefix: str, sample_id: str) -> str:
        norm_prefix = normalize_text(prefix or self.family).replace(" ", "_")
        norm_sample = normalize_text(sample_id).replace(" ", "_")
        return f"{norm_prefix}_{norm_sample}"[:80]
