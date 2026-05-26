from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from memory_eval.adapters.base import BaseMemoryAdapter
from memory_eval.eval_core.models import AdapterTrace, RetrievedItem
from memory_eval.eval_core.utils import normalize_text, split_tokens, text_match


@dataclass(frozen=True)
class GenericTextAdapterConfig:
    family: str = "generic_text"
    flavor: str = "generic_text_export"
    source_system_dir: str = ""
    memory_dir: str = "outputs/generic_text_memory"
    run_id_prefix: str = "generic_text"
    api_key: str = ""
    base_url: str = ""
    llm_model: str = ""
    keys_path: str = ""


class GenericTextMemoryAdapter(BaseMemoryAdapter):
    """
    Explicit lightweight adapter for systems whose native runtime is not wired yet.

    It is intentionally marked as generic_text_export in runtime manifests.  This
    gives the evaluation framework a runnable seventh-baseline surface without
    misrepresenting it as a faithful native reproduction.
    """

    family = "generic_text"
    flavor = "generic_text_export"

    def __init__(self, config: Optional[GenericTextAdapterConfig] = None):
        super().__init__()
        self.config = config or GenericTextAdapterConfig()
        self.family = str(self.config.family or "generic_text")
        self.flavor = str(self.config.flavor or "generic_text_export")

    def capabilities(self) -> Dict[str, Any]:
        out = super().capabilities()
        out.update(
            {
                "family": self.family,
                "flavor": self.flavor,
                "supports_real_native_runtime": False,
                "supports_lightweight_fallback": True,
                "native_runtime_status": "not_implemented_generic_text_export",
                "source_system_dir": self.config.source_system_dir,
            }
        )
        return out

    def ingest_conversation(self, sample_id: str, conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
        turns = self.normalize_turns(conversation)
        run_id = self.build_run_id(self.config.run_id_prefix or self.family, sample_id)
        output_root = (Path(self.config.memory_dir) / sample_id).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        return {
            "sample_id": sample_id,
            "run_id": run_id,
            "output_root": str(output_root),
            "conversation": turns,
            "memory_view": self._build_memory_from_turns(turns),
            "mode": self.flavor,
            "source_system_dir": self.config.source_system_dir,
        }

    def export_full_memory(self, run_ctx: Any) -> List[Dict[str, Any]]:
        raw = list(run_ctx.get("memory_view", [])) if isinstance(run_ctx, dict) else []
        return self.append_aux_records(raw, run_ctx)

    def find_memory_records(
        self,
        run_ctx: Any,
        query: str,
        f_key: List[str],
        memory_corpus: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        signals = [query] + list(f_key or [])
        signal_tokens: set[str] = set()
        for signal in signals:
            signal_tokens.update(split_tokens(str(signal)))
        scored: List[tuple[float, Dict[str, Any]]] = []
        for record in memory_corpus:
            text = str(record.get("text", ""))
            if any(text_match(fact, text) for fact in f_key if str(fact).strip()):
                scored.append((10.0, record))
                continue
            tokens = set(split_tokens(text))
            overlap = len(tokens & signal_tokens) if tokens and signal_tokens else 0
            if overlap > 0:
                scored.append((float(overlap), record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:100]]

    def hybrid_retrieve_candidates(
        self,
        run_ctx: Any,
        query: str,
        f_key: List[str],
        evidence_texts: List[str],
        top_n: int = 100,
    ) -> List[Dict[str, Any]]:
        combined = " ".join([query] + list(f_key or []) + list(evidence_texts or []))
        return self.retrieve_original(run_ctx, combined or query, top_k=top_n)

    def retrieve_original(self, run_ctx: Any, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        memory = self.export_full_memory(run_ctx)
        query_tokens = set(split_tokens(query))
        scored: List[Dict[str, Any]] = []
        for idx, record in enumerate(memory):
            text = str(record.get("text", ""))
            tokens = set(split_tokens(text))
            overlap = len(query_tokens & tokens) if query_tokens and tokens else 0
            denom = len(query_tokens) or 1
            meta = dict(record.get("meta", {})) if isinstance(record.get("meta", {}), dict) else {}
            scored.append(
                {
                    "id": str(record.get("id", f"generic-{idx}")),
                    "text": text,
                    "score": float(overlap / denom),
                    "meta": {
                        **meta,
                        "source": meta.get("source", f"{self.family}_generic_text_retrieval"),
                        "score_source": "lexical_overlap_generic",
                    },
                }
            )
        scored.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return scored[: max(1, int(top_k or 1))]

    def generate_online_answer(self, run_ctx: Any, query: str, top_k: int = 5) -> str:
        retrieved = self.retrieve_original(run_ctx, query, top_k=top_k)
        if not retrieved:
            return "I don't know"
        return str(retrieved[0].get("text", "")).strip() or "I don't know"

    def generate_oracle_answer(self, run_ctx: Any, query: str, oracle_context: str) -> str:
        text = str(oracle_context or "").strip()
        if not text or normalize_text(text) == "no_relevant_memory":
            return "I don't know"
        return text.splitlines()[0].strip()

    def build_trace_for_query(self, run_ctx: Any, query: str, oracle_context: str, top_k: int) -> AdapterTrace:
        memory_view = self.export_full_memory(run_ctx)
        raw_items = self.retrieve_original(run_ctx, query, top_k=top_k)
        retrieved = [
            RetrievedItem(
                id=str(item.get("id", "")),
                text=str(item.get("text", "")),
                score=float(item.get("score", 0.0) or 0.0),
                meta=dict(item.get("meta", {})) if isinstance(item.get("meta", {}), dict) else {},
            )
            for item in raw_items
        ]
        return AdapterTrace(
            memory_view=memory_view,
            retrieved_items=retrieved,
            answer_online=self.generate_online_answer(run_ctx, query, top_k=top_k),
            answer_oracle=self.generate_oracle_answer(run_ctx, query, oracle_context),
            raw_trace={
                "memory_system": self.family,
                "mode": self.flavor,
                "native_runtime_status": "not_implemented_generic_text_export",
            },
        )

    def export_build_artifact(self, run_ctx: Any) -> Dict[str, Any]:
        return {
            "sample_id": str(run_ctx.get("sample_id", "")) if isinstance(run_ctx, dict) else "",
            "run_id": str(run_ctx.get("run_id", "")) if isinstance(run_ctx, dict) else "",
            "output_root": str(run_ctx.get("output_root", "")) if isinstance(run_ctx, dict) else "",
            "source_system_dir": self.config.source_system_dir,
        }

    def load_build_artifact(self, manifest: Dict[str, Any]) -> Any:
        raise RuntimeError(
            f"{self.family} generic_text_export adapter does not persist full runtime state; run ingest on the dataset."
        )

    def _build_memory_from_turns(self, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for turn in turns:
            speaker = str(turn.get("speaker", "")).strip() or "UNKNOWN_SPEAKER"
            timestamp = str(turn.get("timestamp", "")).strip() or "UNKNOWN_TIME"
            text = str(turn.get("text", "")).strip()
            if not text:
                continue
            out.append(
                {
                    "id": f"{self.family}-turn-{turn.get('turn_index', len(out))}",
                    "text": f"{timestamp} | {speaker}: {text}",
                    "meta": {
                        "source": f"{self.family}_conversation_cache",
                        "turn_index": int(turn.get("turn_index", len(out)) or 0),
                        "speaker": speaker,
                        "timestamp": timestamp,
                        "raw_text": text,
                    },
                }
            )
        return out


__all__ = ["GenericTextAdapterConfig", "GenericTextMemoryAdapter"]
