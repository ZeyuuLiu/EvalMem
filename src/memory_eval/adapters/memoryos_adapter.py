"""
MemoryOS adapter — bridges to system/MemoryOS-main with light fallback.

MemoryOS 的核心设计哲学（论文 Kang et al. 2025）：
- Short-term memory: 滚动 N 条最近 QA pair
- Mid-term memory : segment-based "session" with heat scoring
- Long-term memory: user_profile + assistant_knowledge + retrieved knowledge entries
- Retriever 在三层之间做 retrieve_context 融合

本 adapter 优先尝试 import system/MemoryOS-main/memoryos-pypi/memoryos.py 的
`Memoryos` 类（其 API 是 `add_memory(user_input, agent_response, timestamp)` +
`get_response(query)`）。若 import 失败，退回到 light_native 实现：
- short-term: collections.deque
- mid-term  : list of session-summaries (LLM 生成)
- long-term : 全局 user_profile (LLM 抽取，去重写入)
- 检索：embedding 在 mid-term + long-term 中找 top-k

诚实标识：runtime_manifest.flavor = "native_sdk" or "light_native_three_tier"
"""
from __future__ import annotations

import logging
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from memory_eval.adapters.base import BaseMemoryAdapter
from memory_eval.adapters._native_runtime import (
    EmbeddingService,
    InMemoryVectorStore,
    llm_chat,
    parse_timestamp,
)
from memory_eval.eval_core.models import AdapterTrace, RetrievedItem

logger = logging.getLogger(__name__)


@dataclass
class MemoryOSAdapterConfig:
    memoryos_root: str = ""
    api_key: str = ""
    base_url: str = ""
    llm_model: str = "gpt-4o-mini"
    keys_path: str = ""
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    short_term_capacity: int = 10
    mid_term_capacity: int = 200
    long_term_capacity: int = 100
    use_native_sdk: bool = True   # try import Memoryos class first
    data_storage_path: str = ""


class MemoryOSAdapter(BaseMemoryAdapter):
    """MemoryOS native (or light-native fallback) adapter."""

    family = "memoryos"

    def __init__(self, config: Optional[MemoryOSAdapterConfig] = None) -> None:
        super().__init__()
        cfg = config or MemoryOSAdapterConfig()
        creds = self.merge_runtime_credentials(
            api_key=cfg.api_key, base_url=cfg.base_url,
            model=cfg.llm_model, keys_path=cfg.keys_path,
        )
        root = cfg.memoryos_root or str(Path(__file__).resolve().parents[3] / "system" / "MemoryOS-main")
        self.config = MemoryOSAdapterConfig(
            memoryos_root=root,
            api_key=creds["api_key"], base_url=creds["base_url"],
            llm_model=creds["model"] or cfg.llm_model or "gpt-4o-mini",
            keys_path=creds["keys_path"],
            embedding_model_name=cfg.embedding_model_name,
            embedding_device=cfg.embedding_device,
            short_term_capacity=cfg.short_term_capacity,
            mid_term_capacity=cfg.mid_term_capacity,
            long_term_capacity=cfg.long_term_capacity,
            use_native_sdk=cfg.use_native_sdk,
            data_storage_path=cfg.data_storage_path,
        )
        self._embed = EmbeddingService(
            model_name=self.config.embedding_model_name,
            device=self.config.embedding_device,
        )
        # Try to import native Memoryos class
        self._native_cls = None
        if self.config.use_native_sdk:
            self._native_cls = self._try_import_native()
        self.flavor = "native_sdk" if self._native_cls is not None else "light_native_three_tier"

    # --- protocol methods --------------------------------------------------

    def capabilities(self) -> Dict[str, Any]:
        out = super().capabilities()
        out.update({
            "flavor": self.flavor,
            "supports_real_native_runtime": self._native_cls is not None,
            "supports_lightweight_fallback": True,
            "native_runtime_status": "memoryos_pypi" if self._native_cls is not None else "light_native_short_mid_long",
            "embedding_mode": self._embed.mode,
            "design_tiers": ["short_term", "mid_term", "long_term"],
        })
        return out

    def runtime_manifest(self) -> Dict[str, Any]:
        return {"family": self.family, "flavor": self.flavor, "capabilities": self.capabilities()}

    def ingest_conversation(self, sample_id: str, conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
        turns = self.normalize_turns(conversation)
        user_name = self.guess_user_name(turns)
        assistant_name = self.guess_agent_name(turns, user_name)

        if self._native_cls is not None:
            try:
                return self._ingest_native(sample_id, turns, user_name, assistant_name)
            except Exception as exc:
                logger.warning(f"MemoryOS native ingest failed ({exc}); falling back to light_native")
                self.flavor = "light_native_three_tier"

        return self._ingest_light(sample_id, turns, user_name, assistant_name)

    def export_full_memory(self, run_ctx: Any) -> List[Dict[str, Any]]:
        if not isinstance(run_ctx, dict):
            return []
        records = list(run_ctx.get("memory_view", []))
        return self.append_aux_records(records, run_ctx)

    def find_memory_records(self, run_ctx, query, f_key, memory_corpus) -> List[Dict[str, Any]]:
        store: InMemoryVectorStore = run_ctx.get("vector_store") if isinstance(run_ctx, dict) else None
        if not isinstance(store, InMemoryVectorStore) or len(store) == 0:
            return list(memory_corpus or [])[:100]
        signal = " ".join([str(query)] + [str(x) for x in (f_key or [])]).strip()
        if not signal:
            return list(memory_corpus or [])[:100]
        q_vec = self._embed.embed([signal])[0]
        hits = store.search(q_vec, top_k=100)
        return [self._vsi_to_record(item, score) for score, item in hits if score > 0]

    def hybrid_retrieve_candidates(self, run_ctx, query, f_key, evidence_texts, top_n: int = 100) -> List[Dict[str, Any]]:
        signal = " ".join([str(query)] + [str(x) for x in (f_key or [])] + [str(x) for x in (evidence_texts or [])]).strip()
        return self.find_memory_records(run_ctx, signal or query, f_key, run_ctx.get("memory_view", []))[:top_n]

    def retrieve_original(self, run_ctx, query, top_k: int) -> List[Dict[str, Any]]:
        """Three-tier retrieval: short_term + mid_term + long_term knowledge."""
        if self._native_cls is not None and run_ctx.get("native_instance") is not None:
            try:
                return self._retrieve_native(run_ctx, query, top_k)
            except Exception as exc:
                logger.warning(f"MemoryOS native retrieve failed ({exc}); falling back")

        store: InMemoryVectorStore = run_ctx.get("vector_store")
        if not isinstance(store, InMemoryVectorStore) or len(store) == 0:
            return []
        q_vec = self._embed.embed([query])[0]

        # mid-term get top_k * 0.6
        mid_k = max(1, int(top_k * 0.6))
        long_k = max(1, top_k - mid_k)
        mid_hits = store.search(q_vec, top_k=mid_k,
                                filter_fn=lambda it: it.meta.get("tier") == "mid_term")
        long_hits = store.search(q_vec, top_k=long_k,
                                 filter_fn=lambda it: it.meta.get("tier") == "long_term")
        # short-term: just include the most recent K from deque (no filter on relevance)
        short_term = run_ctx.get("short_term", deque())
        short_records: List = []
        for entry in list(short_term)[-self.config.short_term_capacity:]:
            short_records.append((1.0, type("ShortItem", (), {
                "item_id": entry["id"],
                "text": entry["text"],
                "meta": {"tier": "short_term"},
            })()))

        all_hits = mid_hits + long_hits + short_records
        # de-dup
        seen = set()
        out: List[Dict[str, Any]] = []
        for score, item in all_hits:
            iid = getattr(item, "item_id", "")
            if iid in seen:
                continue
            seen.add(iid)
            out.append(self._vsi_to_record(item, score))
        return out[:top_k]

    def generate_online_answer(self, run_ctx, query: str, top_k: int = 5) -> str:
        retrieved = self.retrieve_original(run_ctx, query, top_k=top_k)
        return self._answer_with_context(query, self._render_context(retrieved))

    def generate_oracle_answer(self, run_ctx, query: str, oracle_context: str) -> str:
        return self._answer_with_context(query, str(oracle_context or ""))

    def build_trace_for_query(self, run_ctx, query, oracle_context, top_k: int) -> AdapterTrace:
        memory_view = self.export_full_memory(run_ctx)
        raw = self.retrieve_original(run_ctx, query, top_k=top_k)
        retrieved = [
            RetrievedItem(id=str(r.get("id", "")), text=str(r.get("text", "")),
                          score=float(r.get("score", 0.0) or 0.0),
                          meta=dict(r.get("meta", {})) if isinstance(r.get("meta"), dict) else {})
            for r in raw
        ]
        return AdapterTrace(
            memory_view=memory_view, retrieved_items=retrieved,
            answer_online=self.generate_online_answer(run_ctx, query, top_k=top_k),
            answer_oracle=self.generate_oracle_answer(run_ctx, query, oracle_context),
            raw_trace={"memory_system": self.family, "memory_count": len(memory_view),
                       "retrieved_count": len(retrieved)},
        )

    def export_build_artifact(self, run_ctx) -> Dict[str, Any]:
        return {"sample_id": str(run_ctx.get("sample_id", "")),
                "artifact_refs": dict(run_ctx.get("artifact_refs", {}))}

    def load_build_artifact(self, manifest: Dict[str, Any]) -> Any:
        raise RuntimeError("MemoryOSAdapter: build artifact reuse not implemented; please re-ingest.")

    # --- internal: native bridge ------------------------------------------

    def _try_import_native(self):
        try:
            pypi_path = Path(self.config.memoryos_root) / "memoryos-pypi"
            if str(pypi_path) not in sys.path:
                sys.path.insert(0, str(pypi_path))
            from memoryos import Memoryos  # type: ignore
            return Memoryos
        except Exception as exc:
            logger.info(f"MemoryOS native SDK not available ({exc}); using light_native")
            return None

    def _ingest_native(self, sample_id, turns, user_name, assistant_name) -> Dict[str, Any]:
        """Use the actual Memoryos class from system/MemoryOS-main/memoryos-pypi."""
        import tempfile
        storage = self.config.data_storage_path or tempfile.mkdtemp(prefix=f"memoryos_{sample_id}_")
        instance = self._native_cls(
            user_id=f"{sample_id}_user",
            openai_api_key=self.config.api_key,
            data_storage_path=storage,
            openai_base_url=self.config.base_url or None,
            assistant_id=f"{sample_id}_assistant",
            short_term_capacity=self.config.short_term_capacity,
            mid_term_capacity=self.config.mid_term_capacity,
            long_term_knowledge_capacity=self.config.long_term_capacity,
            llm_model=self.config.llm_model,
        )
        # Pair turns into (user_input, agent_response)
        for i in range(0, len(turns) - 1, 2):
            t_user = turns[i]
            t_agent = turns[i + 1] if i + 1 < len(turns) else {"text": "", "timestamp": t_user.get("timestamp", "")}
            try:
                instance.add_memory(
                    user_input=t_user["text"],
                    agent_response=t_agent["text"],
                    timestamp=t_user.get("timestamp", "") or None,
                )
            except Exception as exc:
                logger.warning(f"MemoryOS native add_memory failed (turn {i}): {exc}")

        # Build a memory_view by exporting whatever we can read
        memory_view: List[Dict[str, Any]] = []
        try:
            short = instance.short_term_memory.get_all()
            for j, qa in enumerate(short):
                memory_view.append({
                    "id": f"memoryos-short-{j}",
                    "text": f"User: {qa.get('user_input', '')}\nAssistant: {qa.get('agent_response', '')}",
                    "meta": {"tier": "short_term", "timestamp": qa.get("timestamp", "")},
                })
        except Exception:
            pass
        try:
            for j, sess in enumerate(getattr(instance.mid_term_memory, "sessions", [])):
                memory_view.append({
                    "id": f"memoryos-mid-{j}",
                    "text": str(sess.get("summary", "")),
                    "meta": {"tier": "mid_term", "session_index": j, "raw": sess},
                })
        except Exception:
            pass
        try:
            profile = instance.user_long_term_memory.get_raw_user_profile(instance.user_id)
            if profile:
                memory_view.append({
                    "id": "memoryos-long-profile",
                    "text": profile,
                    "meta": {"tier": "long_term", "kind": "user_profile"},
                })
        except Exception:
            pass

        # Also build vector store for fallback retrieval
        store = InMemoryVectorStore()
        if memory_view:
            vectors = self._embed.embed([r["text"] for r in memory_view])
            for r, v in zip(memory_view, vectors):
                store.upsert(r["id"], r["text"], v, meta=r["meta"])

        return {
            "sample_id": sample_id, "turns": turns,
            "user_name": user_name, "assistant_name": assistant_name,
            "native_instance": instance, "data_storage_path": storage,
            "memory_view": memory_view,
            "vector_store": store, "short_term": deque(maxlen=self.config.short_term_capacity),
            "mode": "native_sdk", "artifact_refs": {"sample_id": sample_id, "data_storage_path": storage},
        }

    def _retrieve_native(self, run_ctx, query, top_k: int) -> List[Dict[str, Any]]:
        instance = run_ctx["native_instance"]
        result = instance.retriever.retrieve_context(user_query=query, user_id=instance.user_id)
        out: List[Dict[str, Any]] = []
        for j, page in enumerate(result.get("retrieved_pages", [])):
            out.append({
                "id": f"memoryos-mid-page-{j}",
                "text": f"User: {page.get('user_input', '')}\nAssistant: {page.get('agent_response', '')}",
                "score": float(page.get("score", max(top_k - j, 1))),
                "meta": {"tier": "mid_term", "kind": "page", "raw": page},
            })
        for j, kn in enumerate(result.get("retrieved_user_knowledge", [])):
            out.append({
                "id": f"memoryos-long-user-{j}",
                "text": kn.get("knowledge", ""),
                "score": float(kn.get("score", max(top_k - j, 1))),
                "meta": {"tier": "long_term", "kind": "user_knowledge", "raw": kn},
            })
        for j, kn in enumerate(result.get("retrieved_assistant_knowledge", [])):
            out.append({
                "id": f"memoryos-long-assistant-{j}",
                "text": kn.get("knowledge", ""),
                "score": float(kn.get("score", max(top_k - j, 1))),
                "meta": {"tier": "long_term", "kind": "assistant_knowledge", "raw": kn},
            })
        return out[:top_k]

    # --- internal: light_native -------------------------------------------

    def _ingest_light(self, sample_id, turns, user_name, assistant_name) -> Dict[str, Any]:
        # Group turns into QA pairs
        short_term = deque(maxlen=self.config.short_term_capacity)
        mid_term: List[Dict[str, Any]] = []
        # Bucket by ~20 turn segments
        seg_size = max(4, self.config.short_term_capacity * 2)
        all_qa: List[Dict[str, Any]] = []
        for i in range(0, len(turns) - 1, 2):
            t_user = turns[i]
            t_agent = turns[i + 1] if i + 1 < len(turns) else {"text": "", "timestamp": t_user.get("timestamp", "")}
            qa = {
                "id": f"memoryos-qa-{i // 2}",
                "user_input": t_user.get("text", ""),
                "agent_response": t_agent.get("text", ""),
                "text": f"User: {t_user.get('text', '')}\nAssistant: {t_agent.get('text', '')}",
                "timestamp": t_user.get("timestamp", ""),
            }
            all_qa.append(qa)
            short_term.append(qa)

        for seg_start in range(0, len(all_qa), seg_size):
            segment = all_qa[seg_start: seg_start + seg_size]
            full = "\n".join([qa["text"] for qa in segment])[:2400]
            summary = self._llm_chat(
                "Summarize the following dialogue segment into a concise mid-term memory. "
                "Preserve all named entities, decisions, factual claims. 60-120 words.",
                full, max_tokens=300,
            ) or full[:300]
            mid_term.append({
                "id": f"memoryos-mid-{seg_start // seg_size}",
                "text": summary,
                "meta": {"tier": "mid_term", "segment_start": seg_start, "qa_ids": [qa["id"] for qa in segment]},
            })

        # Long-term: distill a user profile from full transcript
        full_transcript = "\n".join([qa["text"] for qa in all_qa])[:4800]
        long_term: List[Dict[str, Any]] = []
        if full_transcript and self.config.api_key:
            profile = self._llm_chat(
                "Extract the user's persistent profile and key knowledge from the dialogue. "
                "Output as bulleted facts (one per line). Each fact ≤ 15 words. ≤ 12 facts.",
                full_transcript, max_tokens=600,
            ) or ""
            for j, line in enumerate([l.strip("- •*").strip() for l in profile.split("\n") if l.strip()][: self.config.long_term_capacity]):
                long_term.append({
                    "id": f"memoryos-long-{j}",
                    "text": line, "meta": {"tier": "long_term", "kind": "user_knowledge"},
                })

        memory_view = (
            [{"id": qa["id"], "text": qa["text"],
              "meta": {"tier": "short_term", "timestamp": qa["timestamp"]}}
             for qa in list(short_term)[-self.config.short_term_capacity:]]
            + mid_term + long_term
        )

        # Vector store across mid + long
        store = InMemoryVectorStore()
        indexable = mid_term + long_term
        if indexable:
            vectors = self._embed.embed([r["text"] for r in indexable])
            for r, v in zip(indexable, vectors):
                store.upsert(r["id"], r["text"], v, meta=r["meta"])

        return {
            "sample_id": sample_id, "turns": turns, "all_qa": all_qa,
            "user_name": user_name, "assistant_name": assistant_name,
            "short_term": short_term, "mid_term": mid_term, "long_term": long_term,
            "memory_view": memory_view, "vector_store": store,
            "native_instance": None, "mode": "light_native_three_tier",
            "artifact_refs": {"sample_id": sample_id},
        }

    def _llm_chat(self, system_prompt: str, user_text: str, max_tokens: int = 300) -> str:
        if not self.config.api_key:
            return ""
        return llm_chat(
            api_key=self.config.api_key, base_url=self.config.base_url,
            model=self.config.llm_model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_text}],
            temperature=0.0, max_tokens=max_tokens,
        )

    def _answer_with_context(self, query: str, context: str) -> str:
        if not self.config.api_key:
            return ""
        prompt = (
            "Answer based ONLY on the user's memory context. Concise (≤30 words). "
            "If unanswerable, say 'I don't know.'\n\n"
            f"Memory context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        return llm_chat(
            api_key=self.config.api_key, base_url=self.config.base_url,
            model=self.config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=200,
        )

    def _render_context(self, retrieved: List[Dict[str, Any]]) -> str:
        return "\n---\n".join([
            f"[{r.get('meta', {}).get('tier', '?')}] {r.get('text', '')}"
            for r in retrieved if r.get("text")
        ])

    def _vsi_to_record(self, item, score: float) -> Dict[str, Any]:
        return {
            "id": getattr(item, "item_id", ""),
            "text": getattr(item, "text", ""),
            "score": float(score),
            "meta": dict(getattr(item, "meta", {}) or {}),
        }


__all__ = ["MemoryOSAdapter", "MemoryOSAdapterConfig"]
