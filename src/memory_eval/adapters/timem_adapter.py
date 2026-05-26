"""
TiMem adapter — Time-hierarchical memory.

TiMem 的核心设计哲学（论文 Li et al. 2026）：
- L1 fragment   ：单条对话片段（原始消息）
- L2 session    ：单 session 内的摘要 + 关键事实抽取
- L3 daily      ：单日总结
- L4 weekly     ：单周总结
- L5 high-level ：跨周持久人物画像 / 偏好

检索路径（论文）：
- Simple    (短问题)：L1 top-5  → 选最佳 session 的 L2
- Hybrid    (中等)：  L1 top-5 + 该 session 的 L3 daily + L4 weekly
- Complex   (复杂)：  L1 top-5 + 该 session 的 L5 monthly profile

本 adapter 实现 TiMem 设计哲学的 light_native 版本：
- 不依赖 TiMem 的 LangGraph + service registry + database 全栈
- L1-L5 层级结构在 in-memory 数据结构中实现
- L2/L3/L4/L5 摘要由 LLM 生成（每 session/day/week/profile 一次）
- 检索复杂度由 query 长度 + 关键词类型简单分类
- Embedding 检索 + 时间桶过滤
- 记忆生成与检索流程严格遵循 TiMem 的"先 L1 后 session 选择，再向上聚合"哲学

诚实标识：runtime_manifest.flavor = "light_native_l1_l5"
"""
from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from memory_eval.adapters.base import BaseMemoryAdapter
from memory_eval.adapters._native_runtime import (
    EmbeddingService,
    InMemoryVectorStore,
    llm_chat,
    parse_timestamp,
    time_bucket,
)
from memory_eval.eval_core.models import AdapterTrace, RetrievedItem

logger = logging.getLogger(__name__)


@dataclass
class TimemAdapterConfig:
    timem_root: str = ""
    api_key: str = ""
    base_url: str = ""
    llm_model: str = "gpt-4o-mini"
    keys_path: str = ""
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    # L2-L5 summary controls
    enable_l2_session_summary: bool = True
    enable_l3_daily_summary: bool = True
    enable_l4_weekly_summary: bool = True
    enable_l5_persistent_profile: bool = True
    summary_max_tokens: int = 500
    profile_max_tokens: int = 800
    # Retrieval params
    l1_top_k: int = 5
    l_higher_top_k: int = 1
    summary_chunk_max_chars: int = 2400


@dataclass
class _TimemMemoryNode:
    node_id: str
    level: str  # L1 / L2 / L3 / L4 / L5
    text: str
    parent_id: Optional[str] = None
    bucket_key: str = ""
    timestamp: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class TimemAdapter(BaseMemoryAdapter):
    """TiMem light-native adapter (L1-L5 in-memory hierarchy)."""

    family = "timem"

    def __init__(self, config: Optional[TimemAdapterConfig] = None) -> None:
        super().__init__()
        cfg = config or TimemAdapterConfig()
        creds = self.merge_runtime_credentials(
            api_key=cfg.api_key, base_url=cfg.base_url,
            model=cfg.llm_model, keys_path=cfg.keys_path,
        )
        self.config = TimemAdapterConfig(
            timem_root=cfg.timem_root or str(Path(__file__).resolve().parents[3] / "system" / "timem-main"),
            api_key=creds["api_key"],
            base_url=creds["base_url"],
            llm_model=creds["model"] or cfg.llm_model or "gpt-4o-mini",
            keys_path=creds["keys_path"],
            embedding_model_name=cfg.embedding_model_name,
            embedding_device=cfg.embedding_device,
            enable_l2_session_summary=cfg.enable_l2_session_summary,
            enable_l3_daily_summary=cfg.enable_l3_daily_summary,
            enable_l4_weekly_summary=cfg.enable_l4_weekly_summary,
            enable_l5_persistent_profile=cfg.enable_l5_persistent_profile,
            summary_max_tokens=cfg.summary_max_tokens,
            profile_max_tokens=cfg.profile_max_tokens,
            l1_top_k=cfg.l1_top_k,
            l_higher_top_k=cfg.l_higher_top_k,
            summary_chunk_max_chars=cfg.summary_chunk_max_chars,
        )
        self._embed = EmbeddingService(
            model_name=self.config.embedding_model_name,
            device=self.config.embedding_device,
        )
        self.flavor = "light_native_l1_l5"

    # --- protocol methods --------------------------------------------------

    def capabilities(self) -> Dict[str, Any]:
        out = super().capabilities()
        out.update(
            {
                "flavor": self.flavor,
                "supports_real_native_runtime": False,
                "supports_lightweight_fallback": True,
                "native_runtime_status": "light_native_l1_l5_in_memory",
                "embedding_mode": self._embed.mode,
                "design_hierarchy": ["L1_fragment", "L2_session", "L3_daily", "L4_weekly", "L5_profile"],
            }
        )
        return out

    def runtime_manifest(self) -> Dict[str, Any]:
        return {"family": self.family, "flavor": self.flavor, "capabilities": self.capabilities()}

    def ingest_conversation(self, sample_id: str, conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
        turns = self.normalize_turns(conversation)
        # L1 fragments: each turn becomes an L1 node
        l1_nodes: List[_TimemMemoryNode] = []
        for idx, turn in enumerate(turns):
            ts = turn.get("timestamp", "")
            session_bucket = time_bucket(parse_timestamp(ts), "session")
            l1_nodes.append(_TimemMemoryNode(
                node_id=f"L1::{sample_id}::{idx}",
                level="L1",
                text=f"{turn.get('speaker', '')}: {turn.get('text', '')}".strip(),
                parent_id=None,
                bucket_key=session_bucket,
                timestamp=ts,
                extra={"turn_index": idx, "speaker": turn.get("speaker", "")},
            ))

        # L2 session summaries: group L1 by session_bucket
        l2_nodes: List[_TimemMemoryNode] = []
        if self.config.enable_l2_session_summary:
            grouped = defaultdict(list)
            for node in l1_nodes:
                grouped[node.bucket_key].append(node)
            for sess_key, members in grouped.items():
                summary = self._summarize(
                    members, level="L2",
                    instruction="Summarize this session into a coherent paragraph that preserves all named entities, facts, and decisions. 80-200 words.",
                )
                l2_node = _TimemMemoryNode(
                    node_id=f"L2::{sample_id}::{sess_key}",
                    level="L2", text=summary,
                    bucket_key=sess_key,
                    timestamp=members[0].timestamp,
                    extra={"member_l1_ids": [n.node_id for n in members]},
                )
                # back-link L1 -> L2
                for n in members:
                    n.parent_id = l2_node.node_id
                l2_nodes.append(l2_node)

        # L3 daily, L4 weekly, L5 high-level — same pattern, different grain
        l3_nodes = self._build_higher_level(
            l2_nodes, sample_id, "L3", "day",
            "Summarize all sessions of this day into a single concise daily report. 80-150 words.",
            enabled=self.config.enable_l3_daily_summary,
        )
        l4_nodes = self._build_higher_level(
            l2_nodes, sample_id, "L4", "week",
            "Summarize all sessions of this week into a weekly trend report. 100-200 words.",
            enabled=self.config.enable_l4_weekly_summary,
        )
        # L5 persistent profile: one node total, summarizing all L4 weekly reports
        l5_nodes: List[_TimemMemoryNode] = []
        if self.config.enable_l5_persistent_profile and l4_nodes:
            full = "\n".join([f"[{n.bucket_key}] {n.text}" for n in l4_nodes])[: self.config.summary_chunk_max_chars]
            persistent_profile = self._llm_summarize(
                full,
                instruction="Distill the user's persistent traits, recurring topics, and stable preferences across all weeks. 100-300 words. Focus on time-invariant facts.",
                max_tokens=self.config.profile_max_tokens,
            )
            l5_nodes.append(_TimemMemoryNode(
                node_id=f"L5::{sample_id}::profile",
                level="L5", text=persistent_profile or "(no profile)",
                bucket_key="profile",
                extra={"derived_from_l4": [n.node_id for n in l4_nodes]},
            ))

        # Build vector store across all levels
        vector_store = InMemoryVectorStore()
        all_nodes = l1_nodes + l2_nodes + l3_nodes + l4_nodes + l5_nodes
        if all_nodes:
            vectors = self._embed.embed([n.text for n in all_nodes])
            for node, vec in zip(all_nodes, vectors):
                vector_store.upsert(
                    item_id=node.node_id, text=node.text, vector=vec,
                    meta={"level": node.level, "bucket": node.bucket_key,
                          "parent_id": node.parent_id, "extra": dict(node.extra),
                          "timestamp": node.timestamp},
                )

        return {
            "sample_id": sample_id,
            "turns": turns,
            "l1_nodes": l1_nodes, "l2_nodes": l2_nodes, "l3_nodes": l3_nodes,
            "l4_nodes": l4_nodes, "l5_nodes": l5_nodes,
            "vector_store": vector_store,
            "memory_view": [self._node_to_record(n) for n in all_nodes],
            "mode": "light_native_l1_l5",
            "artifact_refs": {"sample_id": sample_id},
        }

    def export_full_memory(self, run_ctx: Any) -> List[Dict[str, Any]]:
        if not isinstance(run_ctx, dict):
            return []
        records = list(run_ctx.get("memory_view", []))
        return self.append_aux_records(records, run_ctx)

    def find_memory_records(self, run_ctx: Any, query: str, f_key: List[str], memory_corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        store: InMemoryVectorStore = run_ctx.get("vector_store")
        if not isinstance(store, InMemoryVectorStore) or len(store) == 0:
            return list(memory_corpus or [])[:100]
        signal = " ".join([str(query)] + [str(x) for x in (f_key or [])]).strip()
        if not signal:
            return list(memory_corpus or [])[:100]
        q_vec = self._embed.embed([signal])[0]
        hits = store.search(q_vec, top_k=100)
        return [self._vsi_to_record(item, score) for score, item in hits if score > 0]

    def hybrid_retrieve_candidates(self, run_ctx: Any, query: str, f_key: List[str], evidence_texts: List[str], top_n: int = 100) -> List[Dict[str, Any]]:
        signal = " ".join([str(query)] + [str(x) for x in (f_key or [])] + [str(x) for x in (evidence_texts or [])]).strip()
        return self.find_memory_records(run_ctx, signal or query, f_key, run_ctx.get("memory_view", []))[:top_n]

    def retrieve_original(self, run_ctx: Any, query: str, top_k: int) -> List[Dict[str, Any]]:
        """TiMem-style hierarchical retrieval: L1 → best session → higher levels."""
        store: InMemoryVectorStore = run_ctx.get("vector_store")
        if not isinstance(store, InMemoryVectorStore) or len(store) == 0:
            return []

        complexity = self._estimate_complexity(query)
        q_vec = self._embed.embed([query])[0]

        # Step 1: L1 top-5
        l1_hits = store.search(q_vec, top_k=self.config.l1_top_k,
                               filter_fn=lambda it: it.meta.get("level") == "L1")

        # Step 2: pick best session from L1 hits
        sess_score: Dict[str, float] = defaultdict(float)
        for score, item in l1_hits:
            bucket = item.meta.get("bucket", "")
            if bucket:
                sess_score[bucket] += float(score)
        best_session = max(sess_score.items(), key=lambda x: x[1])[0] if sess_score else None

        # Step 3: gather higher-level summaries for that session
        higher_levels = []
        if best_session is not None:
            for level in (["L2"] if complexity == "simple"
                          else ["L2", "L3", "L4"] if complexity == "hybrid"
                          else ["L2", "L3", "L4", "L5"]):
                if level == "L5":
                    # L5 is global, no bucket filter
                    hits = store.search(q_vec, top_k=self.config.l_higher_top_k,
                                        filter_fn=lambda it, lv=level: it.meta.get("level") == lv)
                else:
                    hits = store.search(
                        q_vec, top_k=self.config.l_higher_top_k,
                        filter_fn=lambda it, lv=level: (
                            it.meta.get("level") == lv and (
                                it.meta.get("bucket") == best_session or
                                lv != "L2"  # L3/L4 may bucket differently; relax to top-k
                            )
                        ),
                    )
                higher_levels.extend(hits)

        # Combine + dedup
        combined: List[Tuple[float, Any]] = list(l1_hits) + list(higher_levels)
        seen_ids: set = set()
        unique: List[Tuple[float, Any]] = []
        for score, item in combined:
            if item.item_id in seen_ids:
                continue
            seen_ids.add(item.item_id)
            unique.append((score, item))
        unique.sort(key=lambda x: x[0], reverse=True)
        return [self._vsi_to_record(item, score) for score, item in unique[: max(1, top_k)]]

    def generate_online_answer(self, run_ctx: Any, query: str, top_k: int = 5) -> str:
        retrieved = self.retrieve_original(run_ctx, query, top_k=top_k)
        context = self._render_context(retrieved)
        return self._answer_with_context(query, context)

    def generate_oracle_answer(self, run_ctx: Any, query: str, oracle_context: str) -> str:
        return self._answer_with_context(query, str(oracle_context or ""))

    def build_trace_for_query(self, run_ctx: Any, query: str, oracle_context: str, top_k: int) -> AdapterTrace:
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
            raw_trace={
                "memory_system": self.family,
                "memory_count": len(memory_view),
                "retrieved_count": len(retrieved),
                "complexity": self._estimate_complexity(query),
            },
        )

    def export_build_artifact(self, run_ctx: Any) -> Dict[str, Any]:
        return {"sample_id": str(run_ctx.get("sample_id", "")),
                "artifact_refs": dict(run_ctx.get("artifact_refs", {}))}

    def load_build_artifact(self, manifest: Dict[str, Any]) -> Any:
        raise RuntimeError("TimemAdapter: build artifact reuse not implemented; please re-ingest.")

    # --- internal helpers --------------------------------------------------

    def _build_higher_level(self, l2_nodes, sample_id, level, grain, instruction, enabled) -> List[_TimemMemoryNode]:
        if not enabled or not l2_nodes:
            return []
        grouped = defaultdict(list)
        for node in l2_nodes:
            ts = parse_timestamp(node.timestamp)
            bucket = time_bucket(ts, grain)
            grouped[bucket].append(node)
        out: List[_TimemMemoryNode] = []
        for bkey, members in grouped.items():
            if len(members) < 1:
                continue
            full = "\n".join([f"[{n.bucket_key}] {n.text}" for n in members])[: self.config.summary_chunk_max_chars]
            summary = self._llm_summarize(
                full, instruction=instruction, max_tokens=self.config.summary_max_tokens,
            ) or full[:300]
            out.append(_TimemMemoryNode(
                node_id=f"{level}::{sample_id}::{bkey}",
                level=level, text=summary, bucket_key=bkey,
                timestamp=members[0].timestamp,
                extra={"derived_from": [n.node_id for n in members]},
            ))
        return out

    def _summarize(self, nodes: List[_TimemMemoryNode], level: str, instruction: str) -> str:
        if not nodes:
            return ""
        text = "\n".join([n.text for n in nodes])[: self.config.summary_chunk_max_chars]
        result = self._llm_summarize(text, instruction=instruction, max_tokens=self.config.summary_max_tokens)
        return result or text[:300]

    def _llm_summarize(self, text: str, instruction: str, max_tokens: int) -> str:
        if not text.strip() or not self.config.api_key:
            return text[:300]
        return llm_chat(
            api_key=self.config.api_key, base_url=self.config.base_url,
            model=self.config.llm_model,
            messages=[{"role": "system", "content": instruction},
                      {"role": "user", "content": text}],
            temperature=0.0, max_tokens=max_tokens,
        ) or text[:300]

    def _answer_with_context(self, query: str, context: str) -> str:
        if not self.config.api_key:
            return ""
        prompt = (
            "You are a helpful assistant answering questions based on the user's long-term memory.\n"
            "Provide a concise answer (≤ 30 words) using ONLY the supplied memory context.\n"
            "If the answer cannot be inferred from the context, reply 'I don't know.'\n\n"
            f"Memory context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        return llm_chat(
            api_key=self.config.api_key, base_url=self.config.base_url,
            model=self.config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=200,
        )

    def _render_context(self, retrieved: List[Dict[str, Any]]) -> str:
        return "\n---\n".join([
            f"[{r.get('meta', {}).get('level', '?')}] {r.get('text', '')}"
            for r in retrieved if r.get("text")
        ])

    def _estimate_complexity(self, query: str) -> str:
        """Light heuristic mirroring TiMem's complexity classifier."""
        q = str(query).lower().strip()
        wcount = len(q.split())
        if wcount <= 6 and not any(kw in q for kw in ("compare", "summarize", "trend", "evolution", "over time", "全部", "对比")):
            return "simple"
        if any(kw in q for kw in ("trend", "evolution", "over time", "对比", "summarize", "总结")):
            return "complex"
        return "hybrid"

    def _node_to_record(self, n: _TimemMemoryNode) -> Dict[str, Any]:
        return {
            "id": n.node_id, "text": n.text,
            "meta": {
                "source": f"timem_{n.level}", "level": n.level,
                "bucket": n.bucket_key, "parent_id": n.parent_id,
                "timestamp": n.timestamp, "extra": dict(n.extra),
            },
        }

    def _vsi_to_record(self, item, score: float) -> Dict[str, Any]:
        return {"id": item.item_id, "text": item.text, "score": float(score), "meta": dict(item.meta)}


__all__ = ["TimemAdapter", "TimemAdapterConfig"]
