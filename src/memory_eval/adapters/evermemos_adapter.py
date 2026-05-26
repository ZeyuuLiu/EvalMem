"""
EverMemOS adapter — light_native implementation of MemCells + BM25/Embedding hybrid.

EverMemOS 的核心设计哲学（论文 Hu et al. 2026，路径见
system/EverOS-main/evaluation/src/adapters/evermemos/）：
- Stage 1: MemCells extraction（LLM 从 conversation 抽出结构化 memory cells）
- Stage 2: Build BM25 + Embedding indexes per conversation
- Stage 3: Memory retrieval（BM25 + Embedding hybrid + RRF）
- Stage 4: Response generation

本 adapter 实现该流程的 light_native 版本：
- MemCell schema：{id, text, type, entities, time}
- 抽取由 LLM 完成；失败时退化为按 turn 切分
- 双索引：SimpleBM25 + InMemoryVectorStore
- 检索：BM25 + Embedding 各取 top-k，RRF 融合
- 生成：用检索到的 cells + question 调 LLM

诚实标识：runtime_manifest.flavor = "light_native_memcells_hybrid"
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from memory_eval.adapters.base import BaseMemoryAdapter
from memory_eval.adapters._native_runtime import (
    EmbeddingService,
    InMemoryVectorStore,
    SimpleBM25,
    llm_chat,
)
from memory_eval.eval_core.models import AdapterTrace, RetrievedItem

logger = logging.getLogger(__name__)


@dataclass
class EverMemOSAdapterConfig:
    everos_root: str = ""
    api_key: str = ""
    base_url: str = ""
    llm_model: str = "gpt-4o-mini"
    keys_path: str = ""
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    # MemCell extraction
    max_cells_per_segment: int = 8
    extraction_segment_size: int = 12       # turns per LLM extraction call
    enable_llm_extraction: bool = True
    # Hybrid retrieval
    bm25_top_k: int = 20
    embedding_top_k: int = 20
    rrf_k: int = 60


class EverMemOSAdapter(BaseMemoryAdapter):
    """EverMemOS light_native adapter (MemCells + BM25/Embedding hybrid)."""

    family = "everos"

    def __init__(self, config: Optional[EverMemOSAdapterConfig] = None) -> None:
        super().__init__()
        cfg = config or EverMemOSAdapterConfig()
        creds = self.merge_runtime_credentials(
            api_key=cfg.api_key, base_url=cfg.base_url,
            model=cfg.llm_model, keys_path=cfg.keys_path,
        )
        root = cfg.everos_root or str(Path(__file__).resolve().parents[3] / "system" / "EverOS-main")
        self.config = EverMemOSAdapterConfig(
            everos_root=root,
            api_key=creds["api_key"], base_url=creds["base_url"],
            llm_model=creds["model"] or cfg.llm_model or "gpt-4o-mini",
            keys_path=creds["keys_path"],
            embedding_model_name=cfg.embedding_model_name,
            embedding_device=cfg.embedding_device,
            max_cells_per_segment=cfg.max_cells_per_segment,
            extraction_segment_size=cfg.extraction_segment_size,
            enable_llm_extraction=cfg.enable_llm_extraction,
            bm25_top_k=cfg.bm25_top_k,
            embedding_top_k=cfg.embedding_top_k,
            rrf_k=cfg.rrf_k,
        )
        self._embed = EmbeddingService(
            model_name=self.config.embedding_model_name,
            device=self.config.embedding_device,
        )
        self.flavor = "light_native_memcells_hybrid"

    # --- protocol methods --------------------------------------------------

    def capabilities(self) -> Dict[str, Any]:
        out = super().capabilities()
        out.update({
            "flavor": self.flavor,
            "supports_real_native_runtime": False,
            "supports_lightweight_fallback": True,
            "native_runtime_status": "light_native_memcells_bm25_embedding",
            "embedding_mode": self._embed.mode,
            "design_stages": ["stage1_memcells", "stage2_dual_index", "stage3_hybrid_retrieve", "stage4_response"],
            "uses_bm25": True,
            "uses_embedding": True,
            "uses_rrf_fusion": True,
        })
        return out

    def runtime_manifest(self) -> Dict[str, Any]:
        return {"family": self.family, "flavor": self.flavor, "capabilities": self.capabilities()}

    def ingest_conversation(self, sample_id: str, conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
        turns = self.normalize_turns(conversation)

        # Stage 1: MemCell extraction
        memcells = self._stage1_extract_memcells(sample_id, turns)

        # Stage 2: build BM25 + Embedding indexes
        bm25 = SimpleBM25()
        for cell in memcells:
            bm25.add(cell["id"], cell["text"], meta={"text": cell["text"], "memcell": cell})

        vector_store = InMemoryVectorStore()
        if memcells:
            vectors = self._embed.embed([c["text"] for c in memcells])
            for cell, vec in zip(memcells, vectors):
                vector_store.upsert(
                    item_id=cell["id"], text=cell["text"], vector=vec,
                    meta={"memcell": cell, "type": cell.get("type", ""),
                          "entities": cell.get("entities", []),
                          "time": cell.get("time", "")},
                )

        memory_view = [
            {"id": c["id"], "text": c["text"],
             "meta": {"source": "evermemos_memcell", "type": c.get("type", ""),
                      "entities": c.get("entities", []), "time": c.get("time", "")}}
            for c in memcells
        ]
        return {
            "sample_id": sample_id, "turns": turns, "memcells": memcells,
            "bm25": bm25, "vector_store": vector_store, "memory_view": memory_view,
            "mode": "light_native_memcells_hybrid",
            "artifact_refs": {"sample_id": sample_id},
        }

    def export_full_memory(self, run_ctx: Any) -> List[Dict[str, Any]]:
        if not isinstance(run_ctx, dict):
            return []
        return self.append_aux_records(list(run_ctx.get("memory_view", [])), run_ctx)

    def find_memory_records(self, run_ctx, query, f_key, memory_corpus) -> List[Dict[str, Any]]:
        # Stage 3 in find mode: hybrid candidate generation
        return self._stage3_hybrid_retrieve(
            run_ctx, query=" ".join([str(query)] + [str(x) for x in (f_key or [])]).strip() or query,
            top_k=100,
        )

    def hybrid_retrieve_candidates(self, run_ctx, query, f_key, evidence_texts, top_n: int = 100) -> List[Dict[str, Any]]:
        signal = " ".join([str(query)] + [str(x) for x in (f_key or [])] + [str(x) for x in (evidence_texts or [])]).strip() or query
        return self._stage3_hybrid_retrieve(run_ctx, signal, top_k=top_n)

    def retrieve_original(self, run_ctx, query, top_k: int) -> List[Dict[str, Any]]:
        return self._stage3_hybrid_retrieve(run_ctx, query, top_k=top_k)

    def generate_online_answer(self, run_ctx, query: str, top_k: int = 5) -> str:
        retrieved = self.retrieve_original(run_ctx, query, top_k=top_k)
        return self._stage4_answer(query, self._render_context(retrieved))

    def generate_oracle_answer(self, run_ctx, query: str, oracle_context: str) -> str:
        return self._stage4_answer(query, str(oracle_context or ""))

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
            raw_trace={"memory_system": self.family, "memcell_count": len(memory_view),
                       "retrieved_count": len(retrieved)},
        )

    def export_build_artifact(self, run_ctx) -> Dict[str, Any]:
        return {"sample_id": str(run_ctx.get("sample_id", "")),
                "memcell_count": len(run_ctx.get("memcells", [])),
                "artifact_refs": dict(run_ctx.get("artifact_refs", {}))}

    def load_build_artifact(self, manifest: Dict[str, Any]) -> Any:
        raise RuntimeError("EverMemOSAdapter: build artifact reuse not implemented; please re-ingest.")

    # --- internal: stage 1 -------------------------------------------------

    def _stage1_extract_memcells(self, sample_id: str, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not turns:
            return []
        if not self.config.enable_llm_extraction or not self.config.api_key:
            return self._fallback_memcells(sample_id, turns)

        all_cells: List[Dict[str, Any]] = []
        seg = self.config.extraction_segment_size
        for seg_idx, start in enumerate(range(0, len(turns), seg)):
            segment = turns[start: start + seg]
            transcript = "\n".join([f"{t.get('speaker', '')} ({t.get('timestamp', '')}): {t.get('text', '')}"
                                    for t in segment])
            prompt = (
                "Extract structured MemCells from the following dialogue segment.\n"
                f"Output a JSON array of up to {self.config.max_cells_per_segment} cells.\n"
                "Each cell must have:\n"
                '  "text": short factual statement (≤30 words)\n'
                '  "type": one of [event, fact, preference, plan, status]\n'
                '  "entities": list of named entities mentioned\n'
                '  "time": optional timestamp string from the dialogue or empty\n'
                "Focus on persistent facts, decisions, named entities, and state changes.\n"
                "Skip trivial chitchat. Output ONLY the JSON array, no markdown.\n\n"
                f"Segment:\n{transcript}\n\nJSON array:"
            )
            try:
                raw = llm_chat(
                    api_key=self.config.api_key, base_url=self.config.base_url,
                    model=self.config.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0, max_tokens=1000,
                )
                cells = self._parse_cells_json(raw)
            except Exception as exc:
                logger.warning(f"EverMemOS extraction failed seg {seg_idx}: {exc}")
                cells = []
            if not cells:
                # Fallback: turn-as-cell for this segment
                for t_idx, t in enumerate(segment):
                    cells.append({
                        "text": f"{t.get('speaker', '')}: {t.get('text', '')}",
                        "type": "event", "entities": [],
                        "time": t.get("timestamp", ""),
                    })
            for c_idx, cell in enumerate(cells):
                cell["id"] = f"memcell::{sample_id}::seg{seg_idx}::{c_idx}"
                cell["segment_index"] = seg_idx
                all_cells.append(cell)
        return all_cells

    def _fallback_memcells(self, sample_id: str, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i, t in enumerate(turns):
            out.append({
                "id": f"memcell::{sample_id}::turn{i}",
                "text": f"{t.get('speaker', '')}: {t.get('text', '')}",
                "type": "event", "entities": [], "time": t.get("timestamp", ""),
                "segment_index": i // self.config.extraction_segment_size,
            })
        return out

    @staticmethod
    def _parse_cells_json(raw: str) -> List[Dict[str, Any]]:
        if not raw:
            return []
        text = raw.strip()
        if text.startswith("```"):
            # strip markdown fences
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"```\s*$", "", text)
        # Find first [ and last ]
        s, e = text.find("["), text.rfind("]")
        if s < 0 or e <= s:
            return []
        try:
            arr = json.loads(text[s:e + 1])
            if isinstance(arr, list):
                out: List[Dict[str, Any]] = []
                for cell in arr:
                    if isinstance(cell, dict) and "text" in cell:
                        out.append({
                            "text": str(cell.get("text", "")).strip(),
                            "type": str(cell.get("type", "fact")),
                            "entities": list(cell.get("entities", []) or []),
                            "time": str(cell.get("time", "")),
                        })
                return out
        except Exception as exc:
            logger.debug(f"_parse_cells_json failed: {exc}")
        return []

    # --- internal: stage 3 -------------------------------------------------

    def _stage3_hybrid_retrieve(self, run_ctx, query: str, top_k: int) -> List[Dict[str, Any]]:
        if not isinstance(run_ctx, dict):
            return []
        bm25: Optional[SimpleBM25] = run_ctx.get("bm25")
        store: Optional[InMemoryVectorStore] = run_ctx.get("vector_store")
        memcells: List[Dict[str, Any]] = list(run_ctx.get("memcells", []))
        if not memcells:
            return []

        # BM25 hits
        bm25_hits: List[Tuple[float, str, Dict[str, Any]]] = []
        if isinstance(bm25, SimpleBM25):
            bm25_hits = bm25.search(query, top_k=self.config.bm25_top_k)

        # Embedding hits
        emb_hits: List[Tuple[float, Any]] = []
        if isinstance(store, InMemoryVectorStore) and len(store) > 0:
            q_vec = self._embed.embed([query])[0]
            emb_hits = store.search(q_vec, top_k=self.config.embedding_top_k)

        # RRF fusion
        rrf_scores: Dict[str, float] = {}
        rrf_meta: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        for rank, (_, doc_id, meta) in enumerate(bm25_hits):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.config.rrf_k + rank + 1)
            rrf_meta.setdefault(doc_id, (str(meta.get("text", "")), {**meta, "source": "bm25"}))
        for rank, (_, item) in enumerate(emb_hits):
            doc_id = item.item_id
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.config.rrf_k + rank + 1)
            rrf_meta.setdefault(doc_id, (item.text, {**item.meta, "source": "embedding"}))

        ranked = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[: max(1, top_k)]
        out: List[Dict[str, Any]] = []
        for doc_id, score in ranked:
            text, meta = rrf_meta.get(doc_id, ("", {}))
            out.append({
                "id": doc_id, "text": text, "score": float(score),
                "meta": {**meta, "rrf_score": float(score), "source_system": "evermemos_hybrid"},
            })
        return out

    # --- internal: stage 4 + helpers --------------------------------------

    def _stage4_answer(self, query: str, context: str) -> str:
        if not self.config.api_key:
            return ""
        prompt = (
            "You are EverMemOS answer engine. Answer concisely (≤30 words) "
            "using ONLY the supplied MemCell context. If unanswerable, say 'I don't know.'\n\n"
            f"MemCells:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        return llm_chat(
            api_key=self.config.api_key, base_url=self.config.base_url,
            model=self.config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=200,
        )

    def _render_context(self, retrieved: List[Dict[str, Any]]) -> str:
        return "\n".join([
            f"- [{r.get('meta', {}).get('type', 'fact')}] {r.get('text', '')}"
            for r in retrieved if r.get("text")
        ])

    def ingest_aux_records(self, run_ctx, aux_records) -> List[Dict[str, Any]]:
        """Override: also push aux records into BM25 + vector indexes (so retrieval sees them)."""
        records = super().ingest_aux_records(run_ctx, aux_records)
        if not isinstance(run_ctx, dict):
            return records
        bm25: SimpleBM25 = run_ctx.get("bm25")
        store: InMemoryVectorStore = run_ctx.get("vector_store")
        new_texts: List[str] = []
        new_ids: List[str] = []
        for rec in records:
            rid = str(rec.get("id", "")).strip() or f"aux-{len(new_ids)}"
            text = str(rec.get("text", "")).strip()
            if not text:
                continue
            new_ids.append(rid)
            new_texts.append(text)
            if isinstance(bm25, SimpleBM25):
                bm25.add(rid, text, meta={"text": text, "memwiki_aux": True})
        if new_texts and isinstance(store, InMemoryVectorStore):
            vectors = self._embed.embed(new_texts)
            for rid, text, vec in zip(new_ids, new_texts, vectors):
                store.upsert(rid, text, vec, meta={"memwiki_aux": True, "source": "memwiki_aux"})
        run_ctx["memwiki_aux_native_indexed"] = int(run_ctx.get("memwiki_aux_native_indexed", 0) or 0) + len(new_ids)
        return records


__all__ = ["EverMemOSAdapter", "EverMemOSAdapterConfig"]
