from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from memory_eval.adapters.base import BaseMemoryAdapter
from memory_eval.eval_core.models import AdapterTrace, RetrievedItem


@dataclass
class GAMAdapterConfig:
    gam_root: str = ""
    api_key: str = ""
    base_url: str = ""
    model: str = "gpt-4o-mini"
    llm_model: str = "gpt-4o-mini"
    keys_path: str = ""
    memory_model: str = ""
    research_model: str = ""
    working_model: str = ""
    memory_dir: str = ""
    top_k: int = 5
    max_iters: int = 3
    use_bm25: bool = False
    use_dense: bool = False


class GAMAdapter(BaseMemoryAdapter):
    family = "gam"

    def __init__(self, config: Optional[GAMAdapterConfig] = None):
        super().__init__()
        cfg = config or GAMAdapterConfig()
        creds = self.merge_runtime_credentials(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            model=cfg.llm_model or cfg.model,
            keys_path=cfg.keys_path,
            require_complete=False,
        )
        self.config = GAMAdapterConfig(
            gam_root=cfg.gam_root,
            api_key=creds["api_key"],
            base_url=creds["base_url"],
            model=cfg.model or creds["model"] or "gpt-4o-mini",
            llm_model=creds["model"] or cfg.llm_model or cfg.model or "gpt-4o-mini",
            keys_path=creds["keys_path"],
            memory_model=cfg.memory_model,
            research_model=cfg.research_model,
            working_model=cfg.working_model,
            memory_dir=cfg.memory_dir,
            top_k=cfg.top_k,
            max_iters=cfg.max_iters,
            use_bm25=cfg.use_bm25,
            use_dense=cfg.use_dense,
        )

    def capabilities(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "flavor": "stable_eval",
            "supports_build_manifest": False,
            "supports_full_memory_export": True,
            "supports_original_retrieval": True,
            "supports_online_answer": True,
            "supports_oracle_answer": True,
            "supports_memwiki_aux_ingest": True,
            "memwiki_aux_strategy": "inject_pages_and_rebuild_retrievers",
        }

    def runtime_manifest(self) -> Dict[str, Any]:
        return {"family": self.family, "flavor": "stable_eval", "capabilities": self.capabilities()}

    def ingest_conversation(self, sample_id: str, conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
        turns = self.normalize_turns(conversation)
        deps = self._load_deps()
        run_dir = self._make_run_dir(sample_id)
        memory_store = deps["InMemoryMemoryStore"](dir_path=str(run_dir))
        page_store = deps["InMemoryPageStore"](dir_path=str(run_dir))
        memory_generator = self._make_generator(deps, role="memory")
        research_generator = self._make_generator(deps, role="research")
        working_generator = self._make_generator(deps, role="working")
        memory_agent = deps["MemoryAgent"](
            memory_store=memory_store,
            page_store=page_store,
            generator=memory_generator,
            dir_path=str(run_dir),
        )
        for turn in turns:
            memory_agent.memorize(self._render_turn(turn))
        retrievers = self._build_retrievers(deps, page_store, run_dir)
        research_agent = deps["ResearchAgent"](
            page_store=page_store,
            memory_store=memory_store,
            retrievers=retrievers,
            generator=research_generator,
            max_iters=int(self.config.max_iters or 3),
            dir_path=str(run_dir),
        )
        return {
            "sample_id": sample_id,
            "turns": turns,
            "run_dir": str(run_dir),
            "memory_store": memory_store,
            "page_store": page_store,
            "retrievers": retrievers,
            "memory_agent": memory_agent,
            "research_agent": research_agent,
            "working_generator": working_generator,
            "artifact_refs": {
                "run_dir": str(run_dir),
                "memory_state_file": str(run_dir / "memory_state.json"),
                "pages_file": str(run_dir / "pages.json"),
            },
        }

    def export_full_memory(self, run_ctx: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        memory_state = run_ctx["memory_store"].load()
        for idx, abstract in enumerate(getattr(memory_state, "abstracts", []) or []):
            out.append(
                {
                    "id": f"gam-memory-{idx}",
                    "text": str(abstract),
                    "meta": {"source": "gam_memory_store", "memory_index": idx},
                }
            )
        for idx, page in enumerate(run_ctx["page_store"].load()):
            text = f"{getattr(page, 'header', '')}\n{getattr(page, 'content', '')}".strip()
            out.append(
                {
                    "id": f"gam-page-{idx}",
                    "text": text,
                    "meta": {"source": "gam_page_store", "page_index": idx, "page_meta": getattr(page, "meta", {})},
                }
            )
        return self.append_aux_records(out, run_ctx)

    def find_memory_records(
        self,
        run_ctx: Any,
        query: str,
        f_key: List[str],
        memory_corpus: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        terms = {token.lower() for token in str(query or "").split() if token}
        for key in f_key or []:
            terms.update(token.lower() for token in str(key or "").split() if token)
        results: List[tuple[int, Dict[str, Any]]] = []
        for record in memory_corpus or []:
            text = str(record.get("text", "")).strip()
            score = sum(1 for token in terms if token and token in text.lower())
            if score > 0:
                results.append((score, record))
        results.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in results[:100]]

    def hybrid_retrieve_candidates(
        self,
        run_ctx: Any,
        query: str,
        f_key: List[str],
        evidence_texts: List[str],
        top_n: int = 100,
    ) -> List[Dict[str, Any]]:
        combined_query = " ".join([query] + list(f_key or []) + list(evidence_texts or [])).strip()
        retrieved = self.retrieve_original(run_ctx, combined_query or query, top_k=min(max(top_n, 1), 20))
        if retrieved:
            return retrieved[:top_n]
        return self.find_memory_records(run_ctx, query, f_key, self.export_full_memory(run_ctx))[:top_n]

    def retrieve_original(self, run_ctx: Any, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        retrievers = dict(run_ctx.get("retrievers") or {})
        for retriever_name, retriever in retrievers.items():
            try:
                results = retriever.search([query], top_k=top_k) or []
            except Exception:
                continue
            for rank, hit in enumerate(results[0] if results else []):
                items.append(
                    {
                        "id": f"{retriever_name}-{getattr(hit, 'page_id', rank)}",
                        "text": str(getattr(hit, "snippet", "")),
                        "score": float((getattr(hit, "meta", {}) or {}).get("score", max(top_k - rank, 1))),
                        "meta": {
                            "source": "gam_native_retrieval",
                            "retriever_type": retriever_name,
                            "native_rank": rank,
                            "page_id": getattr(hit, "page_id", None),
                            "hit_meta": dict(getattr(hit, "meta", {}) or {}),
                        },
                    }
                )
        if items:
            return items[:top_k]
        return self._fallback_page_retrieval(run_ctx, query=query, top_k=top_k)

    def generate_online_answer(self, run_ctx: Any, question: str, top_k: int = 5) -> str:
        research_output = run_ctx["research_agent"].research(question)
        summary = str(getattr(research_output, "integrated_memory", "") or "").strip()
        prompt = self._make_answer_prompt(question=question, summary=summary)
        raw = run_ctx["working_generator"].generate_single(prompt=prompt)
        return str(raw.get("text", "")).strip()

    def generate_oracle_answer(self, run_ctx: Any, question: str, oracle_context: str) -> str:
        prompt = self._make_answer_prompt(question=question, summary=str(oracle_context or "").strip())
        raw = run_ctx["working_generator"].generate_single(prompt=prompt)
        return str(raw.get("text", "")).strip()

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
        online_answer = self.generate_online_answer(run_ctx, query, top_k=top_k)
        oracle_answer = self.generate_oracle_answer(run_ctx, query, oracle_context)
        return AdapterTrace(
            memory_view=memory_view,
            retrieved_items=retrieved,
            answer_online=online_answer,
            answer_oracle=oracle_answer,
            raw_trace={
                "memory_system": self.family,
                "run_dir": run_ctx.get("run_dir", ""),
                "memory_count": len(memory_view),
                "retrieved_count": len(retrieved),
            },
        )

    def export_build_artifact(self, run_ctx: Any) -> Dict[str, Any]:
        return {
            "sample_id": str(run_ctx.get("sample_id", "")),
            "run_dir": str(run_ctx.get("run_dir", "")),
            "artifact_refs": dict(run_ctx.get("artifact_refs", {})),
        }

    def load_build_artifact(self, manifest: Dict[str, Any]) -> Any:
        raise RuntimeError("GAMAdapter 暂未实现 build artifact 恢复，请直接使用 baseline/eval on-the-fly ingest。")

    def ingest_aux_records(self, run_ctx: Any, aux_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        records = super().ingest_aux_records(run_ctx, aux_records)
        if not isinstance(run_ctx, dict) or "page_store" not in run_ctx:
            return records
        deps = self._load_deps()
        Page = deps["Page"]
        page_store = run_ctx["page_store"]
        existing_headers = {
            str(getattr(page, "header", ""))
            for page in page_store.load()
        }
        added = 0
        for record in records:
            rid = str(record.get("id", "")).strip()
            header = f"MemWiki Aux Index: {rid}"
            if header in existing_headers:
                continue
            text = str(record.get("text", "")).strip()
            if not text:
                continue
            page_store.add(
                Page(
                    header=header,
                    content=text,
                    meta={
                        "source": "memwiki_aux",
                        "adapter_injected": True,
                        **(dict(record.get("meta", {})) if isinstance(record.get("meta", {}), dict) else {}),
                    },
                )
            )
            existing_headers.add(header)
            added += 1
        if added:
            for retriever in dict(run_ctx.get("retrievers") or {}).values():
                update_fn = getattr(retriever, "update", None)
                build_fn = getattr(retriever, "build", None)
                try:
                    if callable(update_fn):
                        update_fn(page_store)
                    elif callable(build_fn):
                        build_fn(page_store)
                except Exception:
                    continue
            run_ctx["memwiki_aux_native_indexed"] = int(run_ctx.get("memwiki_aux_native_indexed", 0) or 0) + added
        return records

    def _load_deps(self) -> Dict[str, Any]:
        gam_root = Path(self.config.gam_root or Path(__file__).resolve().parents[3] / "system" / "general-agentic-memory-main")
        research_root = gam_root / "research"
        if str(research_root) not in sys.path:
            sys.path.insert(0, str(research_root))
        from gam_research.agents.memory_agent import MemoryAgent
        from gam_research.agents.research_agent import ResearchAgent
        from gam_research.config.generator import OpenAIGeneratorConfig
        from gam_research.generator.openai_generator import OpenAIGenerator
        from gam_research.retriever.index_retriever import IndexRetriever
        from gam_research.schemas.memory import InMemoryMemoryStore
        from gam_research.schemas.page import InMemoryPageStore, Page
        from gam_research.schemas.search import Hit
        BM25Retriever = None
        if self.config.use_bm25:
            try:
                from gam_research.retriever.bm25 import BM25Retriever as _BM25Retriever
                BM25Retriever = _BM25Retriever
            except Exception:
                BM25Retriever = None
        return {
            "MemoryAgent": MemoryAgent,
            "ResearchAgent": ResearchAgent,
            "OpenAIGenerator": OpenAIGenerator,
            "OpenAIGeneratorConfig": OpenAIGeneratorConfig,
            "BM25Retriever": BM25Retriever,
            "IndexRetriever": IndexRetriever,
            "InMemoryMemoryStore": InMemoryMemoryStore,
            "InMemoryPageStore": InMemoryPageStore,
            "Hit": Hit,
            "Page": Page,
        }

    def _make_generator(self, deps: Dict[str, Any], *, role: str):
        model_name = getattr(self.config, f"{role}_model", "") or self.config.llm_model or self.config.model
        cfg = deps["OpenAIGeneratorConfig"](
            model_name=str(model_name or "gpt-4o-mini"),
            api_key=str(self.config.api_key or ""),
            base_url=str(self.config.base_url or ""),
            temperature=0.0,
            max_tokens=300,
        )
        return deps["OpenAIGenerator"].from_config(cfg)

    def _make_run_dir(self, sample_id: str) -> Path:
        base = Path(self.config.memory_dir) if self.config.memory_dir else Path(tempfile.mkdtemp(prefix="gam_adapter_"))
        run_dir = base / str(sample_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _build_retrievers(self, deps: Dict[str, Any], page_store: Any, run_dir: Path) -> Dict[str, Any]:
        retrievers: Dict[str, Any] = {"page_index": deps["IndexRetriever"]({"index_dir": str(run_dir / "index_page")})}
        retrievers["page_index"].build(page_store)
        keyword = _SimpleKeywordRetriever(hit_cls=deps["Hit"])
        keyword.build(page_store)
        retrievers["keyword"] = keyword
        if self.config.use_bm25 and deps.get("BM25Retriever") is not None:
            try:
                bm25 = deps["BM25Retriever"]({"index_dir": str(run_dir / "index_bm25"), "threads": 1})
                bm25.build(page_store)
                retrievers["keyword"] = bm25
            except Exception:
                pass
        if self.config.use_dense:
            # Dense retriever is intentionally disabled by default because some
            # FlagEmbedding/flash-attn builds crash at import time in this environment.
            pass
        return retrievers

    def _render_turn(self, turn: Dict[str, Any]) -> str:
        speaker = str(turn.get("speaker", "unknown")).strip()
        when = str(turn.get("time", "")).strip()
        text = str(turn.get("text", "")).strip()
        if when:
            return f"[{when}] {speaker}: {text}"
        return f"{speaker}: {text}"

    def _fallback_page_retrieval(self, run_ctx: Any, *, query: str, top_k: int) -> List[Dict[str, Any]]:
        scored: List[tuple[int, int, str]] = []
        q_terms = {token for token in str(query or "").lower().split() if token}
        for idx, page in enumerate(run_ctx["page_store"].load()):
            text = f"{getattr(page, 'header', '')}\n{getattr(page, 'content', '')}".strip()
            score = sum(1 for token in q_terms if token in text.lower())
            if score > 0:
                scored.append((score, idx, text))
        scored.sort(reverse=True)
        return [
            {
                "id": f"fallback-{idx}",
                "text": text,
                "score": float(score),
                "meta": {"source": "gam_fallback_retrieval", "native_rank": rank, "page_index": idx},
            }
            for rank, (score, idx, text) in enumerate(scored[:top_k])
        ]

    def _make_answer_prompt(self, *, question: str, summary: str) -> str:
        return (
            "Based on the summary below, write an answer in the form of a short phrase for the following question, not a sentence.\n"
            "The answer should be brief and grounded in the summary.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"SUMMARY:\n{summary}\n\n"
            "Short answer:\n"
        )


class _SimpleKeywordRetriever:
    name = "keyword"

    def __init__(self, hit_cls: Any):
        self._hit_cls = hit_cls
        self._pages: List[Any] = []

    def build(self, page_store) -> None:
        self._pages = list(page_store.load())

    def update(self, page_store) -> None:
        self.build(page_store)

    def search(self, query_list: List[str], top_k: int = 10) -> List[List[Any]]:
        out: List[List[Any]] = []
        for query in query_list:
            q_terms = {token for token in str(query or "").lower().split() if token}
            scored: List[tuple[int, int, str]] = []
            for idx, page in enumerate(self._pages):
                text = f"{getattr(page, 'header', '')}\n{getattr(page, 'content', '')}".strip()
                score = sum(1 for token in q_terms if token in text.lower())
                if score > 0:
                    scored.append((score, idx, text))
            scored.sort(reverse=True)
            hits = [
                self._hit_cls(page_id=str(idx), snippet=text, source="keyword", meta={"rank": rank, "score": float(score)})
                for rank, (score, idx, text) in enumerate(scored[:top_k])
            ]
            out.append(hits)
        return out
