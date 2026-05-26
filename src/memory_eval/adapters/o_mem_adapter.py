from __future__ import annotations

import asyncio
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from memory_eval.adapters.base import BaseMemoryAdapter, load_runtime_credentials
from memory_eval.eval_core.models import AdapterTrace, RetrievedItem
from memory_eval.eval_core.utils import normalize_text, split_tokens, text_match


@dataclass(frozen=True)
class OMemAdapterConfig:
    use_real_omem: bool = False
    api_key: str = ""
    base_url: str = ""
    llm_model: str = "gpt-4o-mini"
    keys_path: str = ""
    embedding_model_name: str = "all-MiniLM-L6-v2"
    memory_dir: str = "outputs/omem_memory"
    retrieval_pieces: int = 15
    retrieval_drop_threshold: float = 0.1
    working_memory_max_size: int = 20
    episodic_memory_refresh_rate: int = 5
    omem_root: str = ""
    allow_fallback_lightweight: bool = False
    async_call_timeout_sec: float = 180.0
    device: str = ""
    auto_select_cuda: bool = True

class OMemAdapter(BaseMemoryAdapter):
    family = "o_mem"

    def __init__(self, config: Optional[OMemAdapterConfig] = None):
        super().__init__()
        cfg = config or OMemAdapterConfig()
        creds = self.merge_runtime_credentials(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            model=cfg.llm_model,
            keys_path=cfg.keys_path,
            require_complete=False,
        )
        self.config = OMemAdapterConfig(
            use_real_omem=cfg.use_real_omem,
            api_key=creds["api_key"],
            base_url=creds["base_url"],
            llm_model=creds["model"] or cfg.llm_model,
            keys_path=creds["keys_path"],
            embedding_model_name=cfg.embedding_model_name,
            memory_dir=cfg.memory_dir,
            retrieval_pieces=cfg.retrieval_pieces,
            retrieval_drop_threshold=cfg.retrieval_drop_threshold,
            working_memory_max_size=cfg.working_memory_max_size,
            episodic_memory_refresh_rate=cfg.episodic_memory_refresh_rate,
            omem_root=cfg.omem_root,
            allow_fallback_lightweight=cfg.allow_fallback_lightweight,
            async_call_timeout_sec=cfg.async_call_timeout_sec,
            device=cfg.device,
            auto_select_cuda=cfg.auto_select_cuda,
        )
        self._omem_root = self._resolve_omem_root(self.config.omem_root)

    def ingest_conversation(self, sample_id: str, conversation: List[Dict[str, Any]]) -> Any:
        turns = self.normalize_turns(conversation)
        if self.config.use_real_omem:
            try:
                return self._ingest_real(sample_id, turns)
            except Exception as exc:
                if not self.config.allow_fallback_lightweight:
                    raise RuntimeError(f"real O-Mem ingest failed: {exc}") from exc
                return {
                    "sample_id": sample_id,
                    "conversation": turns,
                    "memory_view": self._build_memory_from_turns(turns),
                    "mode": "fallback_lightweight",
                    "error": str(exc),
                }
        return {
            "sample_id": sample_id,
            "conversation": turns,
            "memory_view": self._build_memory_from_turns(turns),
            "mode": "lightweight",
        }

    def capabilities(self) -> Dict[str, Any]:
        out = super().capabilities()
        out.update(
            {
                "flavor": "stable_eval" if "stableeval" in str(self._omem_root).lower() else "original",
                "supports_full_memory_export": True,
                "supports_native_retrieval": True,
                "supports_oracle_generation": True,
                "supports_online_generation": True,
                "supports_high_recall_candidates": True,
                "supports_real_native_runtime": True,
                "supports_lightweight_fallback": True,
                "memwiki_aux_strategy": (
                    "inject_native_topic_cache" if self.config.use_real_omem else "append_to_runtime_context"
                ),
            }
        )
        return out

    def ingest_aux_records(self, run_ctx: Any, aux_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        records = super().ingest_aux_records(run_ctx, aux_records)
        if not self.config.use_real_omem or not isinstance(run_ctx, dict):
            return records
        memory_system = run_ctx.get("memory_system")
        if memory_system is None:
            return records

        added = 0
        base_index = 900000
        for idx, record in enumerate(records):
            text = str(record.get("text", "")).strip()
            if not text:
                continue
            timestamp = str((record.get("meta", {}) or {}).get("timestamp", "")).strip() or "MEMWIKI_AUX_TIME"
            topic = self._memwiki_aux_topic(record, idx)
            payload = {
                "index": base_index + idx,
                "topics": topic,
                "raw_message": text,
                "timestamp": timestamp,
                "memwiki_aux": True,
            }
            for role_prefix in ("user", "agent"):
                episodic = getattr(memory_system, f"{role_prefix}_episodic_memory", None)
                topic_dict = getattr(memory_system, f"{role_prefix}_topic_message_dict", None)
                if episodic is not None and hasattr(episodic, "episodic_memory_cache_list"):
                    cache_list = getattr(episodic, "episodic_memory_cache_list")
                    if isinstance(cache_list, list) and not any(
                        bool(item.get("memwiki_aux")) and str(item.get("topics")) == topic
                        for item in cache_list
                        if isinstance(item, dict)
                    ):
                        cache_list.append(dict(payload))
                if isinstance(topic_dict, dict):
                    topic_dict[topic] = [text, timestamp]
            added += 1

        if added:
            detail_fn = getattr(memory_system, "generate_memory_detail_map", None)
            try:
                if callable(detail_fn):
                    detail_fn()
            except Exception:
                pass
            run_ctx.pop("_native_retrieval_cache", None)
            run_ctx["memwiki_aux_native_indexed"] = int(run_ctx.get("memwiki_aux_native_indexed", 0) or 0) + added
        return records

    def build_trace_for_query(self, run_ctx: Any, query: str, oracle_context: str, top_k: int) -> AdapterTrace:
        memory_view = self.export_full_memory(run_ctx)
        raw_items = self.retrieve_original(run_ctx, query, top_k)
        retrieved_items = [
            RetrievedItem(
                id=str(item.get("id", "")),
                text=str(item.get("text", "")),
                score=float(item.get("score", 0.0) or 0.0),
                meta=dict(item.get("meta", {})) if isinstance(item.get("meta", {}), dict) else {},
            )
            for item in raw_items
        ]
        answer_oracle = self.generate_oracle_answer(run_ctx, query, oracle_context)
        answer_online = self._generate_online_answer(run_ctx, query, top_k)
        return AdapterTrace(
            memory_view=memory_view,
            retrieved_items=retrieved_items,
            answer_online=answer_online,
            answer_oracle=answer_oracle,
            raw_trace={
                "mode": str(run_ctx.get("mode", "")) if isinstance(run_ctx, dict) else "",
                "score_strategy": "lexical_overlap_fallback",
                "memory_count": len(memory_view),
                "retrieved_count": len(retrieved_items),
                "run_error": str(run_ctx.get("error", "")) if isinstance(run_ctx, dict) else "",
            },
        )

    def export_full_memory(self, run_ctx: Any) -> List[Dict[str, Any]]:
        raw = run_ctx.get("memory_view", []) if isinstance(run_ctx, dict) else []
        out: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw):
            meta = dict(item.get("meta", {})) if isinstance(item.get("meta", {}), dict) else {}
            # Prefer raw text body to avoid duplicated "time|speaker:" prefix.
            # 优先使用 raw_text，避免重复拼接时间和说话人前缀。
            candidate = str(meta.get("raw_text", "")).strip() or str(item.get("text", "")).strip()
            parsed = self._parse_structured_text(candidate)
            body = parsed.get("body", "").strip() or candidate
            text = self._format_memory_text(
                body,
                timestamp=str(meta.get("timestamp", "")).strip() or parsed.get("timestamp", "").strip(),
                speaker=str(meta.get("speaker", "")).strip() or parsed.get("speaker", "").strip(),
                role=str(meta.get("role", "")),
            )
            if not text:
                continue
            out.append(
                {
                    "id": str(item.get("id", f"m-{idx}")),
                    "text": text,
                    "meta": meta,
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
        query_tokens = set(split_tokens(query))
        fact_variants = [self._fact_variants(fact) for fact in f_key if str(fact or "").strip()]
        for variants in fact_variants:
            for value in variants:
                query_tokens.update(split_tokens(value))
        query_tokens = {t for t in query_tokens if t}
        matches: List[Dict[str, Any]] = []
        for item in memory_corpus:
            text = str(item.get("text", ""))
            if not text:
                continue
            meta = dict(item.get("meta", {})) if isinstance(item.get("meta", {}), dict) else {}
            item_variants = self._record_variants(text, meta)
            has_fact_match = False
            for variants in fact_variants:
                if any(any(text_match(vf, vi) for vi in item_variants) for vf in variants):
                    has_fact_match = True
                    break
            if has_fact_match:
                matches.append(item)
                continue
            text_tokens = set(split_tokens(text))
            overlap = len(query_tokens & text_tokens) if query_tokens and text_tokens else 0
            if overlap >= 2:
                matches.append(item)
        return matches

    def hybrid_retrieve_candidates(
        self,
        run_ctx: Any,
        query: str,
        f_key: List[str],
        evidence_texts: List[str],
        top_n: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid candidate retrieval for encoding judgement.
        编码层混合候选检索：关键词匹配 + 语义近似（当前用词重叠近似）。
        """
        memory = self.export_full_memory(run_ctx)
        signals = [query] + list(f_key or []) + list(evidence_texts or [])
        signal_tokens = set()
        for s in signals:
            signal_tokens.update(split_tokens(str(s)))

        scored: List[Dict[str, Any]] = []
        for item in memory:
            text = str(item.get("text", ""))
            txt_tokens = set(split_tokens(text))
            overlap = len(signal_tokens & txt_tokens) if signal_tokens and txt_tokens else 0
            denom = len(signal_tokens) or 1
            keyword_score = overlap / denom
            # Placeholder semantic score: normalized token overlap variant.
            semantic_score = overlap / (len(txt_tokens) or 1)
            fusion_score = 0.6 * semantic_score + 0.4 * keyword_score
            rec = dict(item)
            meta = dict(rec.get("meta", {})) if isinstance(rec.get("meta", {}), dict) else {}
            meta["hybrid_scores"] = {
                "semantic_score": float(semantic_score),
                "keyword_score": float(keyword_score),
                "fusion_score": float(fusion_score),
            }
            rec["meta"] = meta
            rec["_fusion_score"] = fusion_score
            scored.append(rec)

        scored.sort(key=lambda x: float(x.get("_fusion_score", 0.0)), reverse=True)
        out = []
        for rec in scored[: max(1, int(top_n or 1))]:
            rec.pop("_fusion_score", None)
            out.append(rec)
        return out

    def retrieve_original(self, run_ctx: Any, query: str, top_k: int) -> List[Dict[str, Any]]:
        # Real O-Mem path: return native retrieval payload that is fed to generation.
        # 真实 O-Mem 路径：返回原生检索结果（用于后续生成）。
        if self.config.use_real_omem and isinstance(run_ctx, dict):
            retrieval_result, speaker_a, speaker_b = self._native_retrieve(run_ctx, query, top_k)
            items: List[Dict[str, Any]] = []
            rank = 0
            for msg in retrieval_result.get("retrieved context messages", []) or []:
                raw_text, timestamp = self._normalize_retrieved_message(msg)
                text = self._format_memory_text(raw_text, timestamp=timestamp, speaker=speaker_a, role="user")
                if not text:
                    continue
                items.append(
                    {
                        "id": f"ctx-{rank}",
                        "text": text,
                        "score": float(max(0.0, 1.0 - 0.01 * rank)),
                        "meta": {
                            "source": "omem_native_retrieval",
                            "channel": "retrieved_context_messages",
                            "native_rank": rank,
                            "speaker_a": speaker_a,
                            "speaker_b": speaker_b,
                        },
                    }
                )
                rank += 1

            # Keep persona channels in C_original because O-Mem also injects them into generation prompt.
            for i, attr in enumerate(retrieval_result.get("persona attributes", []) or []):
                at = str(attr).strip()
                if not at:
                    continue
                items.append(
                    {
                        "id": f"attr-{i}",
                        "text": at,
                        "score": 0.3,
                        "meta": {"source": "omem_native_retrieval", "channel": "persona_attributes", "native_rank": i},
                    }
                )
            for i, fact in enumerate(retrieval_result.get("persona facts", []) or []):
                ft = str(fact).strip()
                if not ft:
                    continue
                items.append(
                    {
                        "id": f"fact-{i}",
                        "text": ft,
                        "score": 0.35,
                        "meta": {"source": "omem_native_retrieval", "channel": "persona_facts", "native_rank": i},
                    }
                )
            if items:
                return items[: max(1, int(top_k or 1))]

        # Lightweight fallback path (non-real mode only).
        memory = self.export_full_memory(run_ctx)
        query_tokens = set(split_tokens(query))
        scored: List[Dict[str, Any]] = []
        for idx, item in enumerate(memory):
            text = str(item.get("text", ""))
            text_tokens = set(split_tokens(text))
            overlap = len(query_tokens & text_tokens) if query_tokens and text_tokens else 0
            denom = len(query_tokens) or 1
            score = overlap / denom
            scored.append(
                {
                    "id": str(item.get("id", f"r-{idx}")),
                    "text": text,
                    "score": float(score),
                    "meta": {
                        "score_source": "lexical_overlap_fallback",
                        "memory_meta": dict(item.get("meta", {})) if isinstance(item.get("meta", {}), dict) else {},
                    },
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        k = max(1, int(top_k or 1))
        return scored[:k]

    def generate_oracle_answer(self, run_ctx: Any, query: str, oracle_context: str) -> str:
        if self.config.use_real_omem and isinstance(run_ctx, dict):
            manager = run_ctx.get("memory_manager")
            if manager is not None:
                answer = self._run_awaitable(
                    manager.generate_system_response(
                        query=query,
                        restrieval_result={
                            "persona attributes": [],
                            "persona facts": [],
                            "retrieved context messages": self._oracle_context_to_messages(oracle_context),
                        },
                        client=run_ctx.get("client"),
                        speaker_a=run_ctx.get("user_name", "User"),
                        speaker_b=run_ctx.get("agent_name", "Assistant"),
                        llm_model=self.config.llm_model,
                    ),
                    timeout_sec=self.config.async_call_timeout_sec,
                )
                if isinstance(answer, tuple) and answer:
                    return str(answer[0]).strip()
        return self._oracle_fallback_answer(oracle_context, query)

    def generate_online_answer(self, run_ctx: Any, query: str, top_k: int = 5) -> str:
        return self._generate_online_answer(run_ctx, query, top_k)

    def _generate_online_answer(self, run_ctx: Any, query: str, top_k: int) -> str:
        if self.config.use_real_omem and isinstance(run_ctx, dict):
            retrieval_result, speaker_a, speaker_b = self._native_retrieve(run_ctx, query, top_k)
            manager = run_ctx.get("memory_manager")
            client = run_ctx.get("client")
            if manager is None or client is None:
                raise RuntimeError("real O-Mem run_ctx missing memory_manager/client for online generation")
            answer = self._run_awaitable(
                manager.generate_system_response(
                    query=query,
                    restrieval_result=retrieval_result,
                    client=client,
                    speaker_a=speaker_a,
                    speaker_b=speaker_b,
                    llm_model=self.config.llm_model,
                ),
                timeout_sec=self.config.async_call_timeout_sec,
            )
            if isinstance(answer, tuple) and answer:
                out = str(answer[0]).strip()
                if out:
                    return out
            raise RuntimeError("real O-Mem online generation returned empty answer")

        items = self.retrieve_original(run_ctx, query, top_k)
        if not items:
            return "I don't know"
        best = str(items[0].get("text", "")).strip()
        if not best:
            return "I don't know"
        return best

    def _native_retrieve(self, run_ctx: Dict[str, Any], query: str, top_k: int) -> Tuple[Dict[str, Any], str, str]:
        manager = run_ctx.get("memory_manager")
        if manager is None:
            raise RuntimeError("real O-Mem run_ctx missing memory_manager for native retrieval")

        cache = run_ctx.setdefault("_native_retrieval_cache", {})
        cache_key = f"{query}||{int(top_k or 1)}"
        cached = cache.get(cache_key)
        if isinstance(cached, dict):
            rr = cached.get("retrieval_result")
            sa = str(cached.get("speaker_a", run_ctx.get("user_name", "User")))
            sb = str(cached.get("speaker_b", run_ctx.get("agent_name", "Assistant")))
            if isinstance(rr, dict):
                return rr, sa, sb

        retrieval_result, speaker_a, speaker_b, _, _ = manager.retrieve_from_memory_soft_segmentation(
            question=query,
            topn=max(1, int(top_k or 1)),
            drop_threshold=float(self.config.retrieval_drop_threshold),
        )
        if not isinstance(retrieval_result, dict):
            raise RuntimeError("native O-Mem retrieval returned invalid payload")
        if "retrieved context messages" not in retrieval_result:
            raise RuntimeError("native O-Mem retrieval payload missing 'retrieved context messages'")

        cache[cache_key] = {
            "retrieval_result": retrieval_result,
            "speaker_a": str(speaker_a or run_ctx.get("user_name", "User")),
            "speaker_b": str(speaker_b or run_ctx.get("agent_name", "Assistant")),
        }
        return retrieval_result, str(speaker_a or run_ctx.get("user_name", "User")), str(
            speaker_b or run_ctx.get("agent_name", "Assistant")
        )

    def _normalize_retrieved_message(self, msg: Any) -> Tuple[str, str]:
        if isinstance(msg, (list, tuple)):
            raw_text = str(msg[0]).strip() if len(msg) > 0 else ""
            timestamp = str(msg[1]).strip() if len(msg) > 1 else ""
            return raw_text, timestamp
        if isinstance(msg, dict):
            raw_text = str(msg.get("raw_message") or msg.get("text") or "").strip()
            timestamp = str(msg.get("timestamp") or msg.get("time") or "").strip()
            return raw_text, timestamp
        return str(msg).strip(), ""

    def _oracle_fallback_answer(self, oracle_context: str, query: str) -> str:
        text = str(oracle_context or "").strip()
        if not text or normalize_text(text) == "no_relevant_memory":
            return "I don't know"
        lines = [x.strip() for x in text.splitlines() if x.strip()]
        if not lines:
            return "I don't know"
        first = lines[0]
        if "|" in first:
            _, right = first.split("|", 1)
            if ":" in right:
                return right.split(":", 1)[1].strip()
            return right.strip()
        if ":" in first:
            parts = first.rsplit(":", 1)
            return parts[1].strip()
        return first

    def _oracle_context_to_messages(self, oracle_context: str) -> List[List[str]]:
        lines = [x.strip() for x in str(oracle_context or "").splitlines() if x.strip()]
        out: List[List[str]] = []
        for line in lines:
            if "|" in line:
                left, right = line.split("|", 1)
                message = right.strip()
            else:
                left = ""
                message = line
            out.append([message, left.strip()])
        return out

    def _build_memory_from_turns(self, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for turn in turns:
            role = "user" if normalize_text(turn.get("speaker", "")) not in {"assistant", "agent", "system"} else "agent"
            speaker = str(turn.get("speaker", ""))
            timestamp = str(turn.get("timestamp", ""))
            raw_text = str(turn.get("text", ""))
            out.append(
                {
                    "id": f"conv-{turn['turn_index']}",
                    "text": self._format_memory_text(raw_text, timestamp=timestamp, speaker=speaker, role=role),
                    "meta": {
                        "layer": "conversation_cache",
                        "turn_index": int(turn.get("turn_index", 0)),
                        "role": role,
                        "speaker": speaker,
                        "timestamp": timestamp,
                        "raw_text": raw_text,
                    },
                }
            )
        return out

    def _resolve_omem_root(self, configured: str) -> Path:
        if configured:
            return Path(configured).resolve()
        return Path(__file__).resolve().parents[3] / "system" / "O-Mem"

    def _select_runtime_device(self, torch_mod: Any) -> str:
        configured = str(self.config.device or "").strip()
        if configured:
            return configured
        if not bool(getattr(torch_mod.cuda, "is_available", lambda: False)()):
            return "cpu"
        if not self.config.auto_select_cuda:
            return "cuda"
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            rows: List[Tuple[int, int, int]] = []
            for line in proc.stdout.splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) != 3:
                    continue
                rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
            if rows:
                rows.sort(key=lambda x: (x[1], -x[2]), reverse=True)
                return f"cuda:{rows[0][0]}"
        except Exception:
            pass
        return "cuda"

    def _ingest_real(self, sample_id: str, turns: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.config.api_key or not self.config.base_url:
            raise ValueError("启用真实 O-Mem 模式时必须提供 api_key 与 base_url。")
        omem_root = self._omem_root
        if not omem_root.exists():
            raise FileNotFoundError(f"O-Mem 目录不存在: {omem_root}")
        if str(omem_root) not in sys.path:
            sys.path.insert(0, str(omem_root))
        from memory_chain import MemoryChain, MemoryManager  # type: ignore
        from openai import AsyncOpenAI  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
        import torch  # type: ignore

        user_name = self.guess_user_name(turns)
        agent_name = self.guess_agent_name(turns, user_name)
        memory_dir = str((Path(self.config.memory_dir) / sample_id).resolve())
        Path(memory_dir).mkdir(parents=True, exist_ok=True)
        client = AsyncOpenAI(base_url=self.config.base_url, api_key=self.config.api_key)
        device = self._select_runtime_device(torch)
        if str(device).startswith("cuda:"):
            torch.cuda.set_device(int(str(device).split(":", 1)[1]))
        embedding_path = Path(str(self.config.embedding_model_name))
        embedding_name = str(embedding_path.resolve()) if embedding_path.exists() else self.config.embedding_model_name
        try:
            embedding_model = SentenceTransformer(embedding_name, device=device)
        except Exception as exc:
            raise RuntimeError(
                f"无法加载 O-Mem embedding 模型: {embedding_name}. "
                "请提供可离线加载的本地模型路径，或修复当前 sentence-transformers/transformers 依赖。"
            ) from exc
        cmd_args = SimpleNamespace(
            working_memory_max_size=int(self.config.working_memory_max_size),
            episodic_memory_refresh_rate=int(self.config.episodic_memory_refresh_rate),
            number_of_retrieval_pieces=max(10, int(self.config.retrieval_pieces)),
            drop_threshold=float(self.config.retrieval_drop_threshold),
            output_dir=memory_dir,
        )
        args = {"model": {"llm_model": self.config.llm_model}}

        memory_system = MemoryChain(
            memory_index=0,
            llm_model=self.config.llm_model,
            llm_client=client,
            embedding_model=embedding_model,
            user_name=user_name,
            agent_name=agent_name,
            cmd_args=cmd_args,
            args=args,
            memory_dir=memory_dir,
        )
        memory_manager = MemoryManager(
            memory_index=0,
            memory_system=memory_system,
            user_name=user_name,
            agent_name=agent_name,
            llm_model=self.config.llm_model,
            llm_client=client,
            cmd_args=cmd_args,
            args=args,
            embedding_model=embedding_model,
            memory_dir=memory_dir,
        )
        self._run_awaitable(self._feed_turns(memory_manager, turns, user_name), timeout_sec=None)
        self._sync_omem_state(memory_system)

        return {
            "sample_id": sample_id,
            "conversation": turns,
            "memory_view": self._collect_omem_memory(memory_system),
            "mode": "real_omem",
            "memory_system": memory_system,
            "memory_manager": memory_manager,
            "client": client,
            "user_name": user_name,
            "agent_name": agent_name,
            "runtime_device": device,
        }

    def _collect_omem_memory(self, memory_system: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        groups = [
            ("user_working", list(memory_system.user_working_memory.working_memory_queue.queue)),
            ("user_episodic", list(memory_system.user_episodic_memory.episodic_memory_cache_list)),
            ("agent_working", list(memory_system.agent_working_memory.working_memory_queue.queue)),
            ("agent_episodic", list(memory_system.agent_episodic_memory.episodic_memory_cache_list)),
        ]
        for layer, items in groups:
            for idx, item in enumerate(items):
                role = "user" if "user" in layer else "agent"
                speaker = self._speaker_from_role(role)
                timestamp = str(item.get("timestamp", ""))
                raw_text = str(item.get("raw_message", ""))
                text = self._format_memory_text(raw_text, timestamp=timestamp, speaker=speaker, role=role)
                if not text:
                    continue
                out.append(
                    {
                        "id": f"{layer}-{idx}-{item.get('index', idx)}",
                        "text": text,
                        "meta": {
                            "layer": layer,
                            "turn_index": int(item.get("index", idx)),
                            "role": role,
                            "speaker": speaker,
                            "timestamp": timestamp,
                            "raw_text": raw_text.strip(),
                        },
                    }
                )
        return out

    async def _feed_turns(self, memory_manager: Any, turns: List[Dict[str, Any]], user_name: str) -> None:
        for idx, turn in enumerate(turns):
            speaker = str(turn.get("speaker", ""))
            user_speak = speaker == user_name
            await asyncio.wait_for(
                memory_manager.receive_message(
                    message=str(turn.get("text", "")),
                    index=int(turn.get("turn_index", idx)),
                    client=memory_manager.client,
                    timestamp=str(turn.get("timestamp", "")),
                    user_speak=user_speak,
                ),
                timeout=float(self.config.async_call_timeout_sec),
            )

    def _sync_omem_state(self, memory_system: Any) -> None:
        memory_system.user_topic_message_dict = {}
        memory_system.agent_topic_message_dict = {}
        for message in list(memory_system.user_working_memory.working_memory_queue.queue) + list(
            memory_system.user_episodic_memory.episodic_memory_cache_list
        ):
            memory_system.user_topic_message_dict[message["topics"]] = [message["raw_message"], message["timestamp"]]
        for message in list(memory_system.agent_working_memory.working_memory_queue.queue) + list(
            memory_system.agent_episodic_memory.episodic_memory_cache_list
        ):
            memory_system.agent_topic_message_dict[message["topics"]] = [message["raw_message"], message["timestamp"]]
        memory_system.generate_memory_detail_map()

    def _run_awaitable(self, awaitable: Any, timeout_sec: float | None = None) -> Any:
        async def _runner() -> Any:
            if timeout_sec:
                return await asyncio.wait_for(awaitable, timeout=float(timeout_sec))
            return await awaitable

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_runner())

        result_box: Dict[str, Any] = {}
        error_box: Dict[str, BaseException] = {}

        def _thread_target() -> None:
            try:
                result_box["value"] = asyncio.run(_runner())
            except BaseException as exc:  # noqa: BLE001
                error_box["error"] = exc

        thread = threading.Thread(target=_thread_target, daemon=False)
        thread.start()
        thread.join()
        if "error" in error_box:
            raise error_box["error"]
        return result_box.get("value")

    def _speaker_from_role(self, role: str) -> str:
        nr = normalize_text(role)
        if nr == "user":
            return "User"
        if nr == "agent":
            return "Assistant"
        return ""

    def _format_memory_text(self, raw_text: str, timestamp: str, speaker: str, role: str) -> str:
        txt = str(raw_text or "").strip()
        if not txt:
            return ""
        ts = str(timestamp or "").strip()
        spk = str(speaker or "").strip() or self._speaker_from_role(role)
        left = ts if ts else "UNKNOWN_TIME"
        right_speaker = spk if spk else "UNKNOWN_SPEAKER"
        return f"{left} | {right_speaker}: {txt}"

    def _parse_structured_text(self, text: str) -> Dict[str, str]:
        s = str(text or "").strip()
        m = re.match(r"^\s*([^|]+?)\s*\|\s*([^:]+?)\s*:\s*(.+?)\s*$", s)
        if m:
            return {"timestamp": m.group(1).strip(), "speaker": m.group(2).strip(), "body": m.group(3).strip()}
        m2 = re.match(r"^\s*([^:]+?)\s*:\s*(.+?)\s*$", s)
        if m2:
            return {"timestamp": "", "speaker": m2.group(1).strip(), "body": m2.group(2).strip()}
        return {"timestamp": "", "speaker": "", "body": s}

    def _fact_variants(self, fact: str) -> List[str]:
        parsed = self._parse_structured_text(fact)
        variants = {
            normalize_text(str(fact or "")),
            normalize_text(parsed["body"]),
        }
        if parsed["speaker"] and parsed["body"]:
            variants.add(normalize_text(f"{parsed['speaker']}: {parsed['body']}"))
        if parsed["timestamp"] and parsed["speaker"] and parsed["body"]:
            variants.add(normalize_text(f"{parsed['timestamp']} | {parsed['speaker']}: {parsed['body']}"))
        return [v for v in variants if v]

    def _record_variants(self, text: str, meta: Dict[str, Any]) -> List[str]:
        parsed = self._parse_structured_text(text)
        role = str(meta.get("role", ""))
        ts = str(meta.get("timestamp", "")).strip() or parsed["timestamp"]
        spk = str(meta.get("speaker", "")).strip() or parsed["speaker"] or self._speaker_from_role(role)
        body = str(meta.get("raw_text", "")).strip() or parsed["body"] or str(text).strip()
        canonical = self._format_memory_text(body, timestamp=ts, speaker=spk, role=role)
        variants = {
            normalize_text(text),
            normalize_text(canonical),
            normalize_text(body),
        }
        if spk and body:
            variants.add(normalize_text(f"{spk}: {body}"))
        if ts and spk and body:
            variants.add(normalize_text(f"{ts} | {spk}: {body}"))
        return [v for v in variants if v]

    def _memwiki_aux_topic(self, record: Dict[str, Any], idx: int) -> str:
        text = str(record.get("text", "")).strip()
        meta = dict(record.get("meta", {})) if isinstance(record.get("meta", {}), dict) else {}
        tags = meta.get("tags", {}) if isinstance(meta.get("tags", {}), dict) else {}
        tag_bits: List[str] = []
        for values in tags.values():
            if isinstance(values, list):
                tag_bits.extend(str(v).strip() for v in values if str(v).strip())
            elif str(values).strip():
                tag_bits.append(str(values).strip())
        prefix = "MemWiki auxiliary retrieval index"
        tag_text = " ".join(tag_bits[:12]).strip()
        body = " ".join([prefix, tag_text, text]).strip()
        return body[:600] or f"MemWiki auxiliary retrieval index {idx}"
