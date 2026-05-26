from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from memory_eval.eval_core.models import AdapterTrace, RetrievedItem
from memory_eval.eval_core.utils import normalize_text, split_tokens, text_match


@dataclass(frozen=True)
class OMemAdapterConfig:
    use_real_omem: bool = False
    api_key: str = ""
    base_url: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    memory_dir: str = "outputs/omem_memory"
    retrieval_pieces: int = 15
    retrieval_drop_threshold: float = 0.1
    working_memory_max_size: int = 20
    episodic_memory_refresh_rate: int = 5
    omem_root: str = ""


def load_runtime_credentials(keys_path: Optional[str] = None, require_complete: bool = False) -> Dict[str, str]:
    path_api_key = ""
    path_base_url = ""
    path_model = ""
    if keys_path:
        p = Path(keys_path)
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8-sig"))
            path_api_key = str(raw.get("api_key", "")).strip()
            path_base_url = str(raw.get("base_url", "")).strip()
            path_model = str(raw.get("model", "")).strip()

    api_key = (
        os.getenv("MEMORY_EVAL_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
        or path_api_key
    )
    base_url = (
        os.getenv("MEMORY_EVAL_BASE_URL", "").strip()
        or os.getenv("OPENAI_BASE_URL", "").strip()
        or path_base_url
    )
    model = os.getenv("MEMORY_EVAL_MODEL", "").strip() or path_model

    if require_complete and (not api_key or not base_url):
        raise ValueError(
            "缺少 API 凭据：请设置 MEMORY_EVAL_API_KEY/MEMORY_EVAL_BASE_URL（或 OPENAI_API_KEY/OPENAI_BASE_URL），"
            "或提供本地 keys 文件。"
        )
    return {"api_key": api_key, "base_url": base_url, "model": model}


class OMemAdapter:
    def __init__(self, config: Optional[OMemAdapterConfig] = None):
        self.config = config or OMemAdapterConfig()
        self._omem_root = self._resolve_omem_root(self.config.omem_root)

    def ingest_conversation(self, sample_id: str, conversation: List[Dict[str, Any]]) -> Any:
        turns = self._normalize_turns(conversation)
        if self.config.use_real_omem:
            try:
                return self._ingest_real(sample_id, turns)
            except Exception as exc:
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
        return out

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

    def retrieve_original(self, run_ctx: Any, query: str, top_k: int) -> List[Dict[str, Any]]:
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
                    )
                )
                if isinstance(answer, tuple) and answer:
                    return str(answer[0]).strip()
        return self._oracle_fallback_answer(oracle_context, query)

    def _generate_online_answer(self, run_ctx: Any, query: str, top_k: int) -> str:
        items = self.retrieve_original(run_ctx, query, top_k)
        if not items:
            return "I don't know"
        best = str(items[0].get("text", "")).strip()
        if not best:
            return "I don't know"
        return best

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

    def _normalize_turns(self, conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for idx, turn in enumerate(conversation):
            speaker = str(turn.get("speaker") or turn.get("role") or "UNKNOWN").strip()
            text = str(turn.get("text") or turn.get("content") or "").strip()
            timestamp = str(turn.get("timestamp") or turn.get("time") or "").strip()
            if not text:
                continue
            out.append(
                {
                    "turn_index": int(turn.get("turn_index", idx)),
                    "speaker": speaker,
                    "text": text,
                    "timestamp": timestamp,
                }
            )
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

        user_name = self._guess_user_name(turns)
        agent_name = self._guess_agent_name(turns, user_name)
        memory_dir = str((Path(self.config.memory_dir) / sample_id).resolve())
        Path(memory_dir).mkdir(parents=True, exist_ok=True)
        client = AsyncOpenAI(base_url=self.config.base_url, api_key=self.config.api_key)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_model = SentenceTransformer(self.config.embedding_model_name).to(device)
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
        self._run_awaitable(self._feed_turns(memory_manager, turns, user_name))
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
            await memory_manager.receive_message(
                message=str(turn.get("text", "")),
                index=int(turn.get("turn_index", idx)),
                client=memory_manager.client,
                timestamp=str(turn.get("timestamp", "")),
                user_speak=user_speak,
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

    def _guess_user_name(self, turns: List[Dict[str, Any]]) -> str:
        for turn in turns:
            name = str(turn.get("speaker", "")).strip()
            if name:
                return name
        return "User"

    def _guess_agent_name(self, turns: List[Dict[str, Any]], user_name: str) -> str:
        for turn in turns:
            name = str(turn.get("speaker", "")).strip()
            if name and name != user_name:
                return name
        return "Assistant"

    def _run_awaitable(self, awaitable: Any) -> Any:
        try:
            return asyncio.run(awaitable)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(awaitable)
            finally:
                loop.close()

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
