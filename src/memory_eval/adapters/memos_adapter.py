from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from memory_eval.adapters.base import BaseMemoryAdapter
from memory_eval.eval_core.models import AdapterTrace, RetrievedItem


@dataclass
class MemOSAdapterConfig:
    memos_root: str = ""
    memos_url: str = ""
    memos_online_url: str = ""
    memos_key: str = ""
    api_key: str = ""
    base_url: str = ""
    chat_model_api_key: str = ""
    chat_model_base_url: str = ""
    chat_model: str = "gpt-4o-mini"
    llm_model: str = "gpt-4o-mini"
    keys_path: str = ""
    memory_dir: str = ""
    lib: str = "memos-api"
    search_mode: str = "fast"


class MemOSAdapter(BaseMemoryAdapter):
    family = "memos"

    def __init__(self, config: Optional[MemOSAdapterConfig] = None):
        super().__init__()
        cfg = config or MemOSAdapterConfig()
        creds = self.merge_runtime_credentials(
            api_key=cfg.api_key or cfg.chat_model_api_key,
            base_url=cfg.base_url or cfg.chat_model_base_url,
            model=cfg.chat_model or cfg.llm_model,
            keys_path=cfg.keys_path,
            require_complete=False,
        )
        self.config = MemOSAdapterConfig(
            memos_root=cfg.memos_root,
            memos_url=cfg.memos_url,
            memos_online_url=cfg.memos_online_url,
            memos_key=cfg.memos_key,
            api_key=creds["api_key"],
            base_url=creds["base_url"],
            chat_model_api_key=cfg.chat_model_api_key or creds["api_key"],
            chat_model_base_url=cfg.chat_model_base_url or creds["base_url"],
            chat_model=cfg.chat_model or creds["model"] or "gpt-4o-mini",
            llm_model=creds["model"] or cfg.llm_model or cfg.chat_model or "gpt-4o-mini",
            keys_path=creds["keys_path"],
            memory_dir=cfg.memory_dir,
            lib=cfg.lib,
            search_mode=cfg.search_mode,
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
            "memwiki_aux_strategy": "add_aux_messages_to_native_client",
        }

    def runtime_manifest(self) -> Dict[str, Any]:
        return {"family": self.family, "flavor": "stable_eval", "capabilities": self.capabilities()}

    def ingest_conversation(self, sample_id: str, conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
        turns = self.normalize_turns(conversation)
        deps = self._load_deps()
        self._apply_env()
        client = deps["MemosApiClient"]() if self.config.lib == "memos-api" else deps["MemosApiOnlineClient"]()
        speaker_a, speaker_b = self._resolve_speakers(turns)
        speaker_a_user_id = f"{sample_id}_speaker_a_adapter"
        speaker_b_user_id = f"{sample_id}_speaker_b_adapter"
        run_dir = self._make_run_dir(sample_id)
        added_memories: List[Dict[str, Any]] = []
        for idx, turn in enumerate(turns):
            iso_date = self._to_iso_time(turn.get("time", ""))
            conv_id = f"{sample_id}_session_{idx}"
            a_message, b_message = self._build_messages_for_turn(turn, speaker_a=speaker_a, speaker_b=speaker_b, iso_date=iso_date)
            try:
                added_memories.extend(client.add([a_message], speaker_a_user_id, conv_id, batch_size=1) or [])
            except Exception:
                pass
            try:
                added_memories.extend(client.add([b_message], speaker_b_user_id, conv_id, batch_size=1) or [])
            except Exception:
                pass
        return {
            "sample_id": sample_id,
            "turns": turns,
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "speaker_a_user_id": speaker_a_user_id,
            "speaker_b_user_id": speaker_b_user_id,
            "client": client,
            "run_dir": str(run_dir),
            "added_memories": added_memories,
            "artifact_refs": {"run_dir": str(run_dir)},
        }

    def export_full_memory(self, run_ctx: Any) -> List[Dict[str, Any]]:
        added = list(run_ctx.get("added_memories") or [])
        if added:
            out: List[Dict[str, Any]] = []
            for idx, item in enumerate(added):
                text = self._extract_memory_text(item)
                out.append(
                    {
                        "id": f"memos-added-{idx}",
                        "text": text,
                        "meta": {"source": "memos_add_response", "raw": item},
                    }
                )
            return self.append_aux_records(out, run_ctx)
        out = []
        for idx, turn in enumerate(run_ctx.get("turns") or []):
            out.append(
                {
                    "id": f"memos-turn-{idx}",
                    "text": f"{turn.get('speaker', '')}: {turn.get('text', '')}".strip(),
                    "meta": {"source": "memos_ingest_fallback", "time": turn.get("time", "")},
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
        ranked: List[tuple[int, Dict[str, Any]]] = []
        for record in memory_corpus or []:
            text = str(record.get("text", "")).strip()
            score = sum(1 for token in terms if token and token in text.lower())
            if score > 0:
                ranked.append((score, record))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in ranked[:100]]

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
        client = run_ctx["client"]
        results: List[Dict[str, Any]] = []
        speaker_queries = [
            ("speaker_a", run_ctx["speaker_a_user_id"]),
            ("speaker_b", run_ctx["speaker_b_user_id"]),
        ]
        for speaker_side, user_id in speaker_queries:
            raw = client.search(query=query, user_id=user_id, top_k=top_k)
            text_items = []
            if isinstance(raw, dict):
                text_items = ((raw.get("text_mem") or [{}])[0].get("memories") or [])
            for rank, item in enumerate(text_items):
                text = self._extract_memory_text(item)
                results.append(
                    {
                        "id": f"{speaker_side}-{rank}",
                        "text": text,
                        "score": float(item.get("score", max(top_k - rank, 1))),
                        "meta": {
                            "source": "memos_native_retrieval",
                            "speaker_side": speaker_side,
                            "native_rank": rank,
                            "raw": item,
                        },
                    }
                )
        return results

    def generate_online_answer(self, run_ctx: Any, question: str, top_k: int = 5) -> str:
        retrieved = self.retrieve_original(run_ctx, question, top_k=top_k)
        context = self._render_context(run_ctx, retrieved)
        return self._answer_with_context(question=question, context=context)

    def generate_oracle_answer(self, run_ctx: Any, question: str, oracle_context: str) -> str:
        context = self._render_oracle_context(run_ctx, oracle_context)
        return self._answer_with_context(question=question, context=context)

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
            "speaker_a_user_id": str(run_ctx.get("speaker_a_user_id", "")),
            "speaker_b_user_id": str(run_ctx.get("speaker_b_user_id", "")),
            "artifact_refs": dict(run_ctx.get("artifact_refs", {})),
        }

    def load_build_artifact(self, manifest: Dict[str, Any]) -> Any:
        raise RuntimeError("MemOSAdapter 暂未实现 build artifact 恢复，请在 baseline/eval 阶段实时 ingest。")

    def ingest_aux_records(self, run_ctx: Any, aux_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        records = super().ingest_aux_records(run_ctx, aux_records)
        if not isinstance(run_ctx, dict) or "client" not in run_ctx:
            return records
        client = run_ctx["client"]
        injected_raw: List[Any] = []
        for idx, record in enumerate(records):
            text = str(record.get("text", "")).strip()
            if not text:
                continue
            message = {
                "role": "system",
                "content": f"MemWiki auxiliary retrieval index: {text}",
                "chat_time": "2024-01-01T00:00:00+00:00",
            }
            conv_id = f"{run_ctx.get('sample_id', 'sample')}_memwiki_aux_{idx}"
            for user_key in ("speaker_a_user_id", "speaker_b_user_id"):
                user_id = str(run_ctx.get(user_key, "")).strip()
                if not user_id:
                    continue
                try:
                    injected_raw.extend(client.add([message], user_id, conv_id, batch_size=1) or [])
                except Exception:
                    continue
        if injected_raw:
            run_ctx.setdefault("added_memories", []).extend(injected_raw)
            run_ctx["memwiki_aux_native_indexed"] = int(run_ctx.get("memwiki_aux_native_indexed", 0) or 0) + len(injected_raw)
        return records

    def _load_deps(self) -> Dict[str, Any]:
        memos_root = Path(self.config.memos_root or Path(__file__).resolve().parents[3] / "system" / "MemOS-main")
        eval_root = memos_root / "evaluation"
        locomo_dir = eval_root / "scripts" / "locomo"
        utils_dir = eval_root / "scripts" / "utils"
        for path in (str(eval_root), str(locomo_dir), str(utils_dir), str(memos_root)):
            if path not in sys.path:
                sys.path.insert(0, path)
        from client import MemosApiClient, MemosApiOnlineClient
        from prompts import ANSWER_PROMPT_MEMOS, TEMPLATE_MEMOS
        return {
            "MemosApiClient": MemosApiClient,
            "MemosApiOnlineClient": MemosApiOnlineClient,
            "ANSWER_PROMPT_MEMOS": ANSWER_PROMPT_MEMOS,
            "TEMPLATE_MEMOS": TEMPLATE_MEMOS,
        }

    def _apply_env(self) -> None:
        if self.config.memos_url:
            os.environ["MEMOS_URL"] = str(self.config.memos_url)
        if self.config.memos_online_url:
            os.environ["MEMOS_ONLINE_URL"] = str(self.config.memos_online_url)
        if self.config.memos_key:
            os.environ["MEMOS_KEY"] = str(self.config.memos_key)
        os.environ["SEARCH_MODE"] = str(self.config.search_mode or "fast")
        if self.config.chat_model_api_key or self.config.api_key:
            os.environ["CHAT_MODEL_API_KEY"] = str(self.config.chat_model_api_key or self.config.api_key)
        if self.config.chat_model_base_url or self.config.base_url:
            os.environ["CHAT_MODEL_BASE_URL"] = str(self.config.chat_model_base_url or self.config.base_url)
        if self.config.chat_model:
            os.environ["CHAT_MODEL"] = str(self.config.chat_model)
        elif self.config.llm_model:
            os.environ["CHAT_MODEL"] = str(self.config.llm_model)

    def _resolve_speakers(self, turns: List[Dict[str, Any]]) -> tuple[str, str]:
        speakers = []
        for turn in turns:
            speaker = str(turn.get("speaker", "")).strip()
            if speaker and speaker not in speakers:
                speakers.append(speaker)
        if not speakers:
            return "speaker_a", "speaker_b"
        if len(speakers) == 1:
            return speakers[0], "speaker_b"
        return speakers[0], speakers[1]

    def _build_messages_for_turn(self, turn: Dict[str, Any], *, speaker_a: str, speaker_b: str, iso_date: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
        data = f"{turn.get('speaker', '')}: {turn.get('text', '')}".strip()
        is_a = str(turn.get("speaker", "")).strip() == speaker_a
        speaker_a_message = {"role": "user" if is_a else "assistant", "content": data, "chat_time": iso_date}
        speaker_b_message = {"role": "assistant" if is_a else "user", "content": data, "chat_time": iso_date}
        return speaker_a_message, speaker_b_message

    def _to_iso_time(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return "2024-01-01T00:00:00+00:00"
        if "T" in text:
            return text
        return text.replace(" UTC", "+00:00") if "UTC" in text else text

    def _extract_memory_text(self, item: Any) -> str:
        if isinstance(item, dict):
            for key in ("memory", "memory_value", "text", "content", "preference"):
                if item.get(key):
                    return str(item.get(key))
            return str(item)
        return str(item)

    def _render_context(self, run_ctx: Any, retrieved_items: List[Dict[str, Any]]) -> str:
        speaker_a_memories = [x["text"] for x in retrieved_items if x.get("meta", {}).get("speaker_side") == "speaker_a"]
        speaker_b_memories = [x["text"] for x in retrieved_items if x.get("meta", {}).get("speaker_side") == "speaker_b"]
        return (
            f"Speaker 1 ({run_ctx['speaker_a']}) memories:\n" + "\n".join(speaker_a_memories) + "\n\n"
            f"Speaker 2 ({run_ctx['speaker_b']}) memories:\n" + "\n".join(speaker_b_memories)
        ).strip()

    def _render_oracle_context(self, run_ctx: Any, oracle_context: str) -> str:
        return (
            f"Speaker 1 ({run_ctx['speaker_a']}) memories:\n{oracle_context}\n\n"
            f"Speaker 2 ({run_ctx['speaker_b']}) memories:\n{oracle_context}"
        )

    def _answer_with_context(self, *, question: str, context: str) -> str:
        prompt = (
            "You are a knowledgeable and helpful AI assistant.\n\n"
            "Use the provided memories to answer the question briefly and directly.\n"
            "The answer must be under 5-6 words.\n\n"
            f"{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        client = OpenAI(
            api_key=str(self.config.chat_model_api_key or self.config.api_key or ""),
            base_url=str(self.config.chat_model_base_url or self.config.base_url or "").rstrip("/") or None,
        )
        response = client.chat.completions.create(
            model=str(self.config.chat_model or self.config.llm_model or "gpt-4o-mini"),
            messages=[{"role": "system", "content": prompt}],
            temperature=0,
        )
        return str(response.choices[0].message.content or "").strip()

    def _make_run_dir(self, sample_id: str) -> Path:
        base = Path(self.config.memory_dir) if self.config.memory_dir else Path(tempfile.mkdtemp(prefix="memos_adapter_"))
        run_dir = base / str(sample_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
