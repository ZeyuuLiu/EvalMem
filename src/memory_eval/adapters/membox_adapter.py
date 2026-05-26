from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from memory_eval.adapters.base import BaseMemoryAdapter
from memory_eval.eval_core.models import AdapterTrace, RetrievedItem
from memory_eval.eval_core.utils import normalize_text, split_tokens, text_match


@dataclass(frozen=True)
class MemboxAdapterConfig:
    api_key: str = ""
    base_url: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    keys_path: str = ""
    membox_root: str = ""
    memory_dir: str = "outputs/membox_memory"
    run_id_prefix: str = "membox"
    top_k_retrieve: Optional[int] = None
    answer_top_n: int = 5
    text_modes: Optional[List[str]] = None
    request_timeout_sec: float = 120.0


class MemboxAdapter(BaseMemoryAdapter):
    family = "membox"

    def __init__(self, config: Optional[MemboxAdapterConfig] = None):
        super().__init__()
        cfg = config or MemboxAdapterConfig()
        creds = self.merge_runtime_credentials(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            model=cfg.llm_model,
            keys_path=cfg.keys_path,
            require_complete=False,
        )
        self.config = MemboxAdapterConfig(
            api_key=creds["api_key"],
            base_url=creds["base_url"],
            llm_model=creds["model"] or cfg.llm_model,
            embedding_model=cfg.embedding_model,
            keys_path=creds["keys_path"],
            membox_root=cfg.membox_root,
            memory_dir=cfg.memory_dir,
            run_id_prefix=cfg.run_id_prefix,
            top_k_retrieve=cfg.top_k_retrieve,
            answer_top_n=cfg.answer_top_n,
            text_modes=cfg.text_modes,
            request_timeout_sec=cfg.request_timeout_sec,
        )
        self._membox_root = self._resolve_membox_root(self.config.membox_root)
        self._module_load_error = ""
        try:
            self._module = self._load_membox_module()
        except Exception as exc:
            # Catch any error from loading the upstream Membox module (e.g. PEP-604 syntax
            # which requires Python 3.10+, missing optional deps, etc.).
            # The adapter remains constructable but ingest_conversation will raise.
            self._module = None
            self._module_load_error = f"{type(exc).__name__}: {exc}"

    def ingest_conversation(self, sample_id: str, conversation: List[Dict[str, Any]]) -> Any:
        if self._module is None:
            raise RuntimeError(f"Membox runtime dependencies are unavailable: {self._module_load_error}")
        if not str(self.config.api_key or "").strip():
            raise RuntimeError(
                "MemboxAdapter: api_key is empty. Set MemboxAdapterConfig.api_key or "
                "configs/keys.local.json (or env MEMORY_EVAL_API_KEY / OPENAI_API_KEY)."
            )
        if not str(self.config.base_url or "").strip():
            raise RuntimeError(
                "MemboxAdapter: base_url is empty. Set MemboxAdapterConfig.base_url or "
                "configs/keys.local.json (or env MEMORY_EVAL_BASE_URL / OPENAI_BASE_URL)."
            )
        turns = self.normalize_turns(conversation)
        conv_payload = self._to_membox_conversation(turns)
        run_id = self.build_run_id(self.config.run_id_prefix, sample_id)
        output_root = (Path(self.config.memory_dir) / sample_id).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        raw_data_path = output_root / "raw_data.json"
        raw_data_path.write_text(
            json.dumps([{"sample_id": sample_id, "conversation": conv_payload, "qa": []}], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        cfg = self._apply_runtime_config(raw_data_path=raw_data_path, output_root=output_root, run_id=run_id)
        worker = self._build_worker()
        builder = self._module.MemoryBuilder(worker)
        boxes = builder.build_all()
        if not cfg.CHECKPOINT_EVERY_SAMPLE:
            builder.save(boxes)
        builder.summarize_and_log()

        if any(m in {"content_trace_event", "trace_event"} for m in cfg.GEN_TEXT_MODES):
            linker = self._module.TraceLinker(worker, trace_metrics=cfg.TRACE_METRICS)
            linker.run()

        retriever = self._build_retriever(worker, cfg.TOP_K_RETRIEVE)
        return self._create_runtime_context(
            sample_id=sample_id,
            turns=turns,
            raw_data_path=raw_data_path,
            output_root=output_root,
            run_id=run_id,
            worker=worker,
            retriever=retriever,
        )

    def capabilities(self) -> Dict[str, Any]:
        out = super().capabilities()
        out.update(
            {
                "flavor": "stable_eval" if "stableeval" in str(self._membox_root).lower() else "original",
                "supports_full_memory_export": True,
                "supports_native_retrieval": True,
                "supports_oracle_generation": True,
                "supports_online_generation": True,
                "supports_high_recall_candidates": True,
                "requires_remote_llm": True,
                "supports_build_artifact_reuse": True,
                "runtime_dependency_error": self._module_load_error,
            }
        )
        return out

    def export_build_artifact(self, run_ctx: Any) -> Dict[str, Any]:
        return {
            "sample_id": str(run_ctx.get("sample_id", "")),
            "run_id": str(run_ctx.get("run_id", "")),
            "raw_data_path": str(run_ctx.get("raw_data_path", "")),
            "output_root": str(run_ctx.get("output_root", "")),
            "config_snapshot": dict(run_ctx.get("config_snapshot", {})),
            "runtime_warnings": list(run_ctx.get("runtime_warnings", [])),
        }

    def load_build_artifact(self, manifest: Dict[str, Any]) -> Any:
        if self._module is None:
            raise RuntimeError(f"Membox runtime dependencies are unavailable: {self._module_load_error}")
        sample_id = str(manifest.get("sample_id", "")).strip()
        run_id = str(manifest.get("run_id", "")).strip()
        raw_data_path = Path(str(manifest.get("raw_data_path", "")).strip()).resolve()
        output_root = Path(str(manifest.get("output_root", "")).strip()).resolve()
        if not sample_id or not run_id:
            raise ValueError("membox build artifact missing sample_id or run_id")
        if not raw_data_path.exists():
            raise FileNotFoundError(f"membox build artifact raw_data_path not found: {raw_data_path}")
        cfg = self._apply_runtime_config(raw_data_path=raw_data_path, output_root=output_root, run_id=run_id)
        worker = self._build_worker()
        retriever = self._build_retriever(worker, cfg.TOP_K_RETRIEVE)
        turns = self.normalize_turns(self._read_turns_from_raw_data(raw_data_path))
        return self._create_runtime_context(
            sample_id=sample_id,
            turns=turns,
            raw_data_path=raw_data_path,
            output_root=output_root,
            run_id=run_id,
            worker=worker,
            retriever=retriever,
        )

    def ingest_aux_records(self, run_ctx: Any, aux_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        records = super().ingest_aux_records(run_ctx, aux_records)
        if not isinstance(run_ctx, dict):
            return records
        retriever = run_ctx.get("retriever")
        if retriever is None or not hasattr(retriever, "all_boxes"):
            return records
        existing = {str(box.get("box_id", "")) for box in getattr(retriever, "all_boxes", []) if isinstance(box, dict)}
        pseudo_boxes: List[Dict[str, Any]] = []
        for idx, record in enumerate(records):
            rid = str(record.get("id", f"memwiki_aux_{idx}"))
            box_id = "memwiki_aux_" + normalize_text(rid).replace(" ", "_")[:80]
            if box_id in existing:
                continue
            text = str(record.get("text", "")).strip()
            if not text:
                continue
            meta = dict(record.get("meta", {})) if isinstance(record.get("meta", {}), dict) else {}
            tags = meta.get("tags", {}) if isinstance(meta.get("tags", {}), dict) else {}
            tag_text = " ".join(
                str(v)
                for values in tags.values()
                for v in (values if isinstance(values, list) else [values])
                if str(v).strip()
            )
            pseudo_boxes.append(
                {
                    "sample_id": 0,
                    "box_id": box_id,
                    "start_time": "",
                    "end_time": "",
                    "coverage": {},
                    "features": {
                        "content_text": text,
                        "events_text": "",
                        "topic_kw_text": tag_text,
                        "events": [],
                    },
                    "memwiki_aux_meta": meta,
                }
            )
            existing.add(box_id)
        if pseudo_boxes:
            retriever.all_boxes.extend(pseudo_boxes)
            run_ctx["memwiki_aux_native_indexed"] = int(run_ctx.get("memwiki_aux_native_indexed", 0) or 0) + len(pseudo_boxes)
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
        answer_online = self.generate_online_answer(run_ctx, query, top_k)
        return AdapterTrace(
            memory_view=memory_view,
            retrieved_items=retrieved_items,
            answer_online=answer_online,
            answer_oracle=answer_oracle,
            raw_trace={
                "memory_system": "membox",
                "run_id": str(run_ctx.get("run_id", "")),
                "output_root": str(run_ctx.get("output_root", "")),
                "runtime_warnings": list(run_ctx.get("runtime_warnings", [])),
            },
        )

    def export_full_memory(self, run_ctx: Any) -> List[Dict[str, Any]]:
        boxes = self._load_boxes(run_ctx)
        retriever = run_ctx.get("retriever") if isinstance(run_ctx, dict) else None
        runtime_boxes = list(getattr(retriever, "all_boxes", []) or []) if retriever is not None else []
        existing_box_ids = {str(box.get("box_id", "")) for box in boxes if isinstance(box, dict)}
        boxes.extend(
            [
                box
                for box in runtime_boxes
                if isinstance(box, dict)
                and str(box.get("box_id", "")).startswith("memwiki_aux_")
                and str(box.get("box_id", "")) not in existing_box_ids
            ]
        )
        traces = self._load_time_traces(run_ctx)
        out: List[Dict[str, Any]] = []
        for box in boxes:
            feat = box.get("features", {})
            content_text = str(feat.get("content_text", "")).strip()
            topic_kw_text = str(feat.get("topic_kw_text", "")).strip()
            events_text = str(feat.get("events_text", "")).strip()
            merged_text = "\n".join([x for x in [content_text, events_text, topic_kw_text] if x]).strip()
            if not merged_text:
                continue
            out.append(
                {
                    "id": f"box-{box.get('box_id')}",
                    "text": merged_text,
                    "meta": {
                        "source": "membox_final_box",
                        "box_id": box.get("box_id"),
                        "sample_id": box.get("sample_id"),
                        "start_time": box.get("start_time"),
                        "coverage": dict(box.get("coverage", {})) if isinstance(box.get("coverage", {}), dict) else {},
                        "events": list(feat.get("events", [])) if isinstance(feat.get("events", []), list) else [],
                        "content_text": content_text,
                        "events_text": events_text,
                        "topic_kw_text": topic_kw_text,
                    },
                }
            )
        for trace in traces:
            entries_text = str(trace.get("entries_text", "")).strip()
            if not entries_text:
                continue
            entry_events: List[str] = []
            for entry in trace.get("entries", []) if isinstance(trace.get("entries", []), list) else []:
                if isinstance(entry, dict):
                    entry_events.extend([str(ev).strip() for ev in entry.get("events", []) if str(ev).strip()])
            out.append(
                {
                    "id": f"trace-{trace.get('trace_id')}",
                    "text": entries_text,
                    "meta": {
                        "source": "membox_time_trace",
                        "trace_id": trace.get("trace_id"),
                        "sample_id": trace.get("sample_id"),
                        "metric": trace.get("metric"),
                        "box_ids": list(trace.get("box_ids", [])) if isinstance(trace.get("box_ids", []), list) else [],
                        "entries_text": entries_text,
                        "events": entry_events,
                    },
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
        signals = [query] + list(f_key or [])
        signal_tokens = set()
        for s in signals:
            signal_tokens.update(split_tokens(str(s)))

        scored: List[tuple[float, Dict[str, Any]]] = []
        for rec in memory_corpus:
            text = str(rec.get("text", ""))
            variants = [text]
            meta = rec.get("meta", {})
            if isinstance(meta, dict):
                variants.extend(
                    [
                        str(meta.get("content_text", "")),
                        str(meta.get("events_text", "")),
                        str(meta.get("topic_kw_text", "")),
                        str(meta.get("entries_text", "")),
                    ]
                )
                variants.extend([str(ev) for ev in meta.get("events", []) if str(ev).strip()] if isinstance(meta.get("events", []), list) else [])
            if any(text_match(fact, variant) for fact in f_key for variant in variants if fact):
                scored.append((10.0, rec))
                continue
            text_tokens = set(split_tokens(text))
            overlap = len(signal_tokens & text_tokens) if signal_tokens and text_tokens else 0
            if overlap > 0:
                scored.append((float(overlap), rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [rec for _, rec in scored]

    def hybrid_retrieve_candidates(
        self,
        run_ctx: Any,
        query: str,
        f_key: List[str],
        evidence_texts: List[str],
        top_n: int = 100,
    ) -> List[Dict[str, Any]]:
        rankings = self._score_and_rank(run_ctx, " ".join([query] + list(f_key or []) + list(evidence_texts or [])), [])
        box_map = {str(rec.get("id", "")): rec for rec in self.export_full_memory(run_ctx)}
        out: List[Dict[str, Any]] = []
        for rank, item in enumerate(rankings[: max(1, int(top_n or 1))]):
            rec = box_map.get(item["id"])
            if rec is None:
                continue
            cloned = dict(rec)
            meta = dict(cloned.get("meta", {})) if isinstance(cloned.get("meta", {}), dict) else {}
            meta["hybrid_rank"] = rank + 1
            meta["hybrid_score"] = float(item.get("score", 0.0))
            cloned["meta"] = meta
            out.append(cloned)
        return out

    def retrieve_original(self, run_ctx: Any, query: str, top_k: int) -> List[Dict[str, Any]]:
        rankings = self._score_and_rank(run_ctx, query, [])
        box_map = {str(rec.get("id", "")): rec for rec in self.export_full_memory(run_ctx)}
        out: List[Dict[str, Any]] = []
        for rank, item in enumerate(rankings[: max(1, int(top_k or 1))]):
            rec = box_map.get(item["id"])
            if rec is None:
                continue
            meta = dict(rec.get("meta", {})) if isinstance(rec.get("meta", {}), dict) else {}
            meta["source"] = "membox_native_retrieval"
            meta["native_rank"] = rank
            out.append(
                {
                    "id": rec["id"],
                    "text": str(rec.get("text", "")),
                    "score": float(item.get("score", 0.0)),
                    "meta": meta,
                }
            )
        return out

    def generate_online_answer(self, run_ctx: Any, query: str, top_k: int = 5) -> str:
        retrieved = self.retrieve_original(run_ctx, query, top_k)
        contexts = [str(item.get("text", "")).strip() for item in retrieved if str(item.get("text", "")).strip()]
        if not contexts:
            return ""
        prompt = self._module.Config.PROMPT_QA_ANSWER.format(memories="\n\n".join(contexts), question=query)
        return str(run_ctx["worker"].chat_completion(prompt, note="MemboxAdapter_OnlineAnswer")).strip()

    def generate_oracle_answer(self, run_ctx: Any, query: str, oracle_context: str) -> str:
        context = str(oracle_context or "").strip()
        if not context:
            return ""
        prompt = self._module.Config.PROMPT_QA_ANSWER.format(memories=context, question=query)
        return str(run_ctx["worker"].chat_completion(prompt, note="MemboxAdapter_OracleAnswer")).strip()

    def _load_boxes(self, run_ctx: Any) -> List[Dict[str, Any]]:
        path = Path(str(run_ctx["config_snapshot"]["final_content_file"]))
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _load_time_traces(self, run_ctx: Any) -> List[Dict[str, Any]]:
        path = Path(str(run_ctx["config_snapshot"]["time_trace_file"]))
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _score_and_rank(self, run_ctx: Any, query: str, evidence: List[str]) -> List[Dict[str, Any]]:
        retriever = run_ctx.get("retriever")
        if retriever is None:
            raise RuntimeError("membox run_ctx missing retriever")
        qa = {
            "question": query,
            "evidence": list(evidence or []),
            "id": normalize_text(query)[:80],
        }
        rankings, sim_map, _ = retriever._score_and_rank(0, qa)
        ordered = rankings.get("content_event_topic_kw", []) or []
        return [{"id": f"box-{bid}", "score": float(sim_map.get(bid, -1.0)), "box_id": bid} for bid in ordered]

    def _resolve_membox_root(self, configured: str) -> Path:
        if configured:
            return Path(configured).resolve()
        return Path(__file__).resolve().parents[3] / "system" / "Membox_stableEval"

    def _apply_runtime_config(self, *, raw_data_path: Path, output_root: Path, run_id: str):
        cfg = self._module.Config
        cfg.API_KEY = self.config.api_key
        cfg.BASE_URL = self.config.base_url
        cfg.LLM_MODEL = self.config.llm_model
        cfg.EMBEDDING_MODEL = self.config.embedding_model
        cfg.RAW_DATA_FILE = str(raw_data_path)
        cfg.OUTPUT_BASE_DIR = str(output_root)
        cfg.LIMIT_CONVERSATIONS = 1
        cfg.LIMIT_SESSIONS = None
        cfg.TOP_K_RETRIEVE = self.config.top_k_retrieve
        cfg.ANSWER_TOP_N = int(self.config.answer_top_n or 5)
        cfg.GEN_TEXT_MODES = list(self.config.text_modes or ["content_trace_event"])
        cfg.apply_run_id(run_id)
        return cfg

    def _build_worker(self):
        worker = self._module.LLMWorker()
        runtime_warnings = []
        worker._memory_eval_runtime_warnings = runtime_warnings
        timeout = float(self.config.request_timeout_sec or 0.0)
        if timeout <= 0:
            return worker

        def get_embedding(text, note="Emb"):
            try:
                if not text:
                    return [0.0] * 1536
                resp = worker.client.embeddings.create(
                    input=str(text).replace("\n", " "),
                    model=self._module.Config.EMBEDDING_MODEL,
                    timeout=timeout,
                )
                emb = None
                try:
                    emb = resp.data[0].embedding
                except Exception as exc:
                    runtime_warnings.append(f"embedding response missing embedding payload: {exc}")
                    emb = None
                return emb if emb is not None else [0.0] * 1536
            except Exception as exc:
                runtime_warnings.append(f"embedding request failed: {exc}")
                return [0.0] * 1536

        def chat_completion(prompt, note="Completion", json_mode=False, extra=None):
            try:
                kwargs = {
                    "model": self._module.Config.LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "timeout": timeout,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = worker.client.chat.completions.create(**kwargs)
                extra_payload = {"prompt_tokens_est": worker.count_tokens(prompt)}
                if extra:
                    extra_payload.update(extra)
                self._module.TokenAnalyzer.log_usage(resp.usage, note, extra_payload)
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                runtime_warnings.append(f"chat completion failed: {exc}")
                return "{}" if json_mode else ""

        worker.get_embedding = get_embedding
        worker.chat_completion = chat_completion
        return worker

    def _build_retriever(self, worker: Any, top_k: Optional[int]):
        retriever = self._module.SimpleRetriever(worker, top_k=top_k)
        retriever.load()
        return retriever

    def _create_runtime_context(
        self,
        *,
        sample_id: str,
        turns: List[Dict[str, Any]],
        raw_data_path: Path,
        output_root: Path,
        run_id: str,
        worker: Any,
        retriever: Any,
    ) -> Dict[str, Any]:
        cfg = self._module.Config
        return {
            "sample_id": sample_id,
            "conversation": turns,
            "raw_data_path": str(raw_data_path),
            "output_root": str(output_root),
            "run_id": run_id,
            "worker": worker,
            "retriever": retriever,
            "runtime_warnings": getattr(worker, "_memory_eval_runtime_warnings", []),
            "config_snapshot": {
                "llm_model": cfg.LLM_MODEL,
                "embedding_model": cfg.EMBEDDING_MODEL,
                "output_dir": cfg.OUTPUT_DIR,
                "final_content_file": cfg.FINAL_CONTENT_FILE,
                "time_trace_file": cfg.TIME_TRACE_FILE,
            },
        }

    def _read_turns_from_raw_data(self, raw_data_path: Path) -> List[Dict[str, Any]]:
        payload = json.loads(raw_data_path.read_text(encoding="utf-8"))
        if not payload:
            return []
        conv = payload[0].get("conversation", {})
        turns: List[Dict[str, Any]] = []
        turn_index = 0
        for key in sorted(conv.keys()):
            if not str(key).startswith("session_") or str(key).endswith("_date_time"):
                continue
            session_turns = conv.get(key, [])
            timestamp = str(conv.get(f"{key}_date_time", "")).strip()
            if not isinstance(session_turns, list):
                continue
            for turn in session_turns:
                if not isinstance(turn, dict):
                    continue
                turns.append(
                    {
                        "turn_index": turn_index,
                        "speaker": str(turn.get("speaker", "")).strip(),
                        "text": str(turn.get("text", "")).strip(),
                        "timestamp": timestamp,
                    }
                )
                turn_index += 1
        return turns

    def _load_membox_module(self):
        module_path = self._membox_root / "membox.py"
        if not module_path.exists():
            raise FileNotFoundError(f"Membox entry file not found: {module_path}")
        spec = importlib.util.spec_from_file_location("memory_eval_membox_runtime", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load Membox module from {module_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("memory_eval_membox_runtime", mod)
        spec.loader.exec_module(mod)
        return mod

    def _to_membox_conversation(self, turns: List[Dict[str, Any]]) -> Dict[str, Any]:
        speaker_a = self.guess_user_name(turns)
        speaker_b = self.guess_agent_name(turns, speaker_a)
        sessions: Dict[str, List[Dict[str, Any]]] = {}
        session_ts: Dict[str, str] = {}
        for turn in turns:
            session_id = self._timestamp_to_session(turn.get("timestamp", ""))
            sessions.setdefault(session_id, [])
            if session_id not in session_ts:
                session_ts[session_id] = str(turn.get("timestamp", "")).strip()
            sessions[session_id].append(
                {
                    "speaker": str(turn.get("speaker", "")).strip() or speaker_a,
                    "text": str(turn.get("text", "")).strip(),
                }
            )

        out: Dict[str, Any] = {"speaker_a": speaker_a, "speaker_b": speaker_b}
        for idx, session_name in enumerate(sorted(sessions.keys()), start=1):
            key = f"session_{idx}"
            out[key] = sessions[session_name]
            out[f"{key}_date_time"] = session_ts.get(session_name, "")
        return out

    def _timestamp_to_session(self, timestamp: str) -> str:
        ts = str(timestamp or "").strip()
        return ts or "session_unknown"
