from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from memory_eval.eval_core.models import EvalSample

NEGATIVE_PATTERNS = re.compile(
    r"\b(not mentioned|unknown|cannot be determined|can't determine|no information|n/a|none|not available)\b",
    re.IGNORECASE,
)


@dataclass
class LocomoEvalSample:
    sample_id: str
    question_id: str
    question: str
    answer_gold: str
    task_type: str
    f_key: List[str]
    oracle_context: str
    evidence_ids: List[str]
    evidence_texts: List[str]
    evidence_with_time: List[str]
    category: int = 0  # LoCoMo: 1=multi-hop, 2=temporal, 3=open, 4=single-hop, 5=adversarial
    construction_evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_eval_sample(self) -> EvalSample:
        """
        Convert LOCOMO sample into evaluator core contract.
        将 LOCOMO 样本转换为评估核心统一契约。
        """
        return EvalSample(
            sample_id=self.sample_id,
            question_id=self.question_id,
            question=self.question,
            answer_gold=self.answer_gold,
            task_type=self.task_type,
            f_key=list(self.f_key),
            oracle_context=self.oracle_context,
            category=self.category,
            evidence_ids=list(self.evidence_ids),
            evidence_texts=list(self.evidence_texts),
            evidence_with_time=list(self.evidence_with_time),
            construction_evidence=dict(self.construction_evidence),
        )


@dataclass
class LocomoSampleRegistry:
    """
    Fast lookup registry for runtime evaluation usage.
    """

    samples: List[LocomoEvalSample]
    by_question_id: Dict[str, LocomoEvalSample]
    by_normalized_question: Dict[str, List[LocomoEvalSample]]

    def get_by_question_id(self, question_id: str) -> Optional[LocomoEvalSample]:
        return self.by_question_id.get(str(question_id).strip())

    def find_by_query(self, query: str, sample_id: Optional[str] = None) -> Optional[LocomoEvalSample]:
        nq = _normalize_question(query)
        candidates = self.by_normalized_question.get(nq, [])
        if not candidates:
            return None
        if sample_id is None:
            return candidates[0]
        sid = str(sample_id).strip()
        for c in candidates:
            if c.sample_id == sid:
                return c
        return None


def _normalize_answer(ans: object) -> str:
    if ans is None:
        return ""
    if isinstance(ans, (int, float)):
        return str(ans)
    return str(ans).strip()


def _infer_task_type(answer: str) -> str:
    if not answer:
        return "NEG"
    return "NEG" if NEGATIVE_PATTERNS.search(answer) else "POS"


def _normalize_question(q: str) -> str:
    s = str(q or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _flatten_conversation(conv: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    session_datetime_map: Dict[int, str] = {}
    for key, value in conv.items():
        m = re.match(r"session_(\d+)_date_time$", str(key))
        if m:
            session_datetime_map[int(m.group(1))] = str(value).strip()

    session_items: List[Tuple[int, List[Dict[str, Any]]]] = []
    for key, value in conv.items():
        if key.startswith("session_") and isinstance(value, list):
            m = re.match(r"session_(\d+)$", key)
            if m:
                session_items.append((int(m.group(1)), value))
    session_items.sort(key=lambda x: x[0])

    utterances: List[Dict[str, Any]] = []
    utt_map: Dict[str, Dict[str, Any]] = {}
    turn_index = 0
    for session_idx, session in session_items:
        for turn in session:
            dia_id = str(turn.get("dia_id", "")).strip()
            if not dia_id:
                continue
            item = {
                "dia_id": dia_id,
                "speaker": str(turn.get("speaker", "")).strip(),
                "text": str(turn.get("text", "")).strip(),
                "turn_index": turn_index,
                "session_index": session_idx,
                "session_datetime": session_datetime_map.get(session_idx, ""),
            }
            turn_index += 1
            utterances.append(item)
            utt_map[dia_id] = item
    return utterances, utt_map


def _session_from_dia_id(dia_id: str) -> Optional[int]:
    m = re.match(r"D(\d+):\d+", str(dia_id))
    if not m:
        return None
    return int(m.group(1))


def _build_from_evidence(evidence_ids: List[str], utt_map: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str], str]:
    evidence_texts: List[str] = []
    evidence_with_time: List[str] = []
    for eid in evidence_ids:
        utt = utt_map.get(eid)
        if not utt:
            continue
        session_idx = utt.get("session_index")
        if session_idx is None:
            session_idx = _session_from_dia_id(eid)
        session_dt = str(utt.get("session_datetime", "")).strip()
        speaker = str(utt.get("speaker", "")).strip()
        text = str(utt.get("text", "")).strip()
        evidence_texts.append(text)
        evidence_with_time.append(
            "{dt} | {spk}: {txt}".format(
                dt=session_dt if session_dt else "UNKNOWN_TIME",
                spk=speaker if speaker else "UNKNOWN_SPEAKER",
                txt=text,
            )
        )
    oracle_context = "\n".join(evidence_with_time)
    return evidence_texts, evidence_with_time, oracle_context


def _default_f_key_extractor(evidence_texts: List[str], evidence_with_time: List[str], question: str, answer_gold: str) -> List[str]:
    # Rule-based fallback: use time-augmented evidence directly.
    if evidence_with_time:
        return list(evidence_with_time)
    return list(evidence_texts)


def build_locomo_eval_samples(
    dataset_path: str,
    limit: int | None = None,
    f_key_mode: str = "rule",
    f_key_extractor: Optional[Callable[[List[str], List[str], str, str], List[str]]] = None,
) -> List[LocomoEvalSample]:
    path = Path(dataset_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results: List[LocomoEvalSample] = []
    for episode in data:
        sample_id = str(episode.get("sample_id", "")).strip()
        _, utt_map = _flatten_conversation(episode.get("conversation", {}))
        for idx, qa in enumerate(episode.get("qa", [])):
            question = str(qa.get("question", "")).strip()
            answer_gold = _normalize_answer(qa.get("answer", ""))
            evidence_ids = [str(x).strip() for x in qa.get("evidence", []) if str(x).strip()]
            task_type = _infer_task_type(answer_gold)
            category = int(qa.get("category", 0))

            evidence_texts, evidence_with_time, oracle_context = _build_from_evidence(evidence_ids, utt_map)
            extractor = f_key_extractor or _default_f_key_extractor
            if f_key_mode == "llm":
                f_key = extractor(evidence_texts, evidence_with_time, question, answer_gold)
                construction_mode = "evidence_time_mapping_llm_fkey"
            else:
                f_key = extractor(evidence_texts, evidence_with_time, question, answer_gold)
                construction_mode = "evidence_time_mapping_rule_fkey"

            # NEG handling policy for this framework:
            # force empty key facts and no-relevant-memory oracle context.
            if task_type == "NEG":
                f_key = []
                oracle_context = "NO_RELEVANT_MEMORY"
                construction_mode = "negative_forced_empty_keyfacts"

            item = LocomoEvalSample(
                sample_id=sample_id,
                question_id=f"{sample_id}:{idx}",
                question=question,
                answer_gold=answer_gold,
                task_type=task_type,
                f_key=f_key,
                oracle_context=oracle_context,
                evidence_ids=evidence_ids,
                evidence_texts=evidence_texts,
                evidence_with_time=evidence_with_time,
                category=category,
                construction_evidence={
                    "mode": construction_mode,
                    "evidence_count": len(evidence_ids),
                    "resolved_evidence_count": len(evidence_texts),
                    "resolved_time_evidence_count": len(evidence_with_time),
                    "f_key_mode": f_key_mode,
                    "note": "oracle_context includes time-aware evidence; NEG uses forced empty keyfacts and no relevant memory context.",
                },
            )
            results.append(item)
            if limit is not None and len(results) >= limit:
                return results
    return results


def build_locomo_sample_registry(
    dataset_path: str,
    f_key_mode: str = "rule",
    f_key_extractor: Optional[Callable[[List[str], List[str], str, str], List[str]]] = None,
) -> LocomoSampleRegistry:
    samples = build_locomo_eval_samples(
        dataset_path=dataset_path,
        limit=None,
        f_key_mode=f_key_mode,
        f_key_extractor=f_key_extractor,
    )
    by_qid: Dict[str, LocomoEvalSample] = {}
    by_query: Dict[str, List[LocomoEvalSample]] = {}
    for s in samples:
        by_qid[s.question_id] = s
        nq = _normalize_question(s.question)
        by_query.setdefault(nq, []).append(s)
    return LocomoSampleRegistry(samples=samples, by_question_id=by_qid, by_normalized_question=by_query)
