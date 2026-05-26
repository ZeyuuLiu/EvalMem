from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List

from memory_eval.eval_core.prompts import (
    build_attribution_prompt,
    build_correctness_judge_prompt,
    build_encoding_neg_prompt,
    build_encoding_pos_prompt,
    build_generation_neg_answer_prompt,
    build_generation_neg_comparison_prompt,
    build_generation_pos_answer_prompt,
    build_generation_pos_comparison_prompt,
    build_retrieval_neg_prompt,
    build_retrieval_pos_prompt,
)


@dataclass(frozen=True)
class LLMAssistConfig:
    api_key: str
    base_url: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.0


def _extract_json_object(text: str) -> Dict[str, Any]:
    content = str(text or "").strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    if content.lower().startswith("json"):
        content = content[4:].strip()
    try:
        return json.loads(content)
    except Exception:
        pass
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        return json.loads(content[start : end + 1])
    raise ValueError("No valid JSON object found in LLM response")


def _chat_json(cfg: LLMAssistConfig, prompt: str, *, must_succeed: bool = False) -> Dict[str, Any] | None:
    if not cfg.api_key or not cfg.base_url:
        if must_succeed:
            raise RuntimeError("LLM assist requires non-empty api_key and base_url")
        return None
    payload = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "messages": [
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        url=f"{cfg.base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg.api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
        obj = json.loads(raw)
        content = str(obj["choices"][0]["message"]["content"]).strip()
        return _extract_json_object(content)
    except Exception as exc:
        if must_succeed:
            raise RuntimeError(f"LLM chat/completions failed or returned non-JSON: {exc}") from exc
        return None


def llm_judge_retrieval_noise(
    cfg: LLMAssistConfig,
    query: str,
    retrieved_items: List[Dict[str, Any]],
    *,
    must_succeed: bool = False,
) -> Dict[str, Any] | None:
    prompt = (
        "You are judging retrieval noise for a NEG task.\n"
        "Return JSON: {\"is_noise\": true|false, \"reason\": \"...\"}\n"
        f"Query: {query}\n"
        "Retrieved items:\n"
        + "\n".join([f"- score={it.get('score', 0)} text={it.get('text', '')}" for it in retrieved_items[:5]])
    )
    return _chat_json(cfg, prompt, must_succeed=must_succeed)


def llm_judge_encoding_storage(
    cfg: LLMAssistConfig,
    query: str,
    f_key: List[str],
    evidence_texts: List[str],
    candidates: List[Dict[str, Any]],
    task_type: str,
    *,
    must_succeed: bool = False,
) -> Dict[str, Any] | None:
    prompt = (
        build_encoding_neg_prompt(query=query, evidence_texts=evidence_texts, candidates=candidates)
        if task_type == "NEG"
        else build_encoding_pos_prompt(query=query, f_key=f_key, evidence_texts=evidence_texts, candidates=candidates)
    )
    return _chat_json(cfg, prompt, must_succeed=must_succeed)


def llm_judge_retrieval_quality_pos(
    cfg: LLMAssistConfig,
    query: str,
    f_key: List[str],
    evidence_texts: List[str],
    retrieved_items: List[Dict[str, Any]],
    *,
    rank_index: int = -1,
    hit_indices: List[int] | None = None,
    snr: float = 0.0,
    tau_rank: int = 5,
    tau_snr: float = 0.2,
    must_succeed: bool = False,
) -> Dict[str, Any] | None:
    hit_indices = list(hit_indices or [])
    prompt = build_retrieval_pos_prompt(
        query=query,
        f_key=f_key,
        evidence_texts=evidence_texts,
        retrieved_items=retrieved_items,
        rank_index=rank_index,
        hit_indices=hit_indices,
        snr=snr,
        tau_rank=tau_rank,
        tau_snr=tau_snr,
    )
    return _chat_json(cfg, prompt, must_succeed=must_succeed)


def llm_judge_retrieval_quality_neg(
    cfg: LLMAssistConfig,
    query: str,
    retrieved_items: List[Dict[str, Any]],
    *,
    must_succeed: bool = False,
) -> Dict[str, Any] | None:
    return _chat_json(cfg, build_retrieval_neg_prompt(query=query, retrieved_items=retrieved_items), must_succeed=must_succeed)


def llm_judge_fact_match(
    cfg: LLMAssistConfig,
    question: str,
    fact: str,
    candidate_text: str,
    *,
    must_succeed: bool = False,
) -> Dict[str, Any] | None:
    prompt = (
        "You are matching a key fact against a memory text snippet.\n"
        "Return JSON: {\"match\": true|false, \"ambiguous\": true|false, \"reason\": \"...\"}\n"
        f"Question: {question}\n"
        f"Fact: {fact}\n"
        f"Candidate: {candidate_text}\n"
    )
    return _chat_json(cfg, prompt, must_succeed=must_succeed)


def llm_judge_generation_answer(
    cfg: LLMAssistConfig,
    question: str,
    oracle_context: str,
    answer_oracle: str,
    answer_gold: str,
    task_type: str,
    *,
    must_succeed: bool = False,
) -> Dict[str, Any] | None:
    """
    Judge generation result with optional subtype classification.
    LLM 判题并输出 FAIL 子类（GH/GF/GRF）。
    """
    prompt = (
        build_generation_neg_answer_prompt(query=question, oracle_context=oracle_context, answer_oracle=answer_oracle, answer_gold=answer_gold)
        if task_type == "NEG"
        else build_generation_pos_answer_prompt(query=question, oracle_context=oracle_context, answer_oracle=answer_oracle, answer_gold=answer_gold)
    )
    return _chat_json(cfg, prompt, must_succeed=must_succeed)


def llm_judge_generation_comparison(
    cfg: LLMAssistConfig,
    question: str,
    task_type: str,
    answer_gold: str,
    answer_online: str,
    answer_oracle: str,
    oracle_context: str,
    *,
    must_succeed: bool = False,
) -> Dict[str, Any] | None:
    prompt = (
        build_generation_neg_comparison_prompt(
            query=question,
            answer_gold=answer_gold,
            answer_online=answer_online,
            answer_oracle=answer_oracle,
            oracle_context=oracle_context,
        )
        if task_type == "NEG"
        else build_generation_pos_comparison_prompt(
            query=question,
            answer_gold=answer_gold,
            answer_online=answer_online,
            answer_oracle=answer_oracle,
            oracle_context=oracle_context,
        )
    )
    return _chat_json(cfg, prompt, must_succeed=must_succeed)


def llm_judge_attribution(
    cfg: LLMAssistConfig,
    task_type: str,
    query: str,
    answer_gold: str,
    enc_summary: Dict[str, Any],
    ret_summary: Dict[str, Any],
    gen_summary: Dict[str, Any],
    *,
    must_succeed: bool = False,
) -> Dict[str, Any] | None:
    prompt = build_attribution_prompt(
        task_type=task_type,
        query=query,
        answer_gold=answer_gold,
        enc_summary=enc_summary,
        ret_summary=ret_summary,
        gen_summary=gen_summary,
    )
    return _chat_json(cfg, prompt, must_succeed=must_succeed)


def llm_judge_answer_correctness(
    cfg: LLMAssistConfig,
    *,
    task_type: str,
    question: str,
    answer_gold: str,
    answer_pred: str,
    judge_mode: str = "online",
    oracle_context: str = "",
    retrieved_context: str = "",
    must_succeed: bool = False,
) -> Dict[str, Any] | None:
    payload = _chat_json(
        cfg,
        build_correctness_judge_prompt(
            task_type=task_type,
            question=question,
            answer_gold=answer_gold,
            answer_pred=answer_pred,
            judge_mode=judge_mode,
            oracle_context=oracle_context,
            retrieved_context=retrieved_context,
        ),
        must_succeed=must_succeed,
    )
    if not isinstance(payload, dict):
        return payload
    label = str(payload.get("label", "")).strip().upper()
    payload["label"] = label
    payload["correct"] = label == "CORRECT"
    payload["judge_mode"] = str(judge_mode or "online")
    for key in ("semantic_match", "temporal_match", "refusal_expected", "refusal_present", "fabricated"):
        payload[key] = bool(payload.get(key, False))
    return payload
