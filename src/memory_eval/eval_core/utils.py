from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from memory_eval.eval_core.models import DEFECT_ORDER, EvaluatorConfig


AMBIG_TOKENS: Set[str] = {
    "he",
    "she",
    "it",
    "they",
    "them",
    "this",
    "that",
    "these",
    "those",
    "him",
    "her",
    "his",
    "hers",
    "their",
    "theirs",
    "its",
    "ta",
    "这",
    "那",
    "这个",
    "那个",
    "这些",
    "那些",
    "他",
    "她",
    "它",
    "他们",
    "她们",
    "它们",
}

ABSTAIN_PATTERNS: Sequence[str] = (
    "i don't know",
    "i do not know",
    "not sure",
    "cannot",
    "can't",
    "no information",
    "unknown",
    "not mentioned",
    "n/a",
    "none",
    "不知道",
    "不清楚",
    "无法判断",
    "无法回答",
    "没有提到",
    "未提及",
    "没有相关信息",
)


def normalize_text(text: str) -> str:
    """
    Lightweight normalization used by rule-mode probes.
    探针规则模式的轻量文本归一化。
    """
    s = str(text or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def text_match(a: str, b: str) -> bool:
    """
    Symmetric containment + exact match check.
    文本匹配：精确相等或双向包含。
    """
    na = normalize_text(a)
    nb = normalize_text(b)
    if not na or not nb:
        return False
    return na == nb or na in nb or nb in na


def looks_ambiguous(text: str) -> bool:
    """
    Detect pronoun/deictic-only snippets.
    检测是否为代词/指示词主导的含糊文本。
    """
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", normalize_text(text))
    if not tokens:
        return False
    return all(t in AMBIG_TOKENS for t in tokens)


def ordered_defect_union(*defect_groups: Iterable[str]) -> List[str]:
    """
    Stable defect union to keep output deterministic.
    稳定并集，保证输出顺序一致便于审计。
    """
    seen = set()
    merged: List[str] = []
    for code in DEFECT_ORDER:
        for grp in defect_groups:
            if code in grp and code not in seen:
                seen.add(code)
                merged.append(code)
    for grp in defect_groups:
        for code in grp:
            if code not in seen:
                seen.add(code)
                merged.append(code)
    return merged


def is_abstain(answer: str) -> bool:
    """
    Rule-mode abstain detector for NEG tasks.
    NEG 任务拒答检测（规则模式）。
    """
    a = normalize_text(answer)
    if not a:
        return True
    return any(p in a for p in ABSTAIN_PATTERNS)


def split_tokens(text: str) -> List[str]:
    return [x for x in re.split(r"\W+", normalize_text(text)) if x]


def grounding_overlap(answer: str, context: str) -> Tuple[float, Dict[str, int]]:
    """
    Token overlap ratio used for a simple grounding heuristic.
    用 token 重叠率近似判断 grounded 程度。
    """
    a = set(split_tokens(answer))
    c = set(split_tokens(context))
    if not a or not c:
        return 0.0, {"answer_token_count": len(a), "context_token_count": len(c), "overlap_count": 0}
    overlap = len(a & c)
    return overlap / (len(a) + 1e-6), {
        "answer_token_count": len(a),
        "context_token_count": len(c),
        "overlap_count": overlap,
    }


def rank_and_hit_indices(
    retrieved_items: Sequence[Dict[str, str]],
    f_key: Sequence[str],
) -> Tuple[int, List[int]]:
    """
    Find first hit rank and all hit indices against f_key.
    计算首命中排序位次与全部命中索引。
    """
    hit_indices: List[int] = []
    rank = -1
    for idx, it in enumerate(retrieved_items):
        txt = str(it.get("text", ""))
        matched = any(text_match(f, txt) for f in f_key if f)
        if matched:
            hit_indices.append(idx + 1)
            if rank == -1:
                rank = idx + 1
    return rank, hit_indices


def token_overlap_snr(
    retrieved_items: Sequence[Dict[str, str]],
    f_key: Sequence[str],
) -> Tuple[float, Dict[str, int]]:
    """
    SNR = TokenCount(F_key ∩ C_original) / TokenCount(C_original)
    按 token 交并计算信噪比。
    """
    c_tokens: List[str] = []
    for it in retrieved_items:
        c_tokens.extend(split_tokens(str(it.get("text", ""))))
    f_tokens: List[str] = []
    for f in f_key:
        f_tokens.extend(split_tokens(str(f)))

    c_set = set(c_tokens)
    f_set = set(f_tokens)
    overlap = len(c_set & f_set) if c_set and f_set else 0
    denom = len(c_set)
    if denom <= 0:
        return 0.0, {"c_token_count": 0, "f_token_count": len(f_set), "overlap_count": 0}
    return overlap / denom, {"c_token_count": denom, "f_token_count": len(f_set), "overlap_count": overlap}


def is_strict_llm_probe(cfg: Optional[EvaluatorConfig]) -> bool:
    """LLM-only probe profile: no rule fallbacks; LLM calls must succeed."""
    return bool(
        cfg
        and cfg.use_llm_assist
        and cfg.require_llm_judgement
        and cfg.disable_rule_fallback
    )
