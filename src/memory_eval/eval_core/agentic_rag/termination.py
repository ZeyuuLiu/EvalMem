from __future__ import annotations

"""
Bounded convergence control for the encoding-layer memory RAG probe.

The encoding probe cannot assume every F_key exists in the raw memory store.
Coverage is therefore kept as a diagnostic signal, while the loop stops by
bounded retrieval growth and round-to-round convergence.
"""

import re
from dataclasses import dataclass, field

from memory_eval.eval_core.agentic_rag.retrieval import Candidate


_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class RoundConvergence:
    """Similarity diagnostics between the current and previous retrieval rounds."""

    is_stable: bool
    jaccard: float
    new_ratio: float
    current_size: int
    previous_size: int
    new_ids: list[str] = field(default_factory=list)
    reason: str = ""


def candidate_stable_id(candidate: Candidate) -> str:
    """Return a deterministic identity for round convergence and pool dedup."""
    cid = str(candidate.id or "").strip()
    if cid:
        return f"id:{cid}"
    text = _SPACE_RE.sub(" ", str(candidate.text or "").strip().lower())
    return f"text:{text[:240]}"


def candidate_id_set(candidates: list[Candidate]) -> set[str]:
    """Return stable ids for candidates with non-empty identity."""
    return {cid for cid in (candidate_stable_id(c) for c in candidates) if cid not in {"id:", "text:"}}


def compare_rounds(
    current: list[Candidate],
    previous_ids: set[str],
    *,
    jaccard_threshold: float,
    epsilon_new_ratio: float,
) -> RoundConvergence:
    """Check whether the current round is effectively the same as the previous one."""
    current_ids = candidate_id_set(current)
    if not previous_ids:
        return RoundConvergence(
            is_stable=False,
            jaccard=0.0,
            new_ratio=1.0 if current_ids else 0.0,
            current_size=len(current_ids),
            previous_size=0,
            new_ids=sorted(current_ids),
            reason="first_round",
        )

    union = current_ids | previous_ids
    inter = current_ids & previous_ids
    jaccard = (len(inter) / len(union)) if union else 1.0
    new_ids = current_ids - previous_ids
    new_ratio = (len(new_ids) / len(current_ids)) if current_ids else 0.0
    is_stable = jaccard >= jaccard_threshold or new_ratio <= epsilon_new_ratio
    reason = "stable" if is_stable else "new_evidence_found"
    return RoundConvergence(
        is_stable=is_stable,
        jaccard=float(jaccard),
        new_ratio=float(new_ratio),
        current_size=len(current_ids),
        previous_size=len(previous_ids),
        new_ids=sorted(new_ids),
        reason=reason,
    )


def merge_candidate_pool(pool: dict[str, Candidate], candidates: list[Candidate]) -> dict[str, Candidate]:
    """Merge candidates by stable id, keeping the higher-scored version."""
    merged = dict(pool)
    for candidate in candidates:
        cid = candidate_stable_id(candidate)
        if cid in {"id:", "text:"}:
            continue
        prev = merged.get(cid)
        if prev is None or float(candidate.score) > float(prev.score):
            merged[cid] = candidate
    return merged


def pool_to_ranked_list(pool: dict[str, Candidate], *, max_size: int) -> list[Candidate]:
    """Return a capped score-ranked list from the cumulative candidate pool."""
    ranked = sorted(pool.values(), key=lambda c: float(c.score), reverse=True)
    capped = ranked[: max(1, int(max_size or 1))]
    out: list[Candidate] = []
    for rank, candidate in enumerate(capped):
        out.append(
            Candidate(
                id=candidate.id,
                text=candidate.text,
                score=float(candidate.score),
                source_view=candidate.source_view,
                rank_in_view=rank,
                meta=dict(candidate.meta),
            )
        )
    return out


__all__ = [
    "RoundConvergence",
    "candidate_id_set",
    "candidate_stable_id",
    "compare_rounds",
    "merge_candidate_pool",
    "pool_to_ranked_list",
]
