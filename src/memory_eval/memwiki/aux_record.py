from __future__ import annotations

"""
Adapter-neutral MemWiki auxiliary records.

MemWiki is used here as an offline, retrieval-friendly index layer.  The main
experiment must still call each memory system's original retrieval path; these
records are therefore materialized as normal memory records that adapters can
ingest into their existing stores or expose through export_full_memory().
"""

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from memory_eval.memwiki.schema import AtomicFact, WikiEntry, WikiIndex


_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-\u4e00-\u9fff]+", flags=re.UNICODE)


@dataclass(frozen=True)
class AuxRecord:
    """A retrieval-oriented MemWiki record that can be ingested by any adapter."""

    id: str
    text: str
    record_type: str
    source_entry_id: str = ""
    source_record_ids: list[str] = field(default_factory=list)
    tags: dict[str, list[str]] = field(default_factory=dict)
    weight: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_memory_record(self) -> dict[str, Any]:
        """Convert to the common adapter memory-record shape."""
        metadata = {
            "source": "memwiki_aux",
            "storage_kind": "memwiki_aux",
            "record_type": self.record_type,
            "source_entry_id": self.source_entry_id,
            "source_record_ids": list(self.source_record_ids),
            "tags": {k: list(v) for k, v in self.tags.items()},
            "weight": float(self.weight),
            **dict(self.meta),
        }
        return {
            "id": self.id,
            "text": self.text,
            "score": float(self.weight),
            "meta": metadata,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_aux_records(records: list[AuxRecord | dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize AuxRecord or memory-record dictionaries into adapter records."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(records or []):
        if isinstance(record, AuxRecord):
            item = record.to_memory_record()
        elif isinstance(record, dict):
            item = {
                "id": str(record.get("id", f"memwiki-aux-{index}")),
                "text": str(record.get("text", "")),
                "score": float(record.get("score", record.get("weight", 1.0)) or 0.0),
                "meta": dict(record.get("meta", {})) if isinstance(record.get("meta", {}), dict) else {},
            }
        else:
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        key = str(item.get("id", "")).strip() or text[:240]
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def wiki_index_to_aux_records(wiki_index: WikiIndex, *, max_records: int | None = None) -> list[AuxRecord]:
    """Flatten a WikiIndex into search-friendly auxiliary records."""
    records: list[AuxRecord] = []
    for entry in wiki_index.entries.values():
        records.extend(entry_to_aux_records(entry))
        if max_records is not None and len(records) >= max_records:
            return records[:max_records]
    return records


def entry_to_aux_records(entry: WikiEntry) -> list[AuxRecord]:
    """Build deterministic aux records from one WikiEntry without using gold labels."""
    out: list[AuxRecord] = []
    content = _entry_content(entry)
    tags = _entry_tags(entry)
    source_ids = _entry_source_ids(entry)
    base_meta = {
        "page_type": entry.page_type,
        "title": entry.title,
        "quality_low": bool(entry.quality_low),
        "degraded": bool(entry.degraded),
        "wikify_skipped": bool(entry.wikify_skipped),
    }

    tag_text = _render_tag_text(entry, tags, content)
    if tag_text:
        out.append(
            AuxRecord(
                id=f"memwiki:{entry.entry_id}:tags",
                text=tag_text,
                record_type="tag_alias_index",
                source_entry_id=entry.entry_id,
                source_record_ids=source_ids,
                tags=tags,
                weight=1.2,
                meta=base_meta,
            )
        )

    for idx, fact in enumerate(entry.atomic_facts[:8]):
        fact_text = _render_fact_text(entry, fact, tags)
        if not fact_text:
            continue
        out.append(
            AuxRecord(
                id=f"memwiki:{entry.entry_id}:fact:{idx}",
                text=fact_text,
                record_type="atomic_fact_index",
                source_entry_id=entry.entry_id,
                source_record_ids=source_ids,
                tags=tags,
                weight=1.3,
                meta=base_meta,
            )
        )

    for idx, question in enumerate(entry.hypothetical_questions[:6]):
        q = str(question or "").strip()
        if not q:
            continue
        out.append(
            AuxRecord(
                id=f"memwiki:{entry.entry_id}:hq:{idx}",
                text=(
                    f"MemWiki hypothetical query: {q}\n"
                    f"Answer evidence page: {entry.title}\n"
                    f"Search tags: {_flat_tags(tags)}\n"
                    f"Evidence: {_trim(content, 500)}"
                ).strip(),
                record_type="hypothetical_question_index",
                source_entry_id=entry.entry_id,
                source_record_ids=source_ids,
                tags=tags,
                weight=1.1,
                meta=base_meta,
            )
        )

    page_text = _render_page_text(entry, tags, content)
    if page_text:
        out.append(
            AuxRecord(
                id=f"memwiki:{entry.entry_id}:page",
                text=page_text,
                record_type="page_summary_index",
                source_entry_id=entry.entry_id,
                source_record_ids=source_ids,
                tags=tags,
                weight=1.0,
                meta=base_meta,
            )
        )
    return out


def _entry_content(entry: WikiEntry) -> str:
    return str(entry.current_content or entry.source_text or "").strip()


def _entry_tags(entry: WikiEntry) -> dict[str, list[str]]:
    tags: dict[str, list[str]] = {}
    for key, values in dict(entry.tags or {}).items():
        cleaned = [str(v).strip() for v in (values or []) if str(v).strip()]
        if cleaned:
            tags[str(key)] = _dedupe(cleaned)
    time_values = [str(t.iso or t.raw).strip() for t in entry.time_anchors if str(t.iso or t.raw).strip()]
    if time_values:
        tags["time"] = _dedupe(time_values)
    if not tags:
        keywords = extract_keywords(_entry_content(entry), limit=12)
        if keywords:
            tags["keywords"] = keywords
    return tags


def _entry_source_ids(entry: WikiEntry) -> list[str]:
    ids: list[str] = []
    for version in entry.versions:
        ids.extend(str(x).strip() for x in version.source_record_ids if str(x).strip())
    if not ids and entry.page_type == "source":
        ids.append(entry.entry_id.replace("sources/", "", 1))
    return _dedupe(ids)


def _render_tag_text(entry: WikiEntry, tags: dict[str, list[str]], content: str) -> str:
    if not tags and not content:
        return ""
    aliases = _flat_tags(tags)
    return (
        f"MemWiki retrieval tags for page {entry.title} ({entry.page_type}).\n"
        f"Aliases and index tags: {aliases}.\n"
        f"Original evidence: {_trim(content, 600)}"
    ).strip()


def _render_fact_text(entry: WikiEntry, fact: AtomicFact, tags: dict[str, list[str]]) -> str:
    subject = str(fact.subject or "").strip()
    predicate = str(fact.predicate or "").strip()
    obj = str(fact.object or "").strip()
    if not (subject or predicate or obj):
        return ""
    when = f" Time: {fact.time}." if fact.time else ""
    return (
        f"MemWiki atomic fact from {entry.title}: {subject} {predicate} {obj}.{when}\n"
        f"Search aliases: {_flat_tags(tags)}"
    ).strip()


def _render_page_text(entry: WikiEntry, tags: dict[str, list[str]], content: str) -> str:
    if not content:
        return ""
    keywords = extract_keywords(content, limit=12)
    return (
        f"MemWiki page summary index. Page: {entry.title}. Type: {entry.page_type}.\n"
        f"Tags: {_flat_tags(tags)}. Keywords: {', '.join(keywords)}.\n"
        f"Content: {_trim(content, 900)}"
    ).strip()


def extract_keywords(text: str, *, limit: int = 12) -> list[str]:
    """Lightweight deterministic keyword extraction for dry-run MemWiki builds."""
    stop = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "about",
        "you",
        "are",
        "was",
        "were",
        "have",
        "has",
        "had",
        "but",
        "not",
        "what",
        "when",
        "where",
        "which",
    }
    counts: dict[str, int] = {}
    for raw in _TOKEN_RE.findall(str(text or "")):
        token = raw.strip().lower()
        if len(token) < 2 or token in stop:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [token for token, _ in ranked[: max(1, int(limit or 1))]]


def _flat_tags(tags: dict[str, list[str]]) -> str:
    parts: list[str] = []
    for key in sorted(tags.keys()):
        vals = [str(v) for v in tags.get(key, []) if str(v).strip()]
        if vals:
            parts.append(f"{key}={', '.join(vals[:12])}")
    return "; ".join(parts)


def _trim(text: str, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    return cleaned[:limit]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


__all__ = [
    "AuxRecord",
    "entry_to_aux_records",
    "extract_keywords",
    "normalize_aux_records",
    "wiki_index_to_aux_records",
]
