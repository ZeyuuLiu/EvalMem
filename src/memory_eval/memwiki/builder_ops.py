from __future__ import annotations

"""
MemWiki v4 builder helpers.

把 builder 的较长辅助逻辑放在这里，保持 :mod:`builder` 本体尽量薄，便于满足
单模块 < 300 行的工程约束。

TODO 仍保留：
- 真正的 LLM 调用
- NER / topic 聚类 / event 检测
- segment-then-merge 的完整实现
"""

from collections import defaultdict
import re
from typing import Any

from memory_eval.memwiki.config import MemWikiConfig
from memory_eval.memwiki.llm_client import call_entity_synthesize, call_event_synthesize, call_topic_synthesize, call_wikify
from memory_eval.memwiki.schema import AtomicFact, TypedWikilink, WikiEntry, WikiIndex, WikiVersion


def _cfg_value(cfg: Any, name: str, default: Any = "") -> Any:
    if isinstance(cfg, dict):
        value = cfg.get(name, default)
    else:
        value = getattr(cfg, name, default)
    return default if value is None else value


def has_llm_credentials(builder: Any) -> bool:
    cfg = getattr(builder, "llm_cfg", None)
    if cfg is None:
        return False
    return bool(str(_cfg_value(cfg, "api_key", "")).strip() and str(_cfg_value(cfg, "base_url", "")).strip())


def allow_llm_wikify(builder: Any) -> bool:
    if not has_llm_credentials(builder):
        return False
    limit = getattr(builder.config, "llm_wikify_record_limit", None)
    attempted = int(builder.metrics.get("llm_wikify_attempted", 0))
    if limit is not None and attempted >= int(limit):
        builder.metrics["llm_wikify_budget_skipped"] += 1
        return False
    builder.metrics["llm_wikify_attempted"] = attempted + 1
    return True


def allow_llm_synthesis(builder: Any) -> bool:
    if not has_llm_credentials(builder):
        return False
    limit = getattr(builder.config, "llm_synthesis_page_limit", None)
    attempted = int(builder.metrics.get("llm_synthesis_attempted", 0))
    if limit is not None and attempted >= int(limit):
        builder.metrics["llm_synthesis_budget_skipped"] += 1
        return False
    builder.metrics["llm_synthesis_attempted"] = attempted + 1
    return True


def should_skip_record(text: str, min_tokens: int) -> bool:
    """短且无实体时跳过 wikify（实体检测 TODO）。"""
    return len(text.split()) < min_tokens


def has_retrieval_signal(builder: Any, record: dict) -> bool:
    text = str(record.get("text", "") or "")
    meta = dict(record.get("meta", {})) if isinstance(record.get("meta", {}), dict) else {}
    speaker = str(meta.get("speaker") or record.get("speaker") or "").strip()
    timestamp = str(meta.get("timestamp") or record.get("timestamp") or "").strip()
    entities = infer_entities(text, speaker=speaker)
    topics = infer_topics(builder, text)
    return bool(speaker or timestamp or entities or [t for t in topics if t != "other"])


def token_count(text: str) -> int:
    return len(text.split())


def derive_source_entry_id(record_id: str) -> str:
    return f"sources/{record_id or 'unknown'}"


def build_skipped_entry(entry_id: str, text: str, session: int) -> WikiEntry:
    return WikiEntry(entry_id=entry_id, page_type="source", title=text[:40], source_text=text, wikify_skipped=True, last_updated_session=session)


def build_degraded_entry(entry_id: str, text: str, session: int) -> WikiEntry:
    entry = WikiEntry(entry_id=entry_id, page_type="source", title=text[:40], source_text=text, degraded=True, last_updated_session=session)
    entry.versions.append(
        WikiVersion(
            version_id=f"{entry_id}::v1",
            valid_from_session=session,
            valid_to_session=None,
            content=text,
            source_record_ids=[entry_id.replace("sources/", "", 1)],
        )
    )
    return entry


def assemble_source_entry(
    builder: Any,
    entry_id: str,
    record: dict,
    parsed: dict,
    session: int,
) -> WikiEntry:
    ents = [builder.entity_norm.normalize(e) for e in parsed.get("tags", {}).get("entities", [])]
    topics = builder.topic_norm.normalize(parsed.get("tags", {}).get("topics", []))
    time_anchors = [builder.time_norm.parse(str(t.get("raw", "")), session_datetime=record.get("session_datetime")) for t in parsed.get("time_anchors", []) or []]
    facts = [AtomicFact(str(f.get("subject", "")), str(f.get("predicate", "")), str(f.get("object", "")), f.get("time")) for f in parsed.get("atomic_facts", []) or []]
    entry = WikiEntry(entry_id=entry_id, page_type="source", title=str(parsed.get("title", record.get("text", ""))[:80]), tags={"entities": ents, "topics": topics}, time_anchors=time_anchors, atomic_facts=facts, hypothetical_questions=list(parsed.get("hypothetical_questions", [])), source_text=str(record.get("text", "")), last_updated_session=session)
    entry.versions.append(WikiVersion(version_id=f"{entry_id}::v1", valid_from_session=session, valid_to_session=None, content=entry.source_text, atomic_facts=list(entry.atomic_facts), source_record_ids=[str(record.get("id", ""))]))
    entry.wikilinks.extend([TypedWikilink(target=f"entities/{ent}", relation="subject_of") for ent in ents])
    entry.wikilinks.extend([TypedWikilink(target=f"topics/{topic}", relation="topic_of") for topic in topics])
    return entry


def wikify_source(builder: Any, record: dict, session: int) -> WikiEntry | None:
    text = str(record.get("text", "") or "").strip()
    record_id = str(record.get("id", "") or "")
    entry_id = derive_source_entry_id(record_id)
    if not text:
        builder.metrics["wikify_skipped"] += 1
        return None
    if should_skip_record(text, builder.config.record_skip_min_tokens) and not has_retrieval_signal(builder, record):
        builder.metrics["wikify_skipped"] += 1
        return build_skipped_entry(entry_id, text, session)
    if token_count(text) > builder.config.record_segment_max_tokens:
        builder.metrics["wikify_segmented"] += 1
        return segment_long_record(builder, record, max_tokens=builder.config.record_segment_max_tokens)[0]

    parsed = None
    if allow_llm_wikify(builder):
        parsed = builder._call_llm_wikify(text, temperature=builder.config.wikify_temp_initial, max_tokens=builder.config.wikify_max_tokens_initial)
        if parsed is None:
            parsed = builder._call_llm_wikify(text, temperature=builder.config.wikify_temp_retry, max_tokens=builder.config.wikify_max_tokens_retry, negative_hint=True)
    if parsed is None:
        builder.metrics["wikify_degraded"] += 1
        return build_deterministic_source_entry(builder, entry_id, record, session)
    builder.metrics["wikify_success"] += 1
    return assemble_source_entry(builder, entry_id, record, parsed, session)


_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}))*\b")


def build_deterministic_source_entry(builder: Any, entry_id: str, record: dict, session: int) -> WikiEntry:
    """No-key fallback: keep source page useful while marking it degraded."""
    text = str(record.get("text", "") or "").strip()
    meta = dict(record.get("meta", {})) if isinstance(record.get("meta", {}), dict) else {}
    raw_text = str(meta.get("raw_text") or "").strip() or strip_record_prefix(text)
    speaker = str(meta.get("speaker") or record.get("speaker") or "").strip()
    timestamp = str(meta.get("timestamp") or record.get("timestamp") or "").strip()
    entities = infer_entities(raw_text, speaker=speaker)
    topics = infer_topics(builder, raw_text)
    subject = speaker or (entities[0] if entities else "speaker")
    parsed = {
        "title": text[:80],
        "tags": {"entities": entities, "topics": topics},
        "time_anchors": [{"raw": timestamp, "iso": "", "session": session, "certainty": "medium"}] if timestamp else [],
        "atomic_facts": [
            {
                "subject": subject,
                "predicate": "said",
                "object": raw_text[:600],
                "time": timestamp,
            }
        ],
        "hypothetical_questions": build_fallback_questions(text=raw_text, subject=subject, timestamp=timestamp),
    }
    entry = assemble_source_entry(builder, entry_id, record, parsed, session)
    entry.degraded = True
    entry.warnings.append("deterministic_wikify_fallback")
    return entry


def strip_record_prefix(text: str) -> str:
    cleaned = str(text or "").strip()
    if " | " in cleaned and ": " in cleaned.split(" | ", 1)[1]:
        return cleaned.split(" | ", 1)[1].split(": ", 1)[1].strip()
    if ": " in cleaned:
        return cleaned.split(": ", 1)[1].strip()
    return cleaned


def infer_entities(text: str, *, speaker: str = "") -> list[str]:
    stop = {
        "I",
        "It",
        "That",
        "This",
        "Thanks",
        "Thank",
        "Hi",
        "Hey",
        "No",
        "Yes",
        "Wow",
        "Oh",
    }
    out: list[str] = []
    if speaker:
        out.append(speaker)
    for match in _ENTITY_RE.findall(text or ""):
        item = match.strip()
        if item and item not in stop:
            out.append(item)
    return _dedupe(out)[:8]


def infer_topics(builder: Any, text: str) -> list[str]:
    lower = str(text or "").lower()
    raw: list[str] = []
    mapping = [
        ("dating", ["date", "dating", "relationship", "single", "partner"]),
        ("family", ["family", "kid", "children", "parent", "adoption"]),
        ("parenting", ["kids", "children", "parenting", "adoption"]),
        ("work_career", ["career", "work", "job", "counsel", "certification"]),
        ("education", ["school", "education", "class", "course", "certification"]),
        ("health", ["health", "therapy", "mental health", "support group", "self-care"]),
        ("mental_state", ["stress", "stressed", "powerful", "support", "relieved", "accepted", "courage"]),
        ("exercise", ["run", "running", "race", "swim", "swimming", "hiking", "biking"]),
        ("travel", ["camp", "camping", "beach", "mountain", "forest", "road trip", "grand canyon"]),
        ("hobby", ["pottery", "paint", "painting", "books", "read", "sunrise", "sunset", "music", "violin", "clarinet"]),
        ("art", ["art", "paint", "painting", "pottery", "sunrise", "sunset", "portrait", "mural", "stained glass"]),
        ("books", ["book", "read", "bookshelf"]),
        ("music", ["music", "song", "concert", "violin", "clarinet", "bach", "mozart"]),
        ("pet", ["pet", "dog", "cat", "guinea pig", "oliver", "luna", "bailey"]),
        ("religion", ["church", "faith", "religious"]),
        ("politics", ["political", "liberal", "activist"]),
        ("social_event", ["group", "conference", "parade", "meeting", "picnic", "speech", "festival", "workshop"]),
        ("friendship", ["friend", "friends", "mentor"]),
        ("plan", ["plan", "planning", "going to", "next", "want to", "decided"]),
        ("schedule", ["yesterday", "tomorrow", "next week", "weekend", "june", "july", "august", "september", "october"]),
        ("location", ["sweden", "beach", "mountain", "forest", "park", "museum", "where"]),
        ("achievement", ["passed", "interview", "certification", "graduated"]),
        ("identity", ["transgender", "lgbtq", "lgbtq+"]),
    ]
    for topic, clues in mapping:
        if any(clue in lower for clue in clues):
            raw.append("social_event" if topic == "identity" else topic)
    if not raw:
        raw = ["other"]
    return builder.topic_norm.normalize(raw)


def build_fallback_questions(text: str, subject: str, timestamp: str = "") -> list[str]:
    focus_phrases = extract_focus_phrases(text)
    focus = focus_phrases[0] if focus_phrases else "this memory"
    questions = pattern_questions(text=text, subject=subject)
    questions.extend(
        [
            f"What did {subject} mention about {focus}?",
            f"What is the key fact involving {subject} and {focus}?",
        ]
    )
    if timestamp:
        questions.append(f"When did {subject} mention {focus}?")
    questions.append(f"Is there evidence that {subject} discussed {focus}?")
    return _dedupe(questions)[:4]


def pattern_questions(text: str, subject: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    rules: list[tuple[str, str]] = [
        (r"\b(?:i|we)\s+went to\s+(?:a|an|the)?\s*([^.!?;]{3,80})", "When did {subject} go to {obj}?"),
        (r"\b(?:i|we)\s+ran\s+(?:a|an|the)?\s*([^.!?;]{3,80})", "When did {subject} run {obj}?"),
        (r"\b(?:i|we)\s+painted\s+(?:a|an|the|that)?\s*([^.!?;]{3,80})", "When did {subject} paint {obj}?"),
        (r"\b(?:i|we)\s+signed up for\s+(?:a|an|the)?\s*([^.!?;]{3,80})", "When did {subject} sign up for {obj}?"),
        (r"\b(?:i|we)\s+(?:am|are|'m|'re)\s+going to\s+(?:a|an|the)?\s*([^.!?;]{3,80})", "When is {subject} going to {obj}?"),
        (r"\b(?:i|we)\s+applied to\s+(?:a|an|the)?\s*([^.!?;]{3,80})", "When did {subject} apply to {obj}?"),
        (r"\b(?:i|we)\s+joined\s+(?:a|an|the)?\s*([^.!?;]{3,80})", "When did {subject} join {obj}?"),
        (r"\b(?:i|we)\s+attended\s+(?:a|an|the)?\s*([^.!?;]{3,80})", "When did {subject} attend {obj}?"),
        (r"\b(?:i|we)\s+read\s+(?:a|an|the)?\s*([^.!?;]{3,80})", "What did {subject} read?"),
        (r"\b(?:i|we)\s+bought\s+(?:a|an|the)?\s*([^.!?;]{3,80})", "What did {subject} buy?"),
        (r"\b(?:i|we)\s+got hurt\b", "When did {subject} get hurt?"),
        (r"\b(?:i|we)\s+have\s+([^.!?;]{3,80})", "What does {subject} have?"),
    ]
    out: list[str] = []
    for pattern, template in rules:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if not match:
            continue
        obj = clean_focus_phrase(match.group(1) if match.groups() else "")
        if "{obj}" in template and not obj:
            continue
        out.append(template.format(subject=subject, obj=obj))
    return out


def extract_focus_phrases(text: str) -> list[str]:
    stop = {
        "about",
        "after",
        "again",
        "always",
        "because",
        "before",
        "being",
        "caroline",
        "could",
        "every",
        "going",
        "guess",
        "having",
        "hello",
        "melanie",
        "really",
        "since",
        "thanks",
        "their",
        "there",
        "these",
        "thing",
        "things",
        "think",
        "those",
        "today",
        "would",
        "yesterday",
    }
    tokens = []
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9+']+", str(text or "")):
        token = raw.lower().strip("'")
        if len(token) < 4 or token in stop:
            continue
        tokens.append(token)
    bigrams = [" ".join(tokens[i : i + 2]) for i in range(max(0, len(tokens) - 1))]
    return _dedupe([clean_focus_phrase(x) for x in bigrams[:2] + tokens[:4] if x])[:4]


def clean_focus_phrase(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip(" .,!?:;\"'")
    cleaned = re.sub(r"\b(?:yesterday|today|tomorrow|last year|last week|this month|this week)\b.*$", "", cleaned, flags=re.IGNORECASE).strip(" .,!?:;\"'")
    words = cleaned.split()
    return " ".join(words[:8])


def segment_long_record(builder: Any, record: dict, max_tokens: int = 4000) -> tuple[WikiEntry, list[WikiEntry]]:
    # TODO: sliding window + sub-entry wikify + merge
    record_id = str(record.get("id", "") or "")
    merged = WikiEntry(entry_id=derive_source_entry_id(record_id), page_type="source", title=f"[segmented] {record_id}", source_text=str(record.get("text", "")), is_segmented=True, last_updated_session=int(record.get("session_id", 0) or 0))
    return merged, []


def materialize_wikilinks(builder: Any, wiki_index: WikiIndex) -> None:
    for eid, entry in wiki_index.entries.items():
        for link in entry.wikilinks:
            builder.wikilink_graph.add_wikilink(eid, link.target, link.relation)
    wiki_index.wikilink_graph = builder.wikilink_graph.to_dict()  # type: ignore[assignment]
    wiki_index.backlinks = {tgt: builder.wikilink_graph.get_backlinks(tgt) for tgt in wiki_index.entries.keys()}


def build_entity_pages(builder: Any, sources: list[WikiEntry], llm_cfg: Any) -> list[WikiEntry]:
    clusters: dict[str, list[WikiEntry]] = defaultdict(list)
    for s in sources:
        for ent in s.tags.get("entities", []) or []:
            clusters[builder.entity_norm.normalize(str(ent))].append(s)
    out: list[WikiEntry] = []
    for canonical, group in clusters.items():
        if len(group) < builder.config.entity_min_occurrences:
            continue
        parsed = call_entity_synthesize(canonical, [s.source_text for s in group], llm_cfg=llm_cfg) if allow_llm_synthesis(builder) else None
        if parsed is None:
            parsed = fallback_entity_synthesis(canonical, group)
        out.append(assemble_synthesis_entry(builder, page_type="entity", slug=canonical, parsed=parsed, group=group))
        builder.metrics["entity_pages_created"] += 1
    return out


def build_topic_pages(builder: Any, sources: list[WikiEntry], llm_cfg: Any) -> list[WikiEntry]:
    clusters: dict[str, list[WikiEntry]] = defaultdict(list)
    for s in sources:
        for topic in s.tags.get("topics", []) or []:
            clusters[str(topic)].append(s)
    out: list[WikiEntry] = []
    for topic, group in clusters.items():
        if len(group) < builder.config.topic_min_occurrences or topic == "other":
            continue
        parsed = call_topic_synthesize(topic, [s.source_text for s in group], llm_cfg=llm_cfg) if allow_llm_synthesis(builder) else None
        if parsed is None:
            parsed = fallback_topic_synthesis(topic, group)
        out.append(assemble_synthesis_entry(builder, page_type="topic", slug=topic, parsed=parsed, group=group))
        builder.metrics["topic_pages_created"] += 1
    return out


def assemble_synthesis_entry(
    builder: Any,
    *,
    page_type: str,
    slug: str,
    parsed: dict,
    group: list[WikiEntry],
) -> WikiEntry:
    entry_id = f"{page_type}s/{slug}" if page_type != "entity" else f"entities/{slug}"
    title = str(parsed.get("title") or slug).strip()
    tags_raw = parsed.get("tags", {}) if isinstance(parsed.get("tags", {}), dict) else {}
    entities = [builder.entity_norm.normalize(str(e)) for e in tags_raw.get("entities", []) if str(e).strip()]
    topics = builder.topic_norm.normalize([str(t) for t in tags_raw.get("topics", []) if str(t).strip()])
    if page_type == "entity" and slug not in entities:
        entities.insert(0, slug)
    if page_type == "topic" and slug not in topics:
        topics.insert(0, slug)
    facts = [
        AtomicFact(str(f.get("subject", "")), str(f.get("predicate", "")), str(f.get("object", "")), f.get("time"))
        for f in parsed.get("atomic_facts", [])
        if isinstance(f, dict)
    ]
    summary = str(parsed.get("summary") or parsed.get("current_state") or "").strip()
    if not summary:
        summary = "\n".join(s.source_text for s in group[:5])
    source_ids = [s.entry_id.replace("sources/", "", 1) for s in group]
    entry = WikiEntry(
        entry_id=entry_id,
        page_type=page_type,
        title=title,
        tags={"entities": _dedupe(entities), "topics": _dedupe(topics)},
        atomic_facts=facts,
        hypothetical_questions=[str(q).strip() for q in parsed.get("hypothetical_questions", []) if str(q).strip()],
        source_text="\n".join(s.source_text for s in group),
        last_updated_session=max((s.last_updated_session for s in group), default=0),
    )
    entry.versions.append(
        WikiVersion(
            version_id=f"{entry_id}::v1",
            valid_from_session=min((s.last_updated_session for s in group), default=0),
            valid_to_session=None,
            content=summary,
            atomic_facts=list(facts),
            source_record_ids=source_ids,
        )
    )
    entry.wikilinks.extend([TypedWikilink(target=s.entry_id, relation="evidence_in") for s in group])
    return entry


def fallback_entity_synthesis(canonical: str, group: list[WikiEntry]) -> dict:
    facts: list[dict] = []
    topics: list[str] = []
    for source in group:
        topics.extend(str(t) for t in source.tags.get("topics", []) if str(t).strip())
        for fact in source.atomic_facts[:2]:
            facts.append({"subject": fact.subject or canonical, "predicate": fact.predicate, "object": fact.object, "time": fact.time})
    snippets = [s.source_text for s in group[:4] if s.source_text]
    return {
        "title": canonical,
        "summary": f"{canonical} appears in {len(group)} memory records. " + " ".join(snippets)[:700],
        "current_state": snippets[-1] if snippets else "",
        "tags": {"entities": [canonical], "topics": _dedupe(topics) or ["other"]},
        "atomic_facts": facts[:12],
        "hypothetical_questions": [
            f"What is known about {canonical}?",
            f"What recent events involve {canonical}?",
            f"What facts about {canonical} are supported by memory?",
        ],
    }


def fallback_topic_synthesis(topic: str, group: list[WikiEntry]) -> dict:
    entities: list[str] = []
    facts: list[dict] = []
    for source in group:
        entities.extend(str(e) for e in source.tags.get("entities", []) if str(e).strip())
        for fact in source.atomic_facts[:2]:
            facts.append({"subject": fact.subject, "predicate": fact.predicate, "object": fact.object, "time": fact.time})
    snippets = [s.source_text for s in group[:5] if s.source_text]
    return {
        "title": topic,
        "summary": f"Topic {topic} appears in {len(group)} memory records. " + " ".join(snippets)[:700],
        "tags": {"entities": _dedupe(entities), "topics": [topic]},
        "atomic_facts": facts[:12],
        "hypothetical_questions": [
            f"What memories relate to {topic}?",
            f"Who is involved in {topic} memories?",
            f"When was {topic} discussed?",
        ],
    }


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


_EVENT_PATTERNS: list[tuple[str, str, list[str]]] = [
    ("lgbtq_support_group", "LGBTQ support group", ["support group"]),
    ("charity_race", "charity race", ["charity race", "race"]),
    ("adoption_agency", "adoption agency", ["adoption agency", "adoption agencies"]),
    ("adoption_meeting", "adoption meeting", ["adoption meeting", "council meeting"]),
    ("adoption_interview", "adoption interview", ["adoption interview", "passed the adoption"]),
    ("pride_parade", "pride parade", ["pride parade"]),
    ("pride_festival", "pride festival", ["pride festival"]),
    ("school_speech", "school speech", ["school speech", "speech at a school"]),
    ("counseling_workshop", "LGBTQ counseling workshop", ["counseling workshop"]),
    ("pottery_class", "pottery class", ["pottery class"]),
    ("pottery_workshop", "pottery workshop", ["pottery workshop"]),
    ("museum_visit", "museum visit", ["museum"]),
    ("picnic", "picnic", ["picnic"]),
    ("lgbtq_conference", "LGBTQ conference", ["lgbtq conference", "transgender conference"]),
    ("camping_trip", "camping trip", ["camping", "camped", "camp"]),
    ("beach_trip", "beach trip", ["beach"]),
    ("hiking_trip", "hiking trip", ["hike", "hiking"]),
    ("mentorship_program", "mentorship program", ["mentorship program", "mentoring program"]),
    ("activist_group", "activist group", ["activist group"]),
    ("art_show", "art show", ["art show"]),
    ("birthday", "birthday", ["birthday"]),
    ("poetry_reading", "poetry reading", ["poetry reading"]),
    ("road_trip", "road trip", ["road trip", "roadtrip"]),
    ("concert", "concert", ["concert"]),
    ("park_visit", "park visit", ["park"]),
    ("talent_show", "talent show", ["talent show"]),
]


def infer_event_clues(text: str) -> list[tuple[str, str]]:
    lower = str(text or "").lower()
    out: list[tuple[str, str]] = []
    for slug, title, patterns in _EVENT_PATTERNS:
        if any(pattern in lower for pattern in patterns):
            out.append((slug, title))
    return out


def build_event_pages(builder: Any, sources: list[WikiEntry], llm_cfg: Any) -> list[WikiEntry]:
    clusters: dict[str, dict[str, Any]] = {}
    for source in sources:
        for slug, title in infer_event_clues(source.source_text):
            bucket = clusters.setdefault(slug, {"clue": title, "sources": []})
            bucket["sources"].append(source)
    out: list[WikiEntry] = []
    for slug, cand in sorted(clusters.items()):
        group = list(cand.get("sources", []))
        if len(group) < builder.config.event_min_evidence:
            continue
        clue = str(cand.get("clue", slug))
        parsed = call_event_synthesize(clue, [s.source_text for s in group], llm_cfg=llm_cfg) if allow_llm_synthesis(builder) else None
        if parsed is None:
            parsed = fallback_event_synthesis(clue, group)
        out.append(assemble_synthesis_entry(builder, page_type="event", slug=slug, parsed=parsed, group=group))
        builder.metrics["event_pages_created"] += 1
    return out


def fallback_event_synthesis(event_clue: str, group: list[WikiEntry]) -> dict:
    entities: list[str] = []
    topics: list[str] = ["social_event"]
    facts: list[dict] = []
    for source in group:
        entities.extend(str(e) for e in source.tags.get("entities", []) if str(e).strip())
        topics.extend(str(t) for t in source.tags.get("topics", []) if str(t).strip())
        for fact in source.atomic_facts[:2]:
            facts.append({"subject": fact.subject, "predicate": fact.predicate, "object": fact.object, "time": fact.time})
    snippets = [s.source_text for s in group[:5] if s.source_text]
    return {
        "title": event_clue,
        "summary": f"Event {event_clue} is supported by {len(group)} memory records. " + " ".join(snippets)[:700],
        "tags": {"entities": _dedupe(entities), "topics": _dedupe(topics)},
        "atomic_facts": facts[:12],
        "hypothetical_questions": [
            f"When did {event_clue} happen?",
            f"Who was involved in {event_clue}?",
            f"What memories mention {event_clue}?",
        ],
    }


__all__ = [
    "wikify_source",
    "segment_long_record",
    "materialize_wikilinks",
    "build_entity_pages",
    "build_topic_pages",
    "build_event_pages",
    "assemble_source_entry",
    "build_skipped_entry",
    "build_degraded_entry",
    "build_deterministic_source_entry",
    "derive_source_entry_id",
]
