from __future__ import annotations

"""MemWiki v4 package exports."""

from memory_eval.memwiki.aux_record import AuxRecord, normalize_aux_records, wiki_index_to_aux_records
from memory_eval.memwiki.builder import MemWikiBuilder
from memory_eval.memwiki.composer import ComposerAction, ComposerResult, WikiComposer
from memory_eval.memwiki.config import MemWikiConfig
from memory_eval.memwiki.index_builder import WikiIndexBuilder
from memory_eval.memwiki.injector import InjectionReport, MemWikiInjector
from memory_eval.memwiki.leak_auditor import AuditResult, LeakAuditor
from memory_eval.memwiki.lint import ContradictionIssue, LintReport, MemWikiLint
from memory_eval.memwiki.normalizer import EntityNormalizer, TimeNormalizer, TopicNormalizer
from memory_eval.memwiki.retriever import MemWikiRetriever, ParsedQuery
from memory_eval.memwiki.schema import (
    PAGE_TYPES,
    RELATION_TYPES,
    AtomicFact,
    TimeAnchor,
    TypedWikilink,
    WikiEntry,
    WikiIndex,
    WikiVersion,
)
from memory_eval.memwiki.versioning import TimeQuery, VersionManager
from memory_eval.memwiki.wikilink_graph import WikilinkGraph

__all__ = [
    "MemWikiBuilder",
    "MemWikiInjector",
    "WikiComposer",
    "MemWikiRetriever",
    "MemWikiConfig",
    "WikiIndexBuilder",
    "MemWikiLint",
    "LeakAuditor",
    "VersionManager",
    "WikilinkGraph",
    "EntityNormalizer",
    "TimeNormalizer",
    "TopicNormalizer",
    "ComposerAction",
    "ComposerResult",
    "AuditResult",
    "AuxRecord",
    "InjectionReport",
    "LintReport",
    "ContradictionIssue",
    "ParsedQuery",
    "TimeQuery",
    "PAGE_TYPES",
    "RELATION_TYPES",
    "WikiEntry",
    "WikiIndex",
    "WikiVersion",
    "TypedWikilink",
    "AtomicFact",
    "TimeAnchor",
    "normalize_aux_records",
    "wiki_index_to_aux_records",
]
