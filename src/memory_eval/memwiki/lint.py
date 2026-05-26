from __future__ import annotations

"""
MemWiki Lint（§七）：检查 wiki 内在健康，而非只做 leakage 审计。

当前 skeleton 实现 5 类检查：
1. orphan_pages：无任何入链
2. broken_wikilinks：target 不存在
3. contradictions：同 SPO 不同 O 且无 warning 标记
4. unregistered：页面存在但未出现在 entries_by_type / wiki_index.md 导航体系
5. outdated_versions：version 时间区间不连续 / 多个 current / 乱序

可扩展 TODO：
- relation / topic 受控词表违规
- analysis 页的 derived_from 完整性
- source-only degraded/skipped 占比阈值告警
"""

from dataclasses import dataclass, field

from memory_eval.memwiki.schema import AtomicFact, WikiEntry, WikiIndex
from memory_eval.memwiki.wikilink_graph import WikilinkGraph


@dataclass
class ContradictionIssue:
    """跨页未标记矛盾。"""

    entry_a: str
    entry_b: str
    subject: str
    predicate: str
    object_a: str
    object_b: str
    reason: str = ""


@dataclass
class LintReport:
    orphan_pages: list[str] = field(default_factory=list)
    broken_wikilinks: list[tuple[str, str]] = field(default_factory=list)
    contradictions: list[ContradictionIssue] = field(default_factory=list)
    unregistered: list[str] = field(default_factory=list)
    outdated_versions: list[str] = field(default_factory=list)

    @property
    def total_issues(self) -> int:
        """总问题数。"""
        return (
            len(self.orphan_pages)
            + len(self.broken_wikilinks)
            + len(self.contradictions)
            + len(self.unregistered)
            + len(self.outdated_versions)
        )

    def render_markdown(self) -> str:
        """渲染为 wiki_index.md 的 ``## Lint Report`` 章节。"""
        lines = [f"## Lint Report\nTotal issues: {self.total_issues}"]
        if self.orphan_pages:
            lines.append("### Orphan pages")
            lines.extend(f"- {x}" for x in self.orphan_pages)
        if self.broken_wikilinks:
            lines.append("### Broken wikilinks")
            lines.extend(f"- {src} -> {tgt}" for src, tgt in self.broken_wikilinks)
        if self.contradictions:
            lines.append("### Contradictions")
            for c in self.contradictions:
                lines.append(
                    f"- {c.entry_a} vs {c.entry_b}: ({c.subject}, {c.predicate}, {c.object_a}) != {c.object_b}"
                )
        if self.unregistered:
            lines.append("### Unregistered pages")
            lines.extend(f"- {x}" for x in self.unregistered)
        if self.outdated_versions:
            lines.append("### Outdated versions")
            lines.extend(f"- {x}" for x in self.outdated_versions)
        return "\n".join(lines)


class MemWikiLint:
    """运行全部 lint checks。"""

    def __init__(self, wiki_index: WikiIndex, wikilink_graph: WikilinkGraph) -> None:
        self.wiki_index = wiki_index
        self.wikilink_graph = wikilink_graph

    def lint_all(self) -> LintReport:
        """依次运行全部 lint checks。"""
        return LintReport(
            orphan_pages=self.find_orphan_pages(),
            broken_wikilinks=self.find_broken_wikilinks(),
            contradictions=self.detect_contradictions(),
            unregistered=self.find_unregistered_pages(),
            outdated_versions=self.find_outdated_versions(),
        )

    def find_orphan_pages(self) -> list[str]:
        """无任何入链页面。"""
        return self.wikilink_graph.detect_isolated_pages(list(self.wiki_index.entries.keys()))

    def find_broken_wikilinks(self) -> list[tuple[str, str]]:
        """target 不存在。"""
        return self.wikilink_graph.detect_broken_wikilinks(list(self.wiki_index.entries.keys()))

    def detect_contradictions(self) -> list[ContradictionIssue]:
        """
        暴力 O(N^2) 扫描跨页 atomic_facts：
        同 S+P 不同 O 且双方 warnings 中未出现 contradiction 标记。

        样本级 wiki 通常 <150 页，足够用简单实现。
        """
        entries = list(self.wiki_index.entries.values())
        out: list[ContradictionIssue] = []
        for i in range(len(entries)):
            a = entries[i]
            for j in range(i + 1, len(entries)):
                b = entries[j]
                for fa in a.atomic_facts:
                    for fb in b.atomic_facts:
                        if fa.subject == fb.subject and fa.predicate == fb.predicate and fa.object != fb.object:
                            if self._is_flagged(a, b):
                                continue
                            out.append(
                                ContradictionIssue(
                                    entry_a=a.entry_id,
                                    entry_b=b.entry_id,
                                    subject=fa.subject,
                                    predicate=fa.predicate,
                                    object_a=fa.object,
                                    object_b=fb.object,
                                    reason="same subject+predicate but different object without warning",
                                )
                            )
        return out

    def find_unregistered_pages(self) -> list[str]:
        """
        存在于 entries，但未出现在 entries_by_type 对应 bucket 中的页面。
        ``wiki_index.md`` 本身不在内存中，这里检查其前置条件。
        """
        registered: set[str] = set()
        for ids in self.wiki_index.entries_by_type.values():
            registered.update(ids)
        return [eid for eid in self.wiki_index.entries.keys() if eid not in registered]

    def find_outdated_versions(self) -> list[str]:
        """版本链不连续 / 多 current / 逆序。返回有问题的 entry_id 列表。"""
        out: list[str] = []
        for entry in self.wiki_index.entries.values():
            if len(entry.versions) <= 1:
                continue
            current_count = sum(1 for v in entry.versions if v.valid_to_session is None)
            if current_count != 1:
                out.append(entry.entry_id)
                continue
            # 按 valid_from 排序检查连续性
            vs = sorted(entry.versions, key=lambda v: v.valid_from_session)
            bad = False
            for i in range(len(vs) - 1):
                if vs[i].valid_to_session != vs[i + 1].valid_from_session:
                    bad = True
                    break
            if bad:
                out.append(entry.entry_id)
        return out

    def _is_flagged(self, a: WikiEntry, b: WikiEntry) -> bool:
        joined = " ".join(a.warnings + b.warnings).lower()
        return "contradiction" in joined or "changed" in joined


__all__ = ["MemWikiLint", "LintReport", "ContradictionIssue"]
