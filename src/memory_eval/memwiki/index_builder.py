from __future__ import annotations

"""
MemWiki v4 中央导航（§六）：``wiki_index.md`` 自动生成。

输出 markdown 包含：

- 头部元数据（sample_id / total_pages / last_updated / 各类型计数）
- ## Entities（按入链频次降序）
- ## Topics（受控词表）
- ## Events（因果链锚点）
- ## Recent Sources（最近 5 个 session）
- ## Active Warnings（多版本机制产生的告警）
- ## Lint Report（可选，由 :class:`MemWikiLint` 注入）
"""

import os
from pathlib import Path

from memory_eval.memwiki.schema import WikiEntry, WikiIndex


class WikiIndexBuilder:
    """渲染 :class:`WikiIndex` 为 markdown，并落盘。"""

    def __init__(self, wiki_index: WikiIndex) -> None:
        self.wiki_index = wiki_index
        self._lint_section: str = ""

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------

    def render_markdown(self) -> str:
        idx = self.wiki_index
        parts: list[str] = []
        parts.append(self._render_header())
        parts.append(self._render_entities())
        parts.append(self._render_topics())
        parts.append(self._render_events())
        parts.append(self._render_recent_sources())
        parts.append(self._render_active_warnings())
        if self._lint_section:
            parts.append(self._lint_section)
        return "\n\n".join(s for s in parts if s.strip())

    def attach_lint_report(self, markdown_section: str) -> None:
        """由 :class:`MemWikiLint` 注入 ``## Lint Report`` 章节文本。"""
        self._lint_section = markdown_section

    # ------------------------------------------------------------------
    # 落盘
    # ------------------------------------------------------------------

    def save(self, output_path: str) -> None:
        text = self.render_markdown()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(text, encoding="utf-8")

    def save_default(self, output_root: str) -> str:
        """落盘到 ``{output_root}/{sample_id}/wiki_index.md``，返回完整路径。"""
        target = Path(output_root) / self.wiki_index.sample_id / "wiki_index.md"
        self.save(str(target))
        return str(target)

    # ==================================================================
    # 渲染子步骤
    # ==================================================================

    def _render_header(self) -> str:
        idx = self.wiki_index
        counts = {pt: len(idx.entries_by_type.get(pt, [])) for pt in
                  ("source", "entity", "topic", "event", "analysis")}
        lines = [
            f"# MemWiki Index — sample_id={idx.sample_id}",
            f"last_updated_session: {idx.last_updated_session}",
            f"total_pages: {len(idx.entries)}",
            f"total_records: {idx.total_records}",
            f"total_source_pages: {counts['source']}",
            f"total_entity_pages: {counts['entity']}",
            f"total_topic_pages: {counts['topic']}",
            f"total_event_pages: {counts['event']}",
            f"total_analysis_pages: {counts['analysis']}",
        ]
        return "\n".join(lines)

    def _render_entities(self) -> str:
        entries = self.wiki_index.list_by_type("entity")
        if not entries:
            return ""
        # 按入链频次降序
        ranked = sorted(
            entries,
            key=lambda e: len(self.wiki_index.backlinks.get(e.entry_id, [])),
            reverse=True,
        )
        lines = ["## Entities (Top by reference count)"]
        for e in ranked:
            ref_count = len(self.wiki_index.backlinks.get(e.entry_id, []))
            cur = e.current_version
            state = (cur.content[:80] if cur and cur.content else "").replace("\n", " ")
            lines.append(f"- [[{e.entry_id}]] — {ref_count} refs — {state}")
        return "\n".join(lines)

    def _render_topics(self) -> str:
        entries = self.wiki_index.list_by_type("topic")
        if not entries:
            return ""
        lines = ["## Topics"]
        for e in entries:
            ref_count = len(self.wiki_index.backlinks.get(e.entry_id, []))
            lines.append(
                f"- [[{e.entry_id}]] — {ref_count} sources — last update session {e.last_updated_session}"
            )
        return "\n".join(lines)

    def _render_events(self) -> str:
        entries = self.wiki_index.list_by_type("event")
        if not entries:
            return ""
        lines = ["## Events (Causal chain anchors)"]
        for e in entries:
            ref_count = len(self.wiki_index.backlinks.get(e.entry_id, []))
            lines.append(
                f"- [[{e.entry_id}]] — session {e.last_updated_session}, evidence: {ref_count} sources"
            )
        return "\n".join(lines)

    def _render_recent_sources(self) -> str:
        sources = self.wiki_index.list_by_type("source")
        if not sources:
            return ""
        # last 5 session
        latest_session = max((s.last_updated_session for s in sources), default=0)
        cutoff = max(0, latest_session - 4)
        recent = [s for s in sources if s.last_updated_session >= cutoff]
        recent.sort(key=lambda s: s.last_updated_session, reverse=True)
        lines = [f"## Recent Sources (sessions {cutoff}-{latest_session})"]
        # 按 session 聚合显示数量
        by_session: dict[int, int] = {}
        for s in recent:
            by_session[s.last_updated_session] = by_session.get(s.last_updated_session, 0) + 1
        for sess in sorted(by_session.keys(), reverse=True):
            lines.append(f"- session {sess}: {by_session[sess]} records")
        return "\n".join(lines)

    def _render_active_warnings(self) -> str:
        warnings: list[str] = []
        for entry in self.wiki_index.entries.values():
            for w in entry.warnings:
                warnings.append(f"- {entry.entry_id}: {w}")
        if not warnings:
            return ""
        return "## Active Warnings\n" + "\n".join(warnings)


__all__ = ["WikiIndexBuilder"]
