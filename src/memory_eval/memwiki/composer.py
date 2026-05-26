from __future__ import annotations

"""
MemWiki v4 知识复合（§三 核心）。

当新 record R 到达时：
1. 创建 R 对应的 source 页（与离线 builder 一致的鲁棒兜底）；
2. NER + 主题识别，确定 R 涉及的 entity / topic / event 集合；
3. 在已有 wiki_index 中查找相关页 P；
4. 对每个 P，调用 LLM 在 {rewrite, append, warn, no_change} 中决策；
5. 应用决策：
   - rewrite → 替换 P 的 current_version content
   - append → 仅追加 source_record_id 到 versions[-1]
   - warn → 触发 :class:`VersionManager` 创建新版本 + 附加 warning
   - no_change → 仅记录 source

设计要点：
- ``Composer`` 不维护自己的 LLM 客户端，直接复用 builder.llm_cfg；
- 决策结果保留在 :class:`ComposerResult` 中，供上层日志 / 实验消融。
"""

from dataclasses import dataclass, field

from memory_eval.memwiki.config import MemWikiConfig
from memory_eval.memwiki.schema import (
    AtomicFact,
    TypedWikilink,
    WikiEntry,
    WikiIndex,
    WikiVersion,
)
from memory_eval.memwiki.versioning import VersionManager


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass
class ComposerAction:
    """单页面的决策结果。"""

    action: str                                 # "rewrite" | "append" | "warn" | "no_change"
    reason: str = ""
    rewritten_content: str | None = None
    warning_text: str | None = None
    new_version_content: str | None = None


@dataclass
class ComposerResult:
    """单次 integrate_new_record 的聚合结果。"""

    affected_pages: list[str] = field(default_factory=list)
    actions: dict[str, str] = field(default_factory=dict)       # entry_id -> action
    new_versions_created: int = 0
    warnings_added: int = 0
    new_pages_created: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 主 Composer
# ---------------------------------------------------------------------------


class WikiComposer:
    """
    v4 核心：增量更新时对相关页面执行复合重写。

    使用方式：
    >>> composer = WikiComposer(config=MemWikiConfig(), llm_cfg=llm_cfg)
    >>> result = composer.integrate_new_record(new_record, wiki_index, session=13)
    """

    def __init__(self, config: MemWikiConfig, llm_cfg=None) -> None:
        self.config = config
        self.llm_cfg = llm_cfg
        self.version_manager = VersionManager()

    # ==================================================================
    # 主入口
    # ==================================================================

    def integrate_new_record(
        self,
        new_record: dict,
        wiki_index: WikiIndex,
        session: int,
    ) -> ComposerResult:
        """
        把单条新 record 整合进已有 wiki_index。

        步骤详见模块 docstring。
        """
        result = ComposerResult()

        # Step 1: 创建 source 页（委托 builder 的 wikify 逻辑；这里 stub）
        source_entry = self._wikify_new_source(new_record, session)
        if source_entry is None:
            return result
        wiki_index.upsert(source_entry)
        result.affected_pages.append(source_entry.entry_id)
        result.actions[source_entry.entry_id] = "created"

        # Step 2-3: 识别涉及的实体 / 主题 / 事件 + 查找相关旧页
        related_page_ids = self._find_related_pages(source_entry, wiki_index)

        # Step 4: 对每个相关页 P 执行复合决策
        for page_id in related_page_ids:
            page = wiki_index.get(page_id)
            if page is None:
                continue
            action = self.llm_compose_decision(page, new_record, self.llm_cfg)
            self._apply_action(page, action, new_record, session, result)
            result.affected_pages.append(page_id)
            result.actions[page_id] = action.action

        # Step 5: 触发新页面创建（达到阈值的新 entity/topic/event）
        # TODO: 在 wiki_index 上累计计数，本次新增 source 后若超过
        #       entity_min_occurrences / topic_min_occurrences 阈值则触发新建

        # Step 6: 更新出向 wikilink_graph / backlinks
        # TODO: 调用 builder._materialize_wikilinks 或者增量 add_wikilink

        return result

    # ==================================================================
    # 决策与应用
    # ==================================================================

    def llm_compose_decision(
        self,
        old_page: WikiEntry,
        new_record: dict,
        llm_cfg=None,
    ) -> ComposerAction:
        """
        让 LLM 在 {rewrite, append, warn, no_change} 中作决策。

        失败兜底：LLM 调用失败 / JSON 解析失败 → 返回 no_change，避免错误重写。
        """
        # TODO: 构造 prompt = build_compose_decision_prompt(old_page, new_record)
        # TODO: 调用 _chat_json，解析返回的 JSON
        # TODO: 校验 action 合法，否则回退 no_change
        return ComposerAction(action="no_change", reason="stub")

    def _apply_action(
        self,
        page: WikiEntry,
        action: ComposerAction,
        new_record: dict,
        session: int,
        result: ComposerResult,
    ) -> None:
        record_id = str(new_record.get("id", ""))
        if action.action == "rewrite":
            self.apply_rewrite(page, action.rewritten_content or "", record_id)
            page.last_updated_session = session
        elif action.action == "append":
            self.apply_append(page, record_id)
            page.last_updated_session = session
        elif action.action == "warn":
            self.apply_warn(page, action.warning_text or "", record_id)
            new_ver = self.version_manager.create_new_version(
                page=page,
                new_content=action.new_version_content or page.current_content,
                session=session,
                new_source_record_ids=[record_id],
            )
            result.new_versions_created += 1
            result.warnings_added += 1
        # no_change: 不动 page；source 已经 upsert 进 index

    # ------------------------------------------------------------------
    # 三种写入动作
    # ------------------------------------------------------------------

    def apply_rewrite(
        self,
        page: WikiEntry,
        new_content: str,
        source_record_id: str,
    ) -> WikiEntry:
        """覆盖 current_version.content，附加 source_record_id。"""
        cur = page.current_version
        if cur is None:
            page.versions.append(
                WikiVersion(
                    version_id=f"{page.entry_id}::v1",
                    valid_from_session=page.last_updated_session,
                    valid_to_session=None,
                    content=new_content,
                    source_record_ids=[source_record_id] if source_record_id else [],
                )
            )
        else:
            cur.content = new_content
            if source_record_id and source_record_id not in cur.source_record_ids:
                cur.source_record_ids.append(source_record_id)
        # 自动新增 evidence_in wikilink，便于 retriever 拉回 source 页
        if source_record_id:
            page.wikilinks.append(
                TypedWikilink(target=f"sources/{source_record_id}", relation="evidence_in")
            )
        return page

    def apply_append(self, page: WikiEntry, source_record_id: str) -> WikiEntry:
        """仅追加 source 引用，不改正文。"""
        cur = page.current_version
        if cur is not None and source_record_id and source_record_id not in cur.source_record_ids:
            cur.source_record_ids.append(source_record_id)
        if source_record_id:
            page.wikilinks.append(
                TypedWikilink(target=f"sources/{source_record_id}", relation="evidence_in")
            )
        return page

    def apply_warn(
        self,
        page: WikiEntry,
        conflict_description: str,
        new_record_id: str,
    ) -> WikiEntry:
        """附加 warning 块 + contradicts wikilink。"""
        warning = f"[contradiction] {conflict_description} (evidence: sources/{new_record_id})"
        page.warnings.append(warning)
        if new_record_id:
            page.wikilinks.append(
                TypedWikilink(target=f"sources/{new_record_id}", relation="contradicts")
            )
        return page

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _wikify_new_source(self, new_record: dict, session: int) -> WikiEntry | None:
        """
        增量场景下复用 :class:`MemWikiBuilder.wikify_source` 的逻辑。
        延迟导入避免与 builder 循环依赖。
        """
        from memory_eval.memwiki.builder import MemWikiBuilder

        # TODO: 复用全局 builder 单例，而非每次新建；或抽离 wikify 为独立服务
        builder = MemWikiBuilder(self.config, self.llm_cfg)
        return builder.wikify_source(new_record, session=session)

    def _find_related_pages(self, source_entry: WikiEntry, wiki_index: WikiIndex) -> list[str]:
        """从 source_entry.tags 中拉出涉及的 entity / topic id，找到已存在的页面。"""
        out: list[str] = []
        for ent in source_entry.tags.get("entities", []) or []:
            ent_id = f"entities/{ent}"
            if ent_id in wiki_index.entries:
                out.append(ent_id)
        for topic in source_entry.tags.get("topics", []) or []:
            topic_id = f"topics/{topic}"
            if topic_id in wiki_index.entries:
                out.append(topic_id)
        # TODO: event 页关联检测（基于 source.atomic_facts 与 event.atomic_facts 的 SPO 重叠）
        return out


__all__ = ["WikiComposer", "ComposerResult", "ComposerAction"]
