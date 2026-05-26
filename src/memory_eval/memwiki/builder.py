from __future__ import annotations

"""
MemWiki v4 离线构建：把单 sample 的全量记忆库转写为 5 类页面。

入口 :meth:`MemWikiBuilder.build` 的总体流程（v4 §3.2）：

1. ``adapter.export_full_memory(run_ctx)`` 拉取 raw records；
2. 对每条 record :meth:`wikify_source` → source 页（含三层鲁棒兜底，v3 §1）；
3. :func:`builder_ops.build_entity_pages` 聚类生成 entity 综合页；
4. :func:`builder_ops.build_topic_pages` 聚类生成 topic 综合页；
5. :func:`builder_ops.build_event_pages` LLM 检测事件 → event 页；
6. :func:`builder_ops.materialize_wikilinks` 计算 wikilink_graph + backlinks；
7. 通过 :class:`WikiIndexBuilder` 落盘 wiki_index.md；
8. 可选触发 :class:`MemWikiLint`。

约束：
- 不接触 ``sample.f_key`` / ``sample.answer_gold`` / ``sample.evidence_*``，
  由 :class:`LeakAuditor` 验证；
- 单 record wikify 受 v3 工程鲁棒性状态机约束（L1/L2/L3 + segment + skip）。

为保持单文件 < 300 行，具体实现集中在 :mod:`builder_ops` 与 :mod:`llm_client`。
"""

from typing import Any

from memory_eval.memwiki import builder_ops, llm_client
from memory_eval.memwiki.aux_record import AuxRecord, wiki_index_to_aux_records
from memory_eval.memwiki.config import MemWikiConfig
from memory_eval.memwiki.normalizer import EntityNormalizer, TimeNormalizer, TopicNormalizer
from memory_eval.memwiki.schema import WikiEntry, WikiIndex
from memory_eval.memwiki.wikilink_graph import WikilinkGraph


class MemWikiBuilder:
    """离线 MemWiki 构建器。一次实例对应一份 :class:`MemWikiConfig`。"""

    def __init__(self, config: MemWikiConfig, llm_cfg: Any | None = None) -> None:
        """
        Args:
            config: :class:`MemWikiConfig` 实例。
            llm_cfg: :class:`memory_eval.eval_core.llm_assist.LLMAssistConfig`
                实例（API key / base_url / model）。``None`` 时跑 dry-run 模式：
                所有 LLM 调用被替换为占位（用于单测 / leak audit）。
        """
        self.config = config
        self.llm_cfg = llm_cfg
        self.entity_norm = EntityNormalizer()
        self.time_norm = TimeNormalizer()
        self.topic_norm = TopicNormalizer(config.topics_vocabulary_path)
        self.wikilink_graph = WikilinkGraph()

        # 监控指标，写入 build 阶段日志（v3 §1.5）
        self.metrics: dict[str, int] = {
            "wikify_success": 0,
            "wikify_degraded": 0,
            "wikify_segmented": 0,
            "wikify_skipped": 0,
            "entity_pages_created": 0,
            "topic_pages_created": 0,
            "event_pages_created": 0,
            "llm_wikify_attempted": 0,
            "llm_wikify_budget_skipped": 0,
            "llm_synthesis_attempted": 0,
            "llm_synthesis_budget_skipped": 0,
        }

    # ==================================================================
    # 主入口
    # ==================================================================

    def build(self, adapter: Any, run_ctx: Any, sample_id: str) -> WikiIndex:
        """构建单 sample 的完整 wiki_index。"""
        records = list(adapter.export_full_memory(run_ctx))
        wiki_index = WikiIndex(sample_id=sample_id, total_records=len(records))

        # Step 1: 离线 alias_map
        self.entity_norm.build_alias_map(records, self.llm_cfg)

        # Step 2: source 页
        source_entries: list[WikiEntry] = []
        for rec in records:
            session = int(rec.get("session_id", 0) or 0)
            entry = self.wikify_source(rec, session=session)
            if entry is None:
                continue
            wiki_index.upsert(entry)
            source_entries.append(entry)

        # Step 3-5: 综合页
        for page in (
            *self.build_entity_pages(source_entries, self.llm_cfg),
            *self.build_topic_pages(source_entries, self.llm_cfg),
            *self.build_event_pages(source_entries, self.llm_cfg),
        ):
            wiki_index.upsert(page)

        # Step 6: 计算 wikilink_graph + backlinks
        builder_ops.materialize_wikilinks(self, wiki_index)

        # Step 7: 落盘 wiki_index.md
        if self.config.write_wiki_index_md:
            from memory_eval.memwiki.index_builder import WikiIndexBuilder

            WikiIndexBuilder(wiki_index).save_default(self.config.output_root)

        # Step 8: optional lint
        if self.config.lint_on_build:
            from memory_eval.memwiki.lint import MemWikiLint

            _ = MemWikiLint(wiki_index, self.wikilink_graph).lint_all()
            # TODO: 把 LintReport 写入 wiki_index.md 的 ## Lint Report 章节

        wiki_index.last_updated_session = max(
            (e.last_updated_session for e in wiki_index.entries.values()), default=0
        )
        return wiki_index

    def build_aux_records(
        self,
        adapter: Any,
        run_ctx: Any,
        sample_id: str,
        *,
        max_records: int | None = None,
    ) -> list[AuxRecord]:
        """
        Build MemWiki as adapter-neutral auxiliary memory records.

        This is the fair self-evolution path: records are injected back into the
        memory system's normal storage/export surface instead of being searched
        by a separate MemWiki retriever at query time.
        """
        wiki_index = self.build(adapter=adapter, run_ctx=run_ctx, sample_id=sample_id)
        return self.to_aux_records(wiki_index, max_records=max_records)

    def to_aux_records(self, wiki_index: WikiIndex, *, max_records: int | None = None) -> list[AuxRecord]:
        """Flatten an existing WikiIndex into retrieval-friendly auxiliary records."""
        return wiki_index_to_aux_records(wiki_index, max_records=max_records)

    # ==================================================================
    # 单 record / 综合页（thin delegations to builder_ops）
    # ==================================================================

    def wikify_source(self, record: dict, session: int) -> WikiEntry | None:
        """
        三层鲁棒兜底 + skip 策略。详见 :mod:`builder_ops`。

        - **Skip**：``len(tokens) < record_skip_min_tokens`` 且无命名实体 → 仅 source_text，``wikify_skipped=True``。
        - **Segment**：``len(tokens) > record_segment_max_tokens`` → 走分段合并路径。
        - **L1**：temp=0.0 调一次 LLM；
        - **L2**：JSON 解析失败 → temp=0.3 + 把上一轮失败输出作为 negative example 重试；
        - **L3**：仍失败 → degraded entry。
        """
        return builder_ops.wikify_source(self, record, session=session)

    def build_entity_pages(self, sources: list[WikiEntry], llm_cfg: Any) -> list[WikiEntry]:
        """按 canonical entity 聚类 sources，达到阈值 → 生成 entity 综合页。"""
        return builder_ops.build_entity_pages(self, sources, llm_cfg)

    def build_topic_pages(self, sources: list[WikiEntry], llm_cfg: Any) -> list[WikiEntry]:
        """按受控主题词聚类 sources，达到阈值 → 生成 topic 综合页。"""
        return builder_ops.build_topic_pages(self, sources, llm_cfg)

    def build_event_pages(self, sources: list[WikiEntry], llm_cfg: Any) -> list[WikiEntry]:
        """LLM 检测多次提及的关键事件，达到 event_min_evidence → 生成 event 页。"""
        return builder_ops.build_event_pages(self, sources, llm_cfg)

    def segment_long_record(
        self, record: dict, max_tokens: int = 4000
    ) -> tuple[WikiEntry, list[WikiEntry]]:
        """滑窗分段 wikify 再合并（v3 §1.2）。"""
        return builder_ops.segment_long_record(self, record, max_tokens=max_tokens)

    # ==================================================================
    # 内部 LLM 调用（包装到 llm_client）
    # ==================================================================

    def _call_llm_wikify(
        self,
        text: str,
        *,
        temperature: float,
        max_tokens: int,
        negative_hint: bool = False,
    ) -> dict | None:
        """Thin delegate to :func:`llm_client.call_wikify`。失败返回 None。"""
        return llm_client.call_wikify(
            text,
            llm_cfg=self.llm_cfg,
            temperature=temperature,
            max_tokens=max_tokens,
            negative_hint=negative_hint,
        )


__all__ = ["MemWikiBuilder"]
