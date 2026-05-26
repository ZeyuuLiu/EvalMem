from __future__ import annotations

"""
MemWiki v4 检索器：4 路 RRF + 关系拓展 + 时间感知（§五）。

检索流程（v4 §5.4）：

    parse_query(query)
        → entities, time_refs, topic_terms
    并行 4 路：
        ├─ tag_lookup        (倒排路径)
        ├─ HQ vector search   (假设问题路径)
        ├─ atomic_facts vector search
        └─ source_text vector search
    rrf_fuse(k=60)
    wikilink_graph.expand(top-5, hops=2, relation_filter=strong)
    time_aware_rerank(time_query)
    return top_k

集成约定（TODO 标记的真实实现）：
- 向量索引由 Qdrant collection（每路一个 collection）支撑
- 倒排索引可用内存 dict 或 SQLite 实现，规模 <150 页时 dict 已足够
- query parse 走 LLM（build_query_parse_prompt）；失败时回退到 spaCy NER
"""

from dataclasses import dataclass, field

from memory_eval.memwiki.config import MemWikiConfig
from memory_eval.memwiki.schema import TimeAnchor, WikiEntry, WikiIndex
from memory_eval.memwiki.versioning import TimeQuery, VersionManager
from memory_eval.memwiki.wikilink_graph import WikilinkGraph


# ---------------------------------------------------------------------------
# 解析结果
# ---------------------------------------------------------------------------


@dataclass
class ParsedQuery:
    raw_query: str
    entities: list[str] = field(default_factory=list)      # 归一化后
    time_refs: list[TimeAnchor] = field(default_factory=list)
    topic_terms: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class MemWikiRetriever:
    """
    与单个 :class:`WikiIndex` 绑定的检索器。

    需要在初始化时把 4 路索引建好（构建期通过 :meth:`prepare_indices` 完成）。
    """

    def __init__(
        self,
        wiki_index: WikiIndex,
        config: MemWikiConfig,
        wikilink_graph: WikilinkGraph | None = None,
        version_manager: VersionManager | None = None,
        llm_cfg=None,
    ) -> None:
        self.wiki_index = wiki_index
        self.config = config
        self.wikilink_graph = wikilink_graph or self._materialize_graph(wiki_index)
        self.version_manager = version_manager or VersionManager()
        self.llm_cfg = llm_cfg

        # 内部索引（懒构建；首次 search 前调 prepare_indices）
        self._tag_inverted: dict[str, set[str]] = {}     # tag_key -> entry_ids
        self._hq_index = None                            # 占位：实际为 Qdrant client / collection
        self._fact_index = None
        self._text_index = None
        self._indices_ready = False

    # ==================================================================
    # 索引准备
    # ==================================================================

    def prepare_indices(self) -> None:
        """构建 4 路索引。在第一次 search 前调用一次即可。"""
        self._build_inverted_index()
        # TODO: 实现以下三路向量索引，建议接 Qdrant：
        # self._hq_index = QdrantClient(...).upsert_collection("hq",  embed=hq_texts)
        # self._fact_index = QdrantClient(...).upsert_collection("facts", embed=fact_texts)
        # self._text_index = QdrantClient(...).upsert_collection("source", embed=source_texts)
        self._indices_ready = True

    def _build_inverted_index(self) -> None:
        index: dict[str, set[str]] = {}
        for entry in self.wiki_index.entries.values():
            for ent in entry.tags.get("entities", []) or []:
                index.setdefault(f"entity::{ent.lower()}", set()).add(entry.entry_id)
            for topic in entry.tags.get("topics", []) or []:
                index.setdefault(f"topic::{topic.lower()}", set()).add(entry.entry_id)
        self._tag_inverted = index

    def _materialize_graph(self, wiki_index: WikiIndex) -> WikilinkGraph:
        g = WikilinkGraph()
        for src_id, entry in wiki_index.entries.items():
            for link in entry.wikilinks:
                g.add_wikilink(src_id, link.target, link.relation)
        return g

    # ==================================================================
    # 主入口
    # ==================================================================

    def search(
        self,
        query: str,
        top_k: int = 10,
        time_query: TimeQuery | None = None,
    ) -> list[WikiEntry]:
        """
        v4 检索主入口。

        Args:
            query: 自然语言查询。
            top_k: 最终返回条目数（截断在拓展 + 重排之后）。
            time_query: 时间锚点过滤；默认 latest。
        """
        if not self._indices_ready:
            self.prepare_indices()

        parsed = self.parse_query(query, self.llm_cfg)

        # 4 路召回
        view_results: dict[str, list[tuple[str, float]]] = {
            "tag": [(eid, 1.0) for eid in self.tag_lookup(parsed)],
            "hq": self.hypothetical_questions_search(query, self.config.retrieval_top_k_per_view),
            "fact": self.atomic_facts_search(query, self.config.retrieval_top_k_per_view),
            "text": self.source_text_search(query, self.config.retrieval_top_k_per_view),
        }

        # RRF 融合
        fused: list[str] = self.rrf_fuse(view_results, k_param=self.config.rrf_k)

        # 关系拓展（取 top-5 作为种子）
        seeds = fused[: max(5, top_k // 2)]
        expanded = self.wikilink_graph.expand(
            seeds,
            hops=self.config.expand_hops,
            relation_filter=self.config.expand_relations,
        )
        # 把扩展进来的 entry 接在 fused 末尾，保留 seed 的优先序
        seed_set = set(fused)
        for eid in expanded:
            if eid not in seed_set:
                fused.append(eid)
                seed_set.add(eid)

        # 时间感知重排 + version 过滤
        reranked = self.time_aware_rerank(fused, time_query or TimeQuery(query_type="latest"))

        out: list[WikiEntry] = []
        for eid in reranked[:top_k]:
            entry = self.wiki_index.get(eid)
            if entry is not None:
                out.append(entry)
        return out

    # ==================================================================
    # 子步骤
    # ==================================================================

    def parse_query(self, query: str, llm_cfg=None) -> ParsedQuery:
        """LLM 解析 → 结构化字段。失败时退化为空 ParsedQuery（4 路仍会跑）。"""
        # TODO: build_query_parse_prompt(query) -> _chat_json -> ParsedQuery
        return ParsedQuery(raw_query=query)

    def tag_lookup(self, parsed: ParsedQuery) -> list[str]:
        """倒排路径：实体名 ∩ 主题词（实体或主题任一命中即可）。"""
        hits: set[str] = set()
        for ent in parsed.entities:
            hits |= self._tag_inverted.get(f"entity::{ent.lower()}", set())
        for topic in parsed.topic_terms:
            hits |= self._tag_inverted.get(f"topic::{topic.lower()}", set())
        return list(hits)

    def hypothetical_questions_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """HQ 向量路径。返回 [(entry_id, score), ...]。"""
        # TODO: query_emb = embed(query) ; results = qdrant.search("hq", query_emb, top_k)
        # quality_low entry 的分数应乘以 hq_quality_score_discount
        return []

    def atomic_facts_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """原子事实路径。"""
        # TODO: 同上
        return []

    def source_text_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """原文兜底路径（degraded / skipped entry 仅走此路）。"""
        # TODO: 同上
        return []

    def rrf_fuse(
        self,
        view_results: dict[str, list],
        k_param: int = 60,
    ) -> list[str]:
        """
        标准 RRF 融合：score(entry) = sum_view 1 / (k_param + rank_in_view)。

        view_results 的 value 既可以是 list[str]（倒排）也可以是
        list[(entry_id, score)]（向量）。
        """
        scores: dict[str, float] = {}
        for view, items in view_results.items():
            for rank, item in enumerate(items):
                entry_id = item[0] if isinstance(item, tuple) else str(item)
                scores[entry_id] = scores.get(entry_id, 0.0) + 1.0 / (k_param + rank + 1)
        return [eid for eid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    def time_aware_rerank(
        self,
        candidates: list[str],
        time_query: TimeQuery | None,
    ) -> list[str]:
        """根据 time_query 过滤 / 重排版本。"""
        if time_query is None or time_query.query_type == "latest":
            # 仅保留没有 deprecated_by 的 entry，以及综合页（entity/topic/event）的 latest 版本
            kept: list[str] = []
            for eid in candidates:
                entry = self.wiki_index.get(eid)
                if entry is None:
                    continue
                if entry.deprecated_by is not None and entry.page_type == "source":
                    continue
                kept.append(eid)
            return kept
        if time_query.query_type == "explicit":
            kept = []
            for eid in candidates:
                entry = self.wiki_index.get(eid)
                if entry is None:
                    continue
                if not entry.versions:
                    kept.append(eid)
                    continue
                ver = self.version_manager.query_by_time(entry, time_query)
                if ver is not None:
                    kept.append(eid)
            return kept
        # fuzzy: 全部保留，由 answer LLM 选择
        return candidates


__all__ = ["MemWikiRetriever", "ParsedQuery"]
