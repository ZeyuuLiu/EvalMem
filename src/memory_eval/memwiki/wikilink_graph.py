from __future__ import annotations

"""
MemWiki v4 typed wikilink 图（§五）。

职责：
- 集中维护 entry_id 之间的 typed wikilink 出向 / 入向（backlinks）
- 提供 1-2 跳关系拓展接口，供 :class:`MemWikiRetriever` 在 RRF 之后调用
- 提供 lint 用的孤立页 / 断裂链路检测

约束：
- 关系类型强制在 :data:`RELATION_TYPES` 中；越界关系自动回退到 ``"mentions"``
- 拓展时受 ``relation_filter`` 限制，避免 GraphRAG 式过宽召回（v4 §5.4）
"""

from collections import deque

from memory_eval.memwiki.schema import RELATION_TYPES_SET, TypedWikilink


class WikilinkGraph:
    """
    出向 / 入向 typed wikilink 图。

    内部表示：
    - ``_outbound[entry_id] -> list[TypedWikilink]``
    - ``_inbound[target_entry_id] -> list[(source_entry_id, relation)]``
    """

    def __init__(self) -> None:
        self._outbound: dict[str, list[TypedWikilink]] = {}
        self._inbound: dict[str, list[tuple[str, str]]] = {}

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def add_wikilink(self, source_entry_id: str, target_entry_id: str, relation: str) -> None:
        """
        新增一条 typed wikilink。

        - 自校验 relation 受控词表，越界时回退到 ``"mentions"``
        - 不去重相同 (source, target, relation) 三元组，由调用方在 upsert
          流程中自行控制
        """
        rel = relation if relation in RELATION_TYPES_SET else "mentions"
        link = TypedWikilink(target=target_entry_id, relation=rel)
        self._outbound.setdefault(source_entry_id, []).append(link)
        self._inbound.setdefault(target_entry_id, []).append((source_entry_id, rel))

    def bulk_load(self, outbound_map: dict[str, list[TypedWikilink]]) -> None:
        """从 :class:`WikiIndex.wikilink_graph` 一次性导入。"""
        for src, links in outbound_map.items():
            for link in links:
                self.add_wikilink(src, link.target, link.relation)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get_neighbors(
        self,
        entry_id: str,
        relation_filter: list[str] | None = None,
    ) -> list[TypedWikilink]:
        """获取出向邻居，可按 relation 过滤。"""
        links = self._outbound.get(entry_id, [])
        if not relation_filter:
            return list(links)
        rs = set(relation_filter)
        return [l for l in links if l.relation in rs]

    def get_backlinks(self, entry_id: str) -> list[str]:
        """返回所有指向 entry_id 的源 entry_id 列表（去重）。"""
        seen: set[str] = set()
        out: list[str] = []
        for src, _ in self._inbound.get(entry_id, []):
            if src not in seen:
                seen.add(src)
                out.append(src)
        return out

    # ------------------------------------------------------------------
    # 关系拓展（检索 hook）
    # ------------------------------------------------------------------

    def expand(
        self,
        entry_ids: list[str],
        hops: int = 2,
        relation_filter: list[str] | None = None,
    ) -> set[str]:
        """
        BFS 沿 typed wikilink 拓展 1-2 跳，返回所有可达 entry_id 集合（含种子）。

        v4 §5.4 经验：3 跳以上召回噪声显著上升；2 跳是上限。
        """
        if hops < 1:
            return set(entry_ids)
        seen: set[str] = set(entry_ids)
        frontier: deque[tuple[str, int]] = deque((eid, 0) for eid in entry_ids)
        while frontier:
            cur, depth = frontier.popleft()
            if depth >= hops:
                continue
            for link in self.get_neighbors(cur, relation_filter=relation_filter):
                if link.target not in seen:
                    seen.add(link.target)
                    frontier.append((link.target, depth + 1))
        return seen

    # ------------------------------------------------------------------
    # Lint 辅助
    # ------------------------------------------------------------------

    def detect_isolated_pages(self, all_entry_ids: list[str]) -> list[str]:
        """无任何入链的页面（出链不算）。"""
        out: list[str] = []
        for eid in all_entry_ids:
            if not self._inbound.get(eid):
                out.append(eid)
        return out

    def detect_broken_wikilinks(self, all_entry_ids: list[str]) -> list[tuple[str, str]]:
        """target 不存在于 all_entry_ids 中的链路。返回 (source, target) 对列表。"""
        valid = set(all_entry_ids)
        out: list[tuple[str, str]] = []
        for src, links in self._outbound.items():
            for link in links:
                if link.target not in valid:
                    out.append((src, link.target))
        return out

    # ------------------------------------------------------------------
    # 调试 / 序列化
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, list[dict[str, str]]]:
        return {
            src: [{"target": l.target, "relation": l.relation} for l in links]
            for src, links in self._outbound.items()
        }

    def size(self) -> int:
        return sum(len(v) for v in self._outbound.values())


__all__ = ["WikilinkGraph"]
