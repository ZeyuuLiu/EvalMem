from __future__ import annotations

"""
MemWiki v4 核心数据结构。

本模块定义 MemWiki 的三层架构中 Wiki 层的所有 dataclass：
- WikiEntry：5 类页面（source / entity / topic / event / analysis）的统一容器
- WikiVersion：多版本保留（valid_from/to_session 时间窗口）
- TypedWikilink：受控关系词表的语义化超链接
- AtomicFact：(subject, predicate, object, time) 四元组
- TimeAnchor：归一化时间锚点（ISO + session + certainty）
- WikiIndex：单 sample 维度的全局索引（entries + graph + backlinks）

设计原则：
1. 字段尽量为 plain dataclass，便于序列化为 JSON / YAML 写盘
2. 不持有 LLM / Embedding 句柄，纯数据
3. 与 v3 schema 字段（tags / atomic_facts / hypothetical_questions / source_text）向后兼容
"""

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Schema 受控词表
# ---------------------------------------------------------------------------

PAGE_TYPES: list[str] = ["source", "entity", "topic", "event", "analysis"]
"""5 类页面类型。source 是 raw record 一对一映射；entity/topic/event 是综合页；
analysis 是检索时动态合成的跨页摘要（可缓存）。"""

RELATION_TYPES: list[str] = [
    "subject_of",       # source 页是某 entity 的提及（source -> entity）
    "topic_of",         # source 页归属于某 topic（source -> topic）
    "evidence_in",      # entity/topic/event 页引用某 source 作为证据
    "preceded_by",      # event/state 时序前驱
    "followed_by",      # event/state 时序后继
    "contradicts",      # 跨页矛盾关系（与 warn 复合机制联动）
    "derived_from",     # analysis 页 -> 综合所引用的 entity/topic 页
    "co_occurred_with", # 两个 entity 在同一 source 中共现
    "part_of",          # event 属于更大的 event；topic 属于父 topic
    "mentions",         # 兜底关系：表层提及，无更强语义
    "version_of",       # 旧版本页 -> 新版本页（多版本机制）
    "superseded_by",    # 等价于 deprecated_by，向后兼容 v3
    "corroborated_by",  # source 互相印证（同一事实多次表达）
]
"""typed wikilink 关系受控词表（v4 §5.3）。LLM 输出不在表中的关系自动回退到 `mentions`。"""


PAGE_TYPES_SET = set(PAGE_TYPES)
RELATION_TYPES_SET = set(RELATION_TYPES)


# ---------------------------------------------------------------------------
# 基础数据单元
# ---------------------------------------------------------------------------


@dataclass
class TypedWikilink:
    """
    带语义类型的 wikilink。

    Attributes:
        target: 目标 entry 的标识符。可以是 ``entry_id``，也可以是路径形式
            ``entities/caroline`` / ``sources/conv-26_session-3_turn-7``。
        relation: 关系类型，必须属于 :data:`RELATION_TYPES`，否则由
            :class:`WikilinkGraph` 在写入时回退为 ``"mentions"``。
    """

    target: str
    relation: str


@dataclass
class AtomicFact:
    """
    原子事实，(subject, predicate, object) 三元组 + 可选时间锚点。

    用于 v3 的 SPO 冲突检测（同 subject + 同 predicate + 不同 object => 状态变更）
    与 v4 的多版本机制联动。``confidence`` 暂未启用，预留给后续 LLM 自我评估。
    """

    subject: str
    predicate: str
    object: str
    time: str | None = None
    confidence: float = 1.0


@dataclass
class TimeAnchor:
    """归一化的时间锚点。三层解析的输出统一形式（见 :class:`TimeNormalizer`）。"""

    iso: str                  # "2023-06-27" / "2023-06" / "" 表示无 ISO 形式
    session: int | None = None
    certainty: str = "high"   # "high" | "medium" | "fuzzy"
    raw: str = ""             # 解析前的原始字符串，便于审计


@dataclass
class WikiVersion:
    """
    单个页面的一个版本快照。

    多版本机制核心：valid_from_session 与 valid_to_session 构成半开区间
    [valid_from, valid_to)。``valid_to_session is None`` 表示当前 latest。
    """

    version_id: str
    valid_from_session: int
    valid_to_session: int | None
    content: str
    atomic_facts: list[AtomicFact] = field(default_factory=list)
    source_record_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 核心页面容器
# ---------------------------------------------------------------------------


@dataclass
class WikiEntry:
    """
    MemWiki 单个页面的统一容器（5 类页面共用）。

    设计说明：
    - 字段尽量与 v3 schema 对齐（tags / atomic_facts / hypothetical_questions /
      source_text），方便 v3 的检索路径直接复用。
    - ``versions`` 保存多版本（v4 §四）。最新版本对应 versions[-1]。
    - ``wikilinks`` 保存出向 typed wikilinks，入向由 :class:`WikiIndex.backlinks`
      集中维护，避免双写不一致。
    - 工程旗标（``degraded`` / ``wikify_skipped`` / ``quality_low`` /
      ``is_segmented``）保留 v3 工程鲁棒性输出。
    """

    entry_id: str
    page_type: str                                          # PAGE_TYPES 之一
    title: str
    tags: dict[str, list] = field(default_factory=dict)     # {"entities": [...], "topics": [...]}
    time_anchors: list[TimeAnchor] = field(default_factory=list)
    atomic_facts: list[AtomicFact] = field(default_factory=list)
    hypothetical_questions: list[str] = field(default_factory=list)
    source_text: str = ""
    wikilinks: list[TypedWikilink] = field(default_factory=list)
    versions: list[WikiVersion] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    deprecated_by: str | None = None                        # v3 兼容字段
    is_segmented: bool = False
    degraded: bool = False
    wikify_skipped: bool = False
    quality_low: bool = False
    last_updated_session: int = 0

    # ------------------------------------------------------------------
    # 便利访问器
    # ------------------------------------------------------------------

    @property
    def current_version(self) -> WikiVersion | None:
        """返回最新版本（valid_to_session is None 的那一条）。"""
        for ver in reversed(self.versions):
            if ver.valid_to_session is None:
                return ver
        return self.versions[-1] if self.versions else None

    @property
    def current_content(self) -> str:
        """返回最新版本的 content，缺省回退到 source_text。"""
        cur = self.current_version
        return cur.content if cur else self.source_text


# ---------------------------------------------------------------------------
# 全局索引
# ---------------------------------------------------------------------------


@dataclass
class WikiIndex:
    """
    单 sample 的全局索引：所有 WikiEntry 的注册中心 + 图结构 + 元数据。

    通过 :class:`WikiIndexBuilder` 渲染为 ``wiki_index.md`` 用于中央导航
    （v4 §六）。``entries_by_type`` 与 ``backlinks`` 是冗余字段，应由 Builder
    在 upsert 时同步维护。
    """

    sample_id: str
    entries: dict[str, WikiEntry] = field(default_factory=dict)              # entry_id -> entry
    entries_by_type: dict[str, list[str]] = field(default_factory=dict)      # page_type -> entry_ids
    entity_aliases: dict[str, str] = field(default_factory=dict)             # alias(lower) -> canonical
    wikilink_graph: dict[str, list[TypedWikilink]] = field(default_factory=dict)  # entry_id -> outbound
    backlinks: dict[str, list[str]] = field(default_factory=dict)            # entry_id -> entry_ids that link to it
    total_records: int = 0
    last_updated_session: int = 0

    # ------------------------------------------------------------------
    # 便利访问器（避免到处写裸 dict 查询）
    # ------------------------------------------------------------------

    def get(self, entry_id: str) -> WikiEntry | None:
        return self.entries.get(entry_id)

    def list_by_type(self, page_type: str) -> list[WikiEntry]:
        ids = self.entries_by_type.get(page_type, [])
        return [self.entries[i] for i in ids if i in self.entries]

    def upsert(self, entry: WikiEntry) -> None:
        """
        注册或更新一个 entry，同步维护 entries_by_type / wikilink_graph。
        backlinks 由 :class:`WikilinkGraph` 在 add_wikilink 时单独维护。
        """
        self.entries[entry.entry_id] = entry
        bucket = self.entries_by_type.setdefault(entry.page_type, [])
        if entry.entry_id not in bucket:
            bucket.append(entry.entry_id)
        self.wikilink_graph[entry.entry_id] = list(entry.wikilinks)


__all__ = [
    "PAGE_TYPES",
    "RELATION_TYPES",
    "PAGE_TYPES_SET",
    "RELATION_TYPES_SET",
    "TypedWikilink",
    "AtomicFact",
    "TimeAnchor",
    "WikiVersion",
    "WikiEntry",
    "WikiIndex",
]
