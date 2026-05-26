from __future__ import annotations

"""
MemWiki v4 配置项。

集中维护所有可调参数：v3 工程鲁棒性阈值、v4 页面阈值、RRF / 检索参数、
受控词表路径、LLM / embedding 配置。

设计原则：单 dataclass，全部带默认值；运行期允许 dataclasses.replace 覆盖。
"""

from dataclasses import dataclass, field


@dataclass
class MemWikiConfig:
    """MemWiki v4 全局配置。"""

    # ------------------------------------------------------------------
    # v3 工程鲁棒性（优化 1）
    # ------------------------------------------------------------------
    wikify_max_retries: int = 2                  # LLM 改写最大重试次数
    wikify_temp_initial: float = 0.0
    wikify_temp_retry: float = 0.3
    wikify_max_tokens_initial: int = 1024
    wikify_max_tokens_retry: int = 2048
    record_skip_min_tokens: int = 20             # 短记录跳过阈值
    record_segment_max_tokens: int = 4000        # 超长记录分段阈值
    segment_window: int = 800
    segment_overlap: int = 120

    # ------------------------------------------------------------------
    # v4 页面阈值（§2.2 触发条件）
    # ------------------------------------------------------------------
    entity_min_occurrences: int = 2              # entity 页生成最小提及次数
    topic_min_occurrences: int = 3               # topic 页生成最小提及次数
    event_min_evidence: int = 2                  # event 页最少需要的证据 source 数

    # ------------------------------------------------------------------
    # 假设问题质量（v3 优化 2）
    # ------------------------------------------------------------------
    hq_count_default: int = 4
    hq_count_rich: int = 6                       # ≥ 5 atomic_facts 时
    hq_count_simple: int = 2                     # ≤ 1 atomic_fact 时
    hq_similarity_threshold: float = 0.95        # 余弦相似度去重阈值
    hq_quality_score_discount: float = 0.7       # quality_low entry 检索分数折扣

    # ------------------------------------------------------------------
    # 4 路 RRF 检索（v3 + v4）
    # ------------------------------------------------------------------
    rrf_k: int = 60
    retrieval_top_k_per_view: int = 20
    retrieval_top_k_final: int = 10
    expand_hops: int = 2                         # wikilink 关系拓展跳数
    expand_relations: list[str] = field(
        default_factory=lambda: [
            "evidence_in",
            "subject_of",
            "topic_of",
            "preceded_by",
            "followed_by",
        ]
    )
    """允许参与拓展的强证据型关系（v4 §5.4）。其余关系（mentions / co_occurred_with）
    不参与召回扩张，避免噪声。"""

    # ------------------------------------------------------------------
    # Schema 受控词表路径（相对包根目录）
    # ------------------------------------------------------------------
    relations_vocabulary_path: str = "schema/relations.yaml"
    topics_vocabulary_path: str = "schema/topics_vocabulary.yaml"
    schema_doc_path: str = "schema/MEMWIKI_SCHEMA.md"

    # ------------------------------------------------------------------
    # 多版本机制
    # ------------------------------------------------------------------
    state_change_llm_confirm: bool = True        # SPO 冲突时是否走 LLM 二次确认
    keep_deprecated_for_explain: bool = True     # ?explain=true 是否返回旧版本

    # ------------------------------------------------------------------
    # Lint
    # ------------------------------------------------------------------
    lint_on_build: bool = True
    lint_on_incremental_update: bool = False     # 增量更新时默认不触发 full lint

    # ------------------------------------------------------------------
    # LLM / Embedding
    # ------------------------------------------------------------------
    llm_model: str = "gpt-4o-mini"               # wikify / composer / lint 共用
    llm_temperature: float = 0.0
    llm_wikify_record_limit: int | None = None   # None = no budget cap when credentials are available
    llm_synthesis_page_limit: int | None = None  # None = no budget cap when credentials are available
    embedding_model: str = "Qwen3-Embedding-0.6B"
    embedding_dim: int = 1024
    embedding_batch_size: int = 64

    # ------------------------------------------------------------------
    # 数据落盘
    # ------------------------------------------------------------------
    output_root: str = "outputs/memwiki"
    write_wiki_index_md: bool = True
    write_per_entry_md: bool = False             # 是否单独导出每个 entry 的 markdown


__all__ = ["MemWikiConfig"]
