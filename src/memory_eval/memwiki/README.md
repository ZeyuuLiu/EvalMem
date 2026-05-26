# MemWiki v4

> 自演化、知识复合的私有记忆维基索引层（v4）。本目录是
> `src/memory_eval/memwiki/` 的实现，配套设计与算法见论文 §4 (MemWiki 节)
> 与附录。

## 架构总览

```
                              MemWiki v4
+--------------------------------------------------------------+
|  Schema 层（冻结的 MEMWIKI_SCHEMA.md + relations.yaml +      |
|  topics_vocabulary.yaml；LLM 行为契约的中央规约）            |
+----------------------------+---------------------------------+
                             |
                             v
+--------------------------------------------------------------+
|  Wiki 层（5 类页面）                                         |
|     +-----------+  +---------+  +--------+  +--------+  +----+|
|     |  source   |  | entity  |  | topic  |  | event  |  | analysis ||
|     +-----------+  +---------+  +--------+  +--------+  +----+|
|        ^                                                     |
|        | typed wikilinks (subject_of / evidence_in / ...)    |
|        v                                                     |
|  WikilinkGraph + Backlinks + Multi-Version (valid_from/to)   |
+----------------------------+---------------------------------+
                             |
                             v
+--------------------------------------------------------------+
|  Raw 层（adapter.export_full_memory(run_ctx)）—— 不可变兜底 |
+--------------------------------------------------------------+

检索路径：
    parse_query
        → 4 路并行（tag / HQ / fact / source_text）
        → RRF 融合
        → wikilink_graph.expand(top-5, hops=2, strong relations)
        → time_aware_rerank
        → top_k
```

## Quickstart

```python
from memory_eval.memwiki import (
    MemWikiBuilder,
    MemWikiRetriever,
    MemWikiConfig,
    WikiComposer,
    LeakAuditor,
)
from memory_eval.eval_core.llm_assist import LLMAssistConfig

config = MemWikiConfig()
llm_cfg = LLMAssistConfig(api_key="...", base_url="...")

# 1) 离线构建（每个 sample 一次）
builder = MemWikiBuilder(config=config, llm_cfg=llm_cfg)
wiki_index = builder.build(adapter, run_ctx, sample_id="conv-26")

# 2) 评测期检索
retriever = MemWikiRetriever(wiki_index, config, llm_cfg=llm_cfg)
retriever.prepare_indices()
hits = retriever.search("What is Caroline's relationship status?", top_k=10)

# 3) 增量整合（动态集每个 session 结束后调用）
composer = WikiComposer(config=config, llm_cfg=llm_cfg)
result = composer.integrate_new_record(new_record, wiki_index, session=13)

# 4) 评测合规性审计
auditor = LeakAuditor()
audit = auditor.audit(builder, adapter, run_ctx, sample)
assert audit.passed, audit.summary
```

## 文件一览

| 文件 | 职责 |
| --- | --- |
| `schema.py` | 全部 dataclass：WikiEntry / WikiVersion / TypedWikilink / AtomicFact / TimeAnchor / WikiIndex；以及 `PAGE_TYPES` / `RELATION_TYPES` 常量 |
| `config.py` | `MemWikiConfig` —— 所有可调参数集中处 |
| `prompts.py` | 9 个 LLM prompt 构建函数（wikify × 4 + composer + state_change + query_parse + hq + contradiction） |
| `normalizer.py` | EntityNormalizer / TimeNormalizer / TopicNormalizer（v3 优化 3） |
| `wikilink_graph.py` | typed wikilink 出向 + 入向 + 1-2 跳关系拓展 |
| `versioning.py` | VersionManager + TimeQuery（v4 §四） |
| `builder.py` | 离线构建主流程：5 类页面生成 + 鲁棒兜底（v3 §一） |
| `composer.py` | 知识复合：rewrite/append/warn/no_change（v4 §三 核心） |
| `retriever.py` | 4 路 RRF + 关系拓展 + 时间感知重排 |
| `index_builder.py` | `wiki_index.md` 自动渲染（v4 §六） |
| `lint.py` | MemWiki Lint 5+ 类健康检查（v4 §七） |
| `leak_auditor.py` | 评测合规性审计（不接触 sample.f_key / answer_gold / evidence_*） |
| `schema/MEMWIKI_SCHEMA.md` | LLM 行为冻结规约（等价 LLM-Wiki AGENTS.md） |
| `schema/relations.yaml` | 13 类 typed wikilink 受控词表 |
| `schema/topics_vocabulary.yaml` | 40 项主题受控词表 |

## 关键 TODO（skeleton 占位，需实现）

1. **LLM 调用集成**：`builder._call_llm_wikify` / `composer.llm_compose_decision` /
   `retriever.parse_query` 当前返回占位值。需对接
   `memory_eval.eval_core.llm_assist._chat_json`，并扩展支持 `max_tokens` /
   negative example 二试。
2. **向量索引**：`retriever._hq_index` / `_fact_index` / `_text_index` 当前为
   `None`。建议接 Qdrant，每路一个 collection，embedding 用
   `Qwen3-Embedding-0.6B`（与 EvalMem 主表一致）。
3. **NER 与 entity 聚类**：`builder._should_skip` / `entity_norm.build_alias_map`
   需要 spaCy `en_core_web_sm`，离线一次性预加载。
4. **超长记录分段**：`builder.segment_long_record` 当前仅返回空 sub_entries。
   需实现 sliding_window + 子片段 wikify + 合并逻辑（v3 §1.2）。
5. **Event 检测**：`builder.build_event_pages` 缺事件候选生成（建议先 LLM
   一次扫描所有 source 抽出 event_candidates，再走综合页路径）。
6. **TopicNormalizer 回退**：`_nearest_neighbor` 需实现 Levenshtein ≤ 2 + 向量
   最近邻 cos > 0.85 的双重匹配。
7. **TimeNormalizer 相对时间**：`_anchor_relative` 需结合 `dateutil.parser` +
   session_datetime 偏移计算。
8. **State change LLM 二次确认**：`versioning.detect_state_change` 占位的 LLM
   调用需对接 `build_state_change_detection_prompt`。

## 集成步骤（接入 EvalMem 评测主流程）

1. 在 `memory_eval.adapters.<system>.adapter` 增加 `export_full_memory_since` 方法
   （动态集增量更新需要）；
2. 实现上面 8 项 TODO；
3. 跑 `LeakAuditor` 对每个 sample 的构建做合规性审计；
4. 把 `MemWikiRetriever.search` 接入 `RetrievalAdapterProtocol.retrieve_original`，
   作为新的 retrieval mode（标志位 `--retrieval=memwiki_v4`）；
5. 论文实验 5 主表 / EvalMem-Dyn / 内部消融按 v4 §九 跑全套对比。

## 设计原则速查

- **Skeleton-first**：类型与接口 > 完整实现；TODO 标记每一处需要后填的细节；
- **Schema 冻结**：LLM 行为通过 `schema/MEMWIKI_SCHEMA.md` 与受控词表约束，
  不散落在代码注释；
- **审稿透明**：`LeakAuditor` + Lint Report 可作为论文附录，证明 wiki 构建
  独立于评测目标信息；
- **v3 兼容**：所有 v3 字段（`tags / atomic_facts / hypothetical_questions /
  source_text / cross_refs`）保留，仅新增；4 路 RRF 也保留为 v4 检索的子步骤。
