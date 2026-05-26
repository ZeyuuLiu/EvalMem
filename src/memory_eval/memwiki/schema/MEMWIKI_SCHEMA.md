# MEMWIKI_SCHEMA — 冻结版本规约（v4.0）

> 本文是 MemWiki 的 **AGENTS.md 等价物**。所有 LLM 提示词、决策规则、受控词表都
> 集中在此处定义。论文实验阶段冻结，附录公开 SHA-256 hash 以保证跨 8 个被测系统
> 的 wiki 构建严格一致。

---

## 1. 5 类页面

| 类型 | 触发条件 | 内容职责 | 路径示例 |
| --- | --- | --- | --- |
| **source** | 每条 raw record 一对一生成（除 skip / segment 兜底外） | 保留原文 + 浅层结构化 | `sources/conv-26_session-3_turn-7` |
| **entity** | 同 canonical 实体出现 ≥ `entity_min_occurrences` (默认 2) | 跨多 record 综合该实体的所有信息 | `entities/caroline` |
| **topic** | 同受控主题词出现 ≥ `topic_min_occurrences` (默认 3) | 跨多 record 围绕该主题的演化轨迹 | `topics/relationship_status` |
| **event** | 单一事件被 ≥ `event_min_evidence` (默认 2) 条 source 提及 | 该事件的发生时间、参与者、影响 | `events/caroline_started_dating_D` |
| **analysis** | 检索时跨页综合（可缓存） | LLM 自动生成的对比 / 综合 | `analyses/caroline_dating_pattern` |

页面统一容器是 :class:`memory_eval.memwiki.schema.WikiEntry`，共享 9 个核心字段
（`title / tags / time_anchors / atomic_facts / hypothetical_questions /
source_text / wikilinks / versions / warnings`），page_type 区分类型语义。

---

## 2. Typed Wikilink 关系（13 种）

| relation | 语义 | 典型方向 |
| --- | --- | --- |
| `subject_of` | source/event 页是某 entity 的提及 | source → entity |
| `topic_of` | source/event 页归属于某 topic | source → topic |
| `evidence_in` | 综合页引用某 source 作为证据 | entity/topic/event → source |
| `preceded_by` | event/state 的时序前驱 | event → event |
| `followed_by` | event/state 的时序后继 | event → event |
| `contradicts` | 跨页矛盾（与 warn 复合机制联动） | entry → entry |
| `derived_from` | analysis 页综合所引用的 entity/topic 页 | analysis → entity/topic |
| `co_occurred_with` | 两个 entity 在同一 source 共现 | entity → entity |
| `part_of` | event 属于更大 event；topic 属于父 topic | child → parent |
| `mentions` | 兜底：表层提及，无更强语义 | any → any |
| `version_of` | 旧版本 → 新版本 | old → new |
| `superseded_by` | v3 兼容字段，等价于 deprecated_by | old → new |
| `corroborated_by` | source 互相印证（同一事实多次表达） | source → source |

**强证据型关系**（参与检索关系拓展，见 `MemWikiConfig.expand_relations`）：
`evidence_in / subject_of / topic_of / preceded_by / followed_by`。其余关系仅用
于导航与 lint，不召回扩张。

---

## 3. Wikify 规则（v3 §一 + §二）

### 3.1 鲁棒兜底状态机

```
raw record
  │
  ├─ len < record_skip_min_tokens (20) 且无 NER → skipped (only source_text)
  ├─ len > record_segment_max_tokens (4000) → segment-then-merge, is_segmented=True
  └─ 否则进入 wikify
      ├─ L1 (temp=0.0) → JSON 合法?
      │     ├─ yes → upsert
      │     └─ no  → L2
      ├─ L2 (temp=0.3 + negative_hint) → JSON 合法?
      │     ├─ yes → upsert
      │     └─ no  → L3
      └─ L3 → degraded entry (only source_text), degraded=True
```

监控指标阈值：

| 指标 | 阈值 | 说明 |
| --- | --- | --- |
| `wikify_success_rate` | ≥ 92% | L1 一次成功 |
| `degraded_entry_ratio` | ≤ 5% | L1+L2 双败 |
| `segmented_entry_ratio` | 视系统 | MemBox ~10%，O-Mem ~1% |
| `skipped_record_ratio` | ≤ 8% | 短且无 NER |

### 3.2 假设问题三轴控制

- **Coverage**：每个 atomic_fact 至少 1 问；
- **Granularity**：1 broad + N-2 specific + 1 yes/no；
- **Surface variation**：4 问不重复同一关键名词；
- **Answerability**：仅靠本 record 即可回答，拒绝泛问。

数量自适应（基于 atomic_facts 数）：

| `len(atomic_facts)` | 生成数 |
| --- | --- |
| ≥ 5 | `hq_count_rich` (6) |
| 2-4 | `hq_count_default` (4) |
| ≤ 1 | `hq_count_simple` (2) |

后处理：两两 cos > `hq_similarity_threshold` (0.95) → 重生成；两次仍不达标 →
保留并标记 `quality_low=True`，检索分数 × `hq_quality_score_discount` (0.7)。

---

## 4. Composer 决策规则（v4 §三）

| 输入条件 | action |
| --- | --- |
| new record 带来新维度 + 不矛盾 | `rewrite` |
| new record 仅是新证据 + 无新维度 | `append` |
| new record 与 P 已有信息矛盾 | `warn`（触发多版本 + warning 块） |
| new record 无增量价值 | `no_change` |

四选一中失败兜底：LLM 调用 / JSON 解析失败 → 默认 `no_change`，避免错误重写。

---

## 5. 多版本机制（v4 §四）

- 每个 entry 维护 versions list；
- ``[valid_from_session, valid_to_session)`` 半开区间，``valid_to_session is
  None`` 表示当前 latest；
- 状态变更检测：同 subject + 同 predicate + 不同 object（``state_change_llm_confirm``
  默认 true，走 LLM 二次确认避免误判"并列偏好"）；
- v3 兼容：旧版本同时打 ``deprecated_by = new_version_id``。

时间锚点查询：

| query_type | 行为 |
| --- | --- |
| `latest` | 返回 valid_to is None 的 entry |
| `explicit` (target_session=T) | 返回 valid_from ≤ T < valid_to 的版本 |
| `fuzzy` | 返回 latest（由 answer LLM 在多版本中选择） |

---

## 6. MemWiki Lint 6 类检查

| 检查 | 说明 |
| --- | --- |
| `orphan_pages` | 无入链页面 |
| `broken_wikilinks` | target 不存在 |
| `contradictions` | 同 SPO 不同 O 且双方 warnings 未标记 |
| `unregistered` | 存在于 entries 但未在 entries_by_type 注册 |
| `outdated_versions` | 版本链不连续 / 多 current / 逆序 |
| `vocabulary_violations` | TODO：relation/topic 越出受控词表（暂回退到 `mentions`/`other`） |

Lint 输出作为 ``wiki_index.md`` 的 ``## Lint Report`` 章节追加。

---

## 7. 评测合规性（LeakAuditor）

MemWiki 构建**严禁**接触：

- `sample.f_key`
- `sample.answer_gold`
- `sample.evidence_ids` / `sample.evidence_texts` / `sample.evidence_with_time`
- `sample.oracle_context`

:class:`memory_eval.memwiki.leak_auditor.LeakAuditor` 通过 random substitution +
结构化 diff 验证 MemWiki 输出与上述敏感字段无关。CI 应对每个 sample（或抽样
10%）运行 audit，差异为空才放行。

---

## 8. Schema 演化策略

- 论文实验阶段：**冻结**，附录公布 hash；
- 论文发表后的开源 release：允许 LLM-Wiki 风格的"人 + LLM 共同演化"；
- 任何 schema 改动必须 git 版本控制 + Lint 强制校验。
