# AgenticRAG v3 — 编码探针专用高召回检索器

> EvalMem 编码探针的 `EncodingHighRecallRetriever` 实现。
> 哲学锚：**宁滥勿缺**（high-recall maximization）。
> 设计文档：`新增实验/方案设计/方案1_v3_AgenticRAG市场对标优化.md`

---

## 1. 架构总览（8 节点流水线）

```
HighRecallRequest (query + f_key + memory_corpus)
        │
        ▼
[1] Plan-RAG 复杂度分析               (增量 3, views.analyze_complexity)
        │  → ViewPlan(view_count ∈ [2,7], view_types)
        ▼
[2] 多视图查询生成                    (创新 1, views.generate_views)
        ├─ original  原问题 Q
        ├─ time      时间锚点
        ├─ entity    实体锚点
        ├─ topic     主题锚点
        └─ bm25      稀疏精确匹配     (增量 4)
        ▼
[3] 粒度感知 K 路由                   (创新 3, granularity.detect_granularity)
        │  K = 20 / 12 / 6 / 3  (fragment / turn / summary / document)
        ▼
[4] 并行检索  N 路 dense + 1 路 sparse (retrieval.parallel_retrieve)
        ▼
[5] CRAG 质量自评                     (增量 2, quality.evaluate_per_view)
        ├─ PROCEED              → 进入 RRF
        ├─ CROSS_VIEW_FALLBACK  → 高质视图反哺低质 query → 重检
        └─ VIEW_RECONSTRUCT     → LLM 重解析 F_key（重建上限 1 次）
        ▼
[6] RRF 融合                          (retrieval.rrf_fuse, k=60)
        ▼
[7] LLM-as-Reranker                   (增量 1, reranker.rerank)
        │  time / speaker / content 三轴 0-10 评分
        ▼
[8] F_key 覆盖度判定                  (创新 2, coverage.check_coverage)
        ├─ coverage_rate >= τ_cover (=0.85)  → RETURN
        ├─ < τ_cover && iter < max_iter      → generate_gap_query → 回到 [2]
        └─ iter == max_iter                  → 兜底返回当前 top-K
        ▼
HighRecallResponse(candidates, diagnostics)
```

## 2. 文件清单

| 文件 | 行数（约） | 职责 |
|---|---|---|
| `__init__.py` | 50 | 包导出与 `__all__` 表 |
| `config.py` | 120 | `AgenticRAGConfig` dataclass，集中超参数 |
| `prompts.py` | 200 | 6 个 LLM prompt 模板（严格 JSON 输出） |
| `views.py` | 250 | 创新点 1 + 增量 3：复杂度分析、视图生成、缺口查询 |
| `granularity.py` | 100 | 创新点 3：粒度统计与 K 路由 |
| `retrieval.py` | 280 | 并行 dense + BM25 + RRF；含 BM25 倒排实现 |
| `quality.py` | 130 | 增量 2：CRAG 质量自评 + 跨视图回退决策 |
| `reranker.py` | 120 | 增量 1：LLM-as-Reranker Listwise 精排 |
| `coverage.py` | 130 | 创新点 2：F_key 覆盖度 LLM 判定 |
| `retriever.py` | 270 | 8 节点编排主入口 `AgenticRAGRetriever` |
| `README.md` | 本文件 | 架构、用法、TODO |

## 3. 注入到 EvalMem 评测管线

`EncodingAgent.collect_observations` 会在编码探针执行前调用
`adapter.get_external_high_recall_retriever()`。把本检索器注入到 adapter 即可：

```python
from memory_eval.eval_core.agentic_rag import AgenticRAGRetriever, AgenticRAGConfig

# 1. 准备配置（沿用 EvaluatorConfig 中的 LLM 鉴权）
ar_config = AgenticRAGConfig(
    llm_api_key=os.environ["OPENAI_API_KEY"],
    llm_base_url="https://api.openai.com/v1",
    llm_model="gpt-4o-mini",
    enable_bm25=True,
    enable_memo_rag_clue=False,   # 增量 5 默认关闭，做消融时再打开
)

# 2. 准备 embedding function（可选；如不传，dense 视图退化为字符串兜底）
def my_embed_fn(texts: list[str]) -> list[list[float]]:
    # 用 Qwen3-Embedding-0.6B 或任意兼容服务
    ...

# 3. 实例化并注入
retriever = AgenticRAGRetriever(config=ar_config, embed_fn=my_embed_fn)

# 4. 把它挂到具体 adapter（每个被测系统的 adapter 都要实现 set_external_high_recall_retriever）
adapter.set_external_high_recall_retriever(retriever)

# 5. 正常运行 EvalMem
from memory_eval.eval_core import EncodingAgent
result = EncodingAgent().evaluate_with_adapter(sample, adapter, run_ctx, cfg=evaluator_config)
```

`retriever.retrieve(req)` 返回 `HighRecallResponse(candidates, diagnostics)`；
EncodingAgent 把 candidates 合并进 `native_candidate_view` 后继续走原有
LLM 判定 / 规则兜底流程。**与现有评测无任何调用关系污染**。

## 4. 创新点 / 增量映射

### v2 三创新点（基础设施，保留）

| # | 创新点 | 实现位置 |
|---|---|---|
| 1 | F_key 结构化多视图 | `views.generate_views` + `prompts.build_view_generation_prompt` |
| 2 | F_key 覆盖度终止 | `coverage.check_coverage` + `retriever._retrieve` 循环条件 |
| 3 | 粒度感知 K 路由 | `granularity.detect_granularity` + `route_top_k` |

### v3 四市场对标增量

| # | 增量 | 实现位置 |
|---|---|---|
| 1 | LLM-as-Reranker (RankRAG) | `reranker.rerank` + `prompts.build_rerank_prompt` |
| 2 | CRAG 质量自评 + 跨视图回退 | `quality.evaluate_per_view` / `decide_action` / `build_cross_view_query` |
| 3 | Plan-RAG 自适应视图数 | `views.analyze_complexity` + `prompts.build_complexity_analysis_prompt` |
| 4 | BM25 稀疏第 5 视图 | `retrieval.build_bm25_index` + `_bm25_retrieve` |
| 5 | MemoRAG 全局 clue 扫描（可选，默认关闭） | `AgenticRAGConfig.enable_memo_rag_clue`（TODO） |

## 5. 已实现 vs TODO

### 已实现（skeleton 可直接 import + 调用通路自测）
- 全部 dataclass / Enum 类型与方法签名
- 8 节点编排主循环（含覆盖度未达时的缺口重写）
- BM25 倒排索引（轻量实现，无外部依赖）
- RRF 融合（含 view weights）
- 全部 6 个 LLM prompt 模板（严格 JSON 风格，与 EvalMem 现有 prompts.py 对齐）
- LLM 失败的规则兜底（字符串匹配 / RRF 排序）
- `HighRecallResponse.diagnostics` 含每阶段详细日志（受 `verbose_diagnostics` 控制）

### TODO（投稿前要补全）
- `views.py`：复杂度分析 LLM 调用的 majority voting（3 次取中位）
- `granularity.py`：用与编码 LLM 一致的 tokenizer（当前是简单正则）
- `retrieval.py`：BM25 中英混合 tokenizer（jieba + regex）；如允许引入 `rank_bm25` 可换为 BM25Okapi
- `retrieval.py`：注入真实 embed_fn（当前 dense 在 embed_fn=None 时退化为字符串兜底）
- `quality.py`：`build_cross_view_query` 当前是字符串拼接，可升级为 LLM 重写
- `reranker.py`：`cfg.enable_majority_voting` 的 3 次精排取中位实现
- `retriever.py`：增量 5（MemoRAG 全局 clue）的可选启用分支
- `prompts.py`：上下文超长（GAM 15K tok）时的截断策略

### 与外部模块的依赖
- `memory_eval.eval_core.high_recall.HighRecallRequest/Response` —— 不可改
- `memory_eval.eval_core.llm_assist._chat_json` 思路 —— views.py 内置等价实现，避免循环依赖
- `EvaluatorConfig.llm_api_key / llm_base_url` —— 需上层透传到 `AgenticRAGConfig`

## 6. 关键调试入口

每次 `retrieve()` 后，`HighRecallResponse.diagnostics` 含：

```json
{
  "agentic_rag_version": "v3",
  "plan": {"view_count": 5, "view_types": ["original","time","entity","topic","bm25"], "confidence": 0.92},
  "views_initial": [...],
  "granularity": {"avg_tokens": 220.0, "granularity_type": "summary", "recommended_k": 6},
  "bm25_indexed": 1500,
  "iterations": [
    {
      "iter": 1,
      "view_hits": {"original": 6, "time": 6, "entity": 6, "topic": 6, "bm25": 6},
      "quality_scores": {...},
      "quality_action": "cross_view_fallback",
      "rrf_top5_ids": [...],
      "rerank_top5_ids": [...],
      "coverage": {"coverage_rate": 1.0, "covered_units": [...], "uncovered_units": []},
      "terminated": "coverage_satisfied"
    }
  ],
  "elapsed_sec": 12.3
}
```

## 7. 引用

- RankRAG: Yu et al. 2024
- CRAG: Yan et al. 2024
- Plan-RAG / DSPy: Khattab et al. 2024
- SPLADE: Formal et al. 2021
- BGE-M3: Chen et al. 2024
- MemoRAG: Qian et al. 2024（可选启用）
