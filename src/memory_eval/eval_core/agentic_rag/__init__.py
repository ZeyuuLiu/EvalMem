"""
AgenticRAG v3 — 编码探针专用高召回检索器。

借鉴 2024-2026 SOTA Agentic RAG 工作（RankRAG / CRAG / Plan-RAG / SPLADE）
并叠加 EvalMem 评测场景独有的两个先验（F_key 可见 + 系统集合粒度已知）。

哲学锚：**宁滥勿缺**（high-recall maximization）。

主要导出:
    AgenticRAGRetriever   - 实现 EncodingHighRecallRetriever 协议的主类
    AgenticRAGConfig      - 全部超参数（与 EvaluatorConfig 解耦）
    QualityAction         - CRAG 决策枚举（PROCEED / CROSS_VIEW_FALLBACK / VIEW_RECONSTRUCT）

典型用法:
    >>> from memory_eval.eval_core.agentic_rag import AgenticRAGRetriever, AgenticRAGConfig
    >>> retriever = AgenticRAGRetriever(config=AgenticRAGConfig(llm_api_key="..."))
    >>> adapter.set_external_high_recall_retriever(retriever)
"""

from __future__ import annotations

from memory_eval.eval_core.agentic_rag.config import AgenticRAGConfig
from memory_eval.eval_core.agentic_rag.coverage import CoverageResult, check_coverage
from memory_eval.eval_core.agentic_rag.granularity import GranularityProfile, detect_granularity, route_top_k
from memory_eval.eval_core.agentic_rag.quality import (
    QualityAction,
    QualityScore,
    decide_action,
    evaluate_per_view,
)
from memory_eval.eval_core.agentic_rag.retrieval import BM25Index, Candidate, build_bm25_index, parallel_retrieve, rrf_fuse
from memory_eval.eval_core.agentic_rag.reranker import rerank
from memory_eval.eval_core.agentic_rag.retriever import AgenticRAGRetriever
from memory_eval.eval_core.agentic_rag.views import QueryView, ViewPlan, analyze_complexity, generate_gap_query, generate_views

__all__ = [
    "AgenticRAGRetriever",
    "AgenticRAGConfig",
    "QueryView",
    "ViewPlan",
    "GranularityProfile",
    "Candidate",
    "BM25Index",
    "QualityAction",
    "QualityScore",
    "CoverageResult",
    "analyze_complexity",
    "generate_views",
    "generate_gap_query",
    "detect_granularity",
    "route_top_k",
    "parallel_retrieve",
    "build_bm25_index",
    "rrf_fuse",
    "evaluate_per_view",
    "decide_action",
    "rerank",
    "check_coverage",
]

__version__ = "0.3.0"
