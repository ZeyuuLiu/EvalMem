"""
AgenticRAGRetriever — v3 主入口。

实现 EvalMem `EncodingHighRecallRetriever` 协议：
    retrieve(req: HighRecallRequest) -> HighRecallResponse

8 节点流水线（设计文档 §四）：
    1 Plan-RAG 复杂度分析  | 2 多视图生成 | 3 粒度路由 | 4 并行检索
    5 CRAG 质量自评       | 6 RRF 融合  | 7 LLM 精排 | 8 覆盖度判定 + 缺口重写
"""

from __future__ import annotations

import time
from typing import Any

from memory_eval.eval_core.agentic_rag.config import AgenticRAGConfig
from memory_eval.eval_core.agentic_rag.coverage import (
    CoverageResult,
    check_coverage,
)
from memory_eval.eval_core.agentic_rag.granularity import (
    GranularityProfile,
    detect_granularity,
    route_top_k,
)
from memory_eval.eval_core.agentic_rag.quality import (
    QualityAction,
    QualityScore,
    build_cross_view_query,
    decide_action,
    evaluate_per_view,
    pick_donor_views,
)
from memory_eval.eval_core.agentic_rag.retrieval import (
    BM25Index,
    Candidate,
    EmbedFn,
    build_bm25_index,
    parallel_retrieve,
    rrf_fuse,
)
from memory_eval.eval_core.agentic_rag.reranker import rerank
from memory_eval.eval_core.agentic_rag.termination import (
    candidate_id_set,
    compare_rounds,
    merge_candidate_pool,
    pool_to_ranked_list,
)
from memory_eval.eval_core.agentic_rag.views import (
    QueryView,
    ViewPlan,
    analyze_complexity,
    generate_gap_query,
    generate_views,
)
from memory_eval.eval_core.high_recall import (
    EncodingHighRecallRetriever,
    HighRecallRequest,
    HighRecallResponse,
)


class AgenticRAGRetriever(EncodingHighRecallRetriever):
    """AgenticRAG v3 编码探针专用高召回检索器。

    通过 `adapter.set_external_high_recall_retriever(retriever)` 注入到
    EncodingAgent.collect_observations；输出 HighRecallResponse 后由
    EncodingAgent 合并到 native_candidate_view，进入下游 LLM 判定。
    """

    def __init__(
        self,
        config: AgenticRAGConfig | None = None,
        *,
        embed_fn: EmbedFn | None = None,
    ) -> None:
        self.config = config or AgenticRAGConfig()
        self.embed_fn = embed_fn

    def retrieve(self, request: HighRecallRequest) -> HighRecallResponse:
        """v3 八节点编排入口。返回 HighRecallResponse(candidates, diagnostics)。"""
        cfg = self.config
        t0 = time.time()
        diagnostics: dict[str, Any] = {"agentic_rag_version": "v4-encoding-memory-rag"}

        question = request.query
        f_key = list(request.f_key or [])
        memory_corpus = list(request.memory_corpus or [])

        # 节点 1-3：planning
        plan = analyze_complexity(question=question, f_key=f_key, cfg=cfg)
        views = generate_views(question=question, f_key=f_key, plan=plan, cfg=cfg)
        granularity = detect_granularity(memory_corpus=memory_corpus, cfg=cfg)
        top_k = route_top_k(granularity)
        diagnostics["plan"] = _plan_to_dict(plan)
        diagnostics["views_initial"] = [_view_to_dict(v) for v in views]
        diagnostics["granularity"] = _granularity_to_dict(granularity)

        # BM25 索引一次性构建
        bm25_index: BM25Index | None = None
        if cfg.enable_bm25 and any(v.view_type == "bm25" for v in views):
            bm25_index = build_bm25_index(memory_corpus)
            diagnostics["bm25_indexed"] = len(bm25_index.docs)

        # 节点 4-8：迭代主循环
        final, last_cov, iter_logs = self._run_iterations(
            question=question,
            f_key=f_key,
            memory_corpus=memory_corpus,
            initial_views=views,
            top_k=top_k,
            bm25_index=bm25_index,
        )

        elapsed = time.time() - t0
        if cfg.verbose_diagnostics:
            diagnostics["iterations"] = iter_logs
            diagnostics["final_coverage"] = _coverage_to_dict(last_cov) if last_cov else {}
            diagnostics["elapsed_sec"] = elapsed
            diagnostics["embed_fn_provided"] = self.embed_fn is not None

        return HighRecallResponse(
            candidates=[_candidate_to_record(c) for c in final],
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # Iteration loop (节点 4-8)
    # ------------------------------------------------------------------

    def _run_iterations(
        self,
        *,
        question: str,
        f_key: list[str],
        memory_corpus: list[dict[str, Any]],
        initial_views: list[QueryView],
        top_k: int,
        bm25_index: BM25Index | None,
    ) -> tuple[list[Candidate], CoverageResult | None, list[dict[str, Any]]]:
        cfg = self.config
        current_views = initial_views
        final: list[Candidate] = []
        last_cov: CoverageResult | None = None
        logs: list[dict[str, Any]] = []
        reconstruct_count = 0
        candidate_pool: dict[str, Candidate] = {}
        previous_round_ids: set[str] = set()
        stable_streak = 0

        for it in range(cfg.max_iter):
            round_top_k = min(cfg.max_pool_size, max(1, int(top_k or 1)) * (it + 1))
            log: dict[str, Any] = {"iter": it + 1, "round_top_k": round_top_k}

            # 4. 并行检索
            view_results = parallel_retrieve(
                views=current_views,
                memory_corpus=memory_corpus,
                top_k=round_top_k,
                embed_fn=self.embed_fn,
                bm25_index=bm25_index,
            )
            log["view_hits"] = {vt: len(cs) for vt, cs in view_results.items()}

            # 5. CRAG 质量自评 + 动作决策
            scores = evaluate_per_view(view_results, question, f_key, cfg)
            log["quality_scores"] = {vt: _score_to_dict(s) for vt, s in scores.items()}
            action = decide_action(scores, tau_quality=cfg.tau_quality)
            log["quality_action"] = action.value

            if action == QualityAction.CROSS_VIEW_FALLBACK:
                view_results = self._apply_cross_view_fallback(
                    view_results, scores, current_views, memory_corpus, top_k, bm25_index
                )
                log["after_fallback_hits"] = {vt: len(cs) for vt, cs in view_results.items()}
            elif action == QualityAction.VIEW_RECONSTRUCT and reconstruct_count < cfg.quality_reconstruct_max:
                reconstruct_count += 1
                plan = analyze_complexity(question=question, f_key=f_key, cfg=cfg)
                current_views = generate_views(question=question, f_key=f_key, plan=plan, cfg=cfg)
                log["reconstructed_views"] = [_view_to_dict(v) for v in current_views]
                logs.append(log)
                continue

            # 6-7. RRF + LLM 精排
            fused = rrf_fuse(
                view_results=view_results,
                k_param=cfg.rrf_k,
                view_weights={v.view_type: v.weight for v in current_views},
            )
            rerank_output_k = min(cfg.max_pool_size, max(cfg.rerank_output_k, round_top_k))
            reranked = rerank(candidates=fused, question=question, f_key=f_key, cfg=cfg, top_k=rerank_output_k)
            convergence = compare_rounds(
                reranked,
                previous_round_ids,
                jaccard_threshold=cfg.round_jaccard_threshold,
                epsilon_new_ratio=cfg.epsilon_new_ratio,
            )
            previous_round_ids = candidate_id_set(reranked)
            stable_streak = stable_streak + 1 if convergence.is_stable else 0
            candidate_pool = merge_candidate_pool(candidate_pool, reranked)
            final = pool_to_ranked_list(candidate_pool, max_size=cfg.max_pool_size)
            log["rrf_top5_ids"] = [c.id for c in fused[:5]]
            log["rerank_top5_ids"] = [c.id for c in reranked[:5]]
            log["pool_size"] = len(candidate_pool)
            log["pool_limit"] = cfg.max_pool_size
            log["convergence"] = {
                "stable": convergence.is_stable,
                "jaccard": convergence.jaccard,
                "new_ratio": convergence.new_ratio,
                "new_count": len(convergence.new_ids),
                "current_size": convergence.current_size,
                "previous_size": convergence.previous_size,
                "stable_streak": stable_streak,
                "required_stable_rounds": cfg.consecutive_stable_rounds,
                "reason": convergence.reason,
            }

            # 8. 覆盖度诊断。它可辅助 gap-query 生成，但不再控制终止。
            coverage: CoverageResult | None = None
            if cfg.enable_coverage_diagnostics:
                coverage = check_coverage(candidates=final, f_key=f_key, cfg=cfg)
                log["coverage"] = _coverage_to_dict(coverage)
                log["coverage_threshold_diagnostic"] = cfg.tau_cover
                last_cov = coverage
            else:
                log["coverage"] = {"enabled": False}

            if len(candidate_pool) >= cfg.max_pool_size:
                log["terminated"] = "candidate_pool_limit_reached"
                logs.append(log)
                break

            if stable_streak >= cfg.consecutive_stable_rounds:
                log["terminated"] = "retrieval_results_converged"
                logs.append(log)
                break

            # 缺口重写只负责扩大下一轮查询；即使 raw memory 缺失目标事实，max_iter 仍会兜底。
            if it < cfg.max_iter - 1 and coverage is not None and coverage.uncovered_units:
                gap_view = generate_gap_query(coverage.uncovered_units, cfg)
                if gap_view.query:
                    current_views = list(current_views) + [gap_view]
                    log["gap_query"] = _view_to_dict(gap_view)
            elif it >= cfg.max_iter - 1:
                log["terminated"] = "max_iter_reached"
            else:
                log["no_gap_query_reason"] = "coverage_diagnostics_disabled_or_no_uncovered_units"
            logs.append(log)

        return final, last_cov, logs

    # ------------------------------------------------------------------
    # CRAG cross-view fallback
    # ------------------------------------------------------------------

    def _apply_cross_view_fallback(
        self,
        view_results: dict[str, list[Candidate]],
        scores: dict[str, QualityScore],
        views: list[QueryView],
        memory_corpus: list[dict[str, Any]],
        top_k: int,
        bm25_index: BM25Index | None,
    ) -> dict[str, list[Candidate]]:
        """用高质视图候选反哺低质视图查询并重检。

        TODO: implement smarter donor selection (current: take all high-quality views).
        """
        high_types, low_types = pick_donor_views(scores, tau_quality=self.config.tau_quality)
        if not high_types or not low_types:
            return view_results

        donor: list[Candidate] = []
        for vt in high_types:
            donor.extend(view_results.get(vt, [])[:3])
        view_by_type = {v.view_type: v for v in views}

        rewritten: list[QueryView] = []
        for vt in low_types:
            base = view_by_type.get(vt)
            if base is None:
                continue
            rewritten.append(
                QueryView(
                    view_type=vt,
                    query=build_cross_view_query(vt, donor, base.query),
                    weight=base.weight,
                )
            )

        rebuilt = dict(view_results)
        if rewritten:
            rebuilt.update(
                parallel_retrieve(
                    views=rewritten,
                    memory_corpus=memory_corpus,
                    top_k=top_k,
                    embed_fn=self.embed_fn,
                    bm25_index=bm25_index,
                )
            )
        return rebuilt


# ---------------------------------------------------------------------------
# dict-isation helpers
# ---------------------------------------------------------------------------


def _plan_to_dict(plan: ViewPlan) -> dict[str, Any]:
    return {
        "view_count": plan.view_count,
        "view_types": list(plan.view_types),
        "confidence": plan.confidence,
        "reasoning": plan.reasoning,
        "anchors": dict(plan.anchors),
    }


def _view_to_dict(view: QueryView) -> dict[str, Any]:
    return {"view_type": view.view_type, "query": view.query, "weight": view.weight}


def _granularity_to_dict(g: GranularityProfile) -> dict[str, Any]:
    return {
        "avg_tokens": g.avg_tokens,
        "granularity_type": g.granularity_type,
        "recommended_k": g.recommended_k,
        "drill_down_enabled": g.drill_down_enabled,
        "sample_size": g.sample_size,
    }


def _score_to_dict(s: QualityScore) -> dict[str, Any]:
    return {"score": s.score, "reason": s.reason, "top_supporting_ids": list(s.top_supporting_ids)}


def _coverage_to_dict(c: CoverageResult) -> dict[str, Any]:
    return {
        "coverage_rate": c.coverage_rate,
        "covered_units": list(c.covered_units),
        "uncovered_units": list(c.uncovered_units),
        "evidence_by_unit": dict(c.evidence_by_unit),
        "reasoning": c.reasoning,
    }


def _candidate_to_record(c: Candidate) -> dict[str, Any]:
    """转换为 EvalMem encoding_agent._normalize_records 期望的 dict 形式。"""
    return {
        "id": c.id,
        "text": c.text,
        "score": float(c.score),
        "meta": {**dict(c.meta), "source_view": c.source_view, "agentic_rag_rank": c.rank_in_view},
    }
