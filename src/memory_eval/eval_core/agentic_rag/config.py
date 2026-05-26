"""
AgenticRAG v3 配置数据类。

把 v3 的 4 个市场对标增量与 v2 的 3 个评测专用创新点的所有超参数集中管理。
默认值取自 `方案1_v3_AgenticRAG市场对标优化.md` 主表配置（C4）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _default_granularity_thresholds() -> list[int]:
    """fragment / turn / summary / document 四档切分阈值（单位：token）。"""
    return [80, 200, 600]


def _default_granularity_k_map() -> dict[str, int]:
    """粒度 -> Top-K 映射。来自 v2 创新点 3。"""
    return {"fragment": 20, "turn": 12, "summary": 6, "document": 3}


@dataclass
class AgenticRAGConfig:
    """
    AgenticRAG v3 全部超参数。

    与 EvalMem `EvaluatorConfig` 解耦：本配置只描述 retriever 行为，
    不重复 LLM 鉴权字段（rule fallback / require_llm_judgement 等留给上层）。
    """

    # ---------- 编码层闭环终止：有界扩大 K + 轮间收敛 ----------
    tau_cover: float = 0.85
    """fact-unit 覆盖率诊断阈值；仅用于日志，不再作为循环终止条件。"""

    max_iter: int = 4
    """缺口重写循环上限（含初轮）。"""

    max_pool_size: int = 50
    """累计候选池上限，防止高召回探针无限扩大上下文。"""

    round_jaccard_threshold: float = 0.85
    """当前轮候选集合与上一轮候选集合的 Jaccard 相似度达到该值时视为稳定。"""

    epsilon_new_ratio: float = 0.10
    """当前轮新增候选比例不超过该值时视为稳定。"""

    consecutive_stable_rounds: int = 2
    """连续稳定轮数达到该值后终止。"""

    enable_coverage_diagnostics: bool = True
    """是否保留 F_key 覆盖率诊断与 gap-query 生成；不参与停止判定。"""

    # ---------- v3 增量 2：CRAG 质量自评 ----------
    tau_quality: float = 0.7
    """单视图质量分阈值；< tau_quality 触发跨视图回退或视图重构。"""

    quality_reconstruct_max: int = 1
    """视图重构最大触发次数；CRAG 全低质场景的兜底，避免无限重建。"""

    # ---------- v2 创新点 3：粒度路由 ----------
    granularity_thresholds: list[int] = field(default_factory=_default_granularity_thresholds)
    """avg_tokens 切分阈值，长度必须为 3：(<thr0) -> fragment, [thr0,thr1) -> turn, [thr1,thr2) -> summary, >=thr2 -> document。"""

    granularity_k_map: dict[str, int] = field(default_factory=_default_granularity_k_map)
    """每档对应的 Top-K。"""

    drill_down_enabled: bool = True
    """summary / document 档启用粒度下钻（K 翻倍取细粒度证据）。"""

    # ---------- v3 增量 1：LLM 精排 ----------
    rerank_top_n: int = 30
    """RRF 后送入 LLM 精排的候选数，O-Mem 7K 上限/GAM 15K 上限可降至 20。"""

    rerank_output_k: int = 10
    """LLM 精排输出 Top-K，配合下游覆盖度判定。"""

    # ---------- 检索融合 ----------
    rrf_k: int = 60
    """RRF 平滑参数 k。"""

    enable_bm25: bool = True
    """v3 增量 4：是否启用 BM25 稀疏第 5 视图。"""

    # ---------- v3 增量 3：Plan-RAG 自适应视图数 ----------
    view_count_min: int = 2
    """复杂度分析下界（极简 F_key）。"""

    view_count_max: int = 7
    """复杂度分析上界（多事实长尾 F_key）。"""

    plan_confidence_threshold: float = 0.7
    """复杂度分析置信度低于此值时回退到固定 4 视图（dense）+ 1 视图（sparse）。"""

    # ---------- v3 增量 5：MemoRAG 全局 clue（默认关闭） ----------
    enable_memo_rag_clue: bool = False
    """是否启用全库 clue 预扫描，主表配置关闭，仅作消融。"""

    memo_rag_clue_top_n: int = 100
    """启用时，预扫描保留候选簇大小。"""

    # ---------- LLM 调用 ----------
    llm_model: str = "gpt-4o-mini"
    """所有内部 LLM 调用统一模型名。"""

    llm_temperature: float = 0.0
    """温度，固定 0 以保证可复现。"""

    llm_api_key: str = ""
    """OpenAI 兼容鉴权；走 EvalMem 现有 OpenAI-compatible endpoint。"""

    llm_base_url: str = "https://api.openai.com/v1"
    """OpenAI 兼容 base URL。"""

    llm_timeout_sec: int = 120
    """单次 LLM 调用超时，与 llm_assist._chat_json 对齐。"""

    # ---------- Embedding ----------
    embedding_model: str = "Qwen3-Embedding-0.6B"
    """密集向量模型；通过外部 embed_fn 注入实际实现。"""

    embedding_dim: int = 1024
    """向量维度，仅作元数据记录。"""

    # ---------- 调试与诊断 ----------
    verbose_diagnostics: bool = True
    """retrieve() 的 HighRecallResponse.diagnostics 是否包含 per-stage 详细信息。"""

    enable_majority_voting: bool = False
    """LLM 精排是否做 3 次取中位数（论文风险表所述）。"""

    def __post_init__(self) -> None:
        """基础参数合法性校验（不连真实下游，只保证数值合理）。"""
        if not (0.0 < self.tau_cover <= 1.0):
            raise ValueError(f"tau_cover must be in (0, 1], got {self.tau_cover}")
        if self.max_pool_size < 1:
            raise ValueError(f"max_pool_size must be >= 1, got {self.max_pool_size}")
        if not (0.0 <= self.round_jaccard_threshold <= 1.0):
            raise ValueError(
                f"round_jaccard_threshold must be in [0, 1], got {self.round_jaccard_threshold}"
            )
        if not (0.0 <= self.epsilon_new_ratio <= 1.0):
            raise ValueError(f"epsilon_new_ratio must be in [0, 1], got {self.epsilon_new_ratio}")
        if self.consecutive_stable_rounds < 1:
            raise ValueError(
                f"consecutive_stable_rounds must be >= 1, got {self.consecutive_stable_rounds}"
            )
        if not (0.0 < self.tau_quality <= 1.0):
            raise ValueError(f"tau_quality must be in (0, 1], got {self.tau_quality}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if len(self.granularity_thresholds) != 3:
            raise ValueError("granularity_thresholds must have exactly 3 entries")
        if self.view_count_min > self.view_count_max:
            raise ValueError("view_count_min must be <= view_count_max")
        for required_key in ("fragment", "turn", "summary", "document"):
            if required_key not in self.granularity_k_map:
                raise ValueError(f"granularity_k_map missing key: {required_key}")
