"""
粒度感知 K 路由 — v2 创新点 3。

基于记忆库的平均 token 数把被测系统归入四档：
  fragment (<80 tok)  -> K=20
  turn     (<200 tok) -> K=12
  summary  (<600 tok) -> K=6
  document (>=600)    -> K=3
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from memory_eval.eval_core.agentic_rag.config import AgenticRAGConfig


@dataclass(frozen=True)
class GranularityProfile:
    """粒度统计与 K 路由结果。"""

    avg_tokens: float
    """记忆库样本平均 token 数。"""

    granularity_type: str
    """{fragment, turn, summary, document}。"""

    recommended_k: int
    """该档位的 Top-K。"""

    drill_down_enabled: bool
    """是否启用粒度下钻（summary / document 档可向更细粒度展开）。"""

    sample_size: int = 0
    """统计依据的样本量，便于诊断。"""


def detect_granularity(
    memory_corpus: list[dict[str, Any]],
    cfg: AgenticRAGConfig,
) -> GranularityProfile:
    """
    一次性统计记忆库平均 token 数并归档。

    参数:
        memory_corpus: List[Dict] 记忆条目，含 text 字段
        cfg: AgenticRAGConfig 提供阈值与 K 映射

    返回:
        GranularityProfile dataclass

    TODO: replace simple word-split with a proper tokenizer
          (与 EvalMem llm_assist 使用的 tokenizer 对齐)。
    """
    if not memory_corpus:
        # 空库时回退到 summary 档（中间值），避免极端 K
        return GranularityProfile(
            avg_tokens=0.0,
            granularity_type="summary",
            recommended_k=cfg.granularity_k_map["summary"],
            drill_down_enabled=cfg.drill_down_enabled,
            sample_size=0,
        )

    total_tokens = 0
    sampled = 0
    for record in memory_corpus:
        text = str(record.get("text", "")) if isinstance(record, dict) else str(record)
        if not text:
            continue
        total_tokens += _count_tokens(text)
        sampled += 1
    avg = total_tokens / sampled if sampled else 0.0

    granularity_type = _classify_granularity(avg, cfg.granularity_thresholds)
    return GranularityProfile(
        avg_tokens=avg,
        granularity_type=granularity_type,
        recommended_k=cfg.granularity_k_map[granularity_type],
        drill_down_enabled=cfg.drill_down_enabled and granularity_type in ("summary", "document"),
        sample_size=sampled,
    )


def route_top_k(profile: GranularityProfile) -> int:
    """便捷封装：返回 profile.recommended_k。

    保留为独立函数以便上层 retriever 后续叠加策略（如多视图 K 微调）。
    """
    return int(profile.recommended_k)


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

# 含中日韩 + 英文 token 切分。粒度统计层不需要完美 tokenizer，仅需稳定可比。
_TOKEN_RE = re.compile(r"[\w一-鿿]+", flags=re.UNICODE)


def _count_tokens(text: str) -> int:
    """skeleton 用简单正则切分。

    TODO: replace with the same tokenizer as the encoding LLM
          (e.g. tiktoken for gpt-* models, qwen tokenizer for Qwen3).
    """
    return len(_TOKEN_RE.findall(text))


def _classify_granularity(avg_tokens: float, thresholds: list[int]) -> str:
    """thresholds = [t0, t1, t2] -> [<t0:fragment, <t1:turn, <t2:summary, else:document]."""
    t0, t1, t2 = thresholds
    if avg_tokens < t0:
        return "fragment"
    if avg_tokens < t1:
        return "turn"
    if avg_tokens < t2:
        return "summary"
    return "document"
