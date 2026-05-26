from __future__ import annotations

"""
MemWiki v4 实体 / 时间 / 主题归一化（v3 优化 3）。

三个独立 Normalizer：
- :class:`EntityNormalizer` — alias_map 离线构建 + 查询期归一
- :class:`TimeNormalizer` — 三层时间解析（ISO / 相对 / 模糊）
- :class:`TopicNormalizer` — 受控主题词表 + 最近邻回退

约束（与 v3 §优化 3 / v2 §五的接口契约一致）：
- 实体别名仅依赖 record 自身上下文聚类，**不接触** ``sample.f_key`` /
  ``sample.answer_gold`` / ``sample.evidence_*``，由 :class:`LeakAuditor` 强制审计。
"""

import re
from dataclasses import dataclass
from pathlib import Path

from memory_eval.memwiki.schema import TimeAnchor


# ---------------------------------------------------------------------------
# 实体归一
# ---------------------------------------------------------------------------


@dataclass
class EntityCluster:
    canonical: str
    variants: list[str]


class EntityNormalizer:
    """
    实体别名归一器。

    工作流：
    1. :meth:`build_alias_map` 离线扫描所有 record，先用 spaCy NER 抽实体，
       再让 LLM 对发散表层做聚类，产出 alias -> canonical 字典。
    2. :meth:`normalize` 在 build / index / query 阶段分别调用，把任何表层
       归一到 canonical 形式。
    """

    def __init__(self) -> None:
        self._alias_map: dict[str, str] = {}

    def build_alias_map(self, all_records: list[dict], llm_cfg) -> dict[str, str]:
        """
        构建 alias -> canonical 字典。

        Args:
            all_records: 单 sample 的全量原始 record 列表。
            llm_cfg: :class:`LLMAssistConfig` 实例，用于 LLM 聚类调用。

        Returns:
            alias_map：键为小写形式的 variant，值为 canonical 全名。
        """
        # Deterministic baseline: keep speaker names canonical and add common
        # short-name aliases. This does not touch gold answers or evidence ids.
        speakers: list[str] = []
        for record in all_records or []:
            meta = record.get("meta", {}) if isinstance(record, dict) else {}
            if not isinstance(meta, dict):
                meta = {}
            speaker = str(meta.get("speaker") or record.get("speaker") or "").strip()
            if speaker:
                speakers.append(speaker)

        canonical_by_key: dict[str, str] = {}
        for speaker in speakers:
            key = speaker.lower()
            canonical_by_key.setdefault(key, speaker)

        prefix_candidates: dict[str, list[str]] = {}
        for speaker in canonical_by_key.values():
            first = speaker.split()[0].strip()
            if len(first) >= 4:
                prefix_candidates.setdefault(first[:3].lower(), []).append(speaker)
                prefix_candidates.setdefault(first[:4].lower(), []).append(speaker)

        for alias, candidates in prefix_candidates.items():
            unique = sorted(set(candidates))
            if len(unique) == 1:
                canonical_by_key.setdefault(alias, unique[0])

        self._alias_map = canonical_by_key
        return self._alias_map

    def normalize(self, raw_name: str) -> str:
        """查询 alias_map，未命中时返回原值（首字母大写规范化）。"""
        if not raw_name:
            return ""
        key = raw_name.strip().lower()
        return self._alias_map.get(key, raw_name.strip())

    def load_alias_map(self, alias_map: dict[str, str]) -> None:
        self._alias_map = {k.lower(): v for k, v in alias_map.items()}


# ---------------------------------------------------------------------------
# 时间归一
# ---------------------------------------------------------------------------


_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_ISO_MONTH_RE = re.compile(r"\b(\d{4})-(\d{2})\b")
_RELATIVE_KEYWORDS = {
    "yesterday", "today", "tomorrow",
    "last week", "next week",
    "last month", "next month",
    "last year", "next year",
    "two days ago", "three days ago", "a week ago", "a month ago",
}
_FUZZY_KEYWORDS = {
    "a while back", "recently", "some time ago", "long ago",
    "lately", "back then", "in the past",
}


class TimeNormalizer:
    """
    三层时间解析（v3 优化 3.2）。

    L1 ISO/英文/数字日期 → 直接转换（certainty=high）；
    L2 相对时间 → 基于 ``session_datetime`` 锚点回算（certainty=medium）；
    L3 模糊时间 → 仅设置 raw 与 certainty=fuzzy，不参与精确匹配。
    """

    def parse(self, raw_time: str, session_datetime: str | None = None) -> TimeAnchor:
        if not raw_time:
            return TimeAnchor(iso="", session=None, certainty="fuzzy", raw="")
        text = raw_time.strip()
        # L1: ISO date
        m = _ISO_DATE_RE.search(text)
        if m:
            return TimeAnchor(iso=m.group(0), session=None, certainty="high", raw=text)
        m = _ISO_MONTH_RE.search(text)
        if m:
            return TimeAnchor(iso=m.group(0), session=None, certainty="medium", raw=text)
        # L2: relative
        lower = text.lower()
        if any(kw in lower for kw in _RELATIVE_KEYWORDS):
            iso = self._anchor_relative(lower, session_datetime)
            return TimeAnchor(iso=iso, session=None, certainty="medium", raw=text)
        # L3: fuzzy
        if any(kw in lower for kw in _FUZZY_KEYWORDS):
            return TimeAnchor(iso="", session=None, certainty="fuzzy", raw=text)
        # 未识别 → 退化为 fuzzy
        return TimeAnchor(iso="", session=None, certainty="fuzzy", raw=text)

    def _anchor_relative(self, text: str, session_datetime: str | None) -> str:
        # TODO: 真正实现 dateutil.parser.parse 与基于 session_datetime 的偏移计算
        # 目前只回填空字符串，由调用方在 certainty=medium 标识下做软匹配
        return ""


# ---------------------------------------------------------------------------
# 主题归一
# ---------------------------------------------------------------------------


class TopicNormalizer:
    """
    主题受控词表归一（v3 优化 3.3）。

    构造时接受词表 yaml 路径；运行期对 LLM 输出做：
    1. 精确命中 → 直接采用；
    2. Levenshtein 最近邻 ≤ 2 → 回退到 canonical；
    3. 向量最近邻 cos > 0.85 → 回退；
    4. 仍无匹配 → 落到 ``"other"``。
    """

    def __init__(self, vocabulary_path: str) -> None:
        self._vocabulary_path = vocabulary_path
        self._vocab: list[str] = []
        self._vocab_set: set[str] = set()
        self._load_vocabulary(vocabulary_path)
        # TODO: 预计算每个 vocab item 的 embedding 用于第 3 步回退

    def normalize(self, raw_topics: list[str]) -> list[str]:
        out: list[str] = []
        for topic in raw_topics or []:
            t = topic.strip().lower().replace(" ", "_")
            if not t:
                continue
            if t in self._vocab_set:
                out.append(t)
                continue
            nearest = self._nearest_neighbor(t)
            out.append(nearest or "other")
        # 去重保持顺序
        seen: set[str] = set()
        deduped: list[str] = []
        for x in out:
            if x not in seen:
                seen.add(x)
                deduped.append(x)
        return deduped

    def _nearest_neighbor(self, topic: str) -> str | None:
        """Levenshtein + 向量最近邻双重匹配。"""
        best: tuple[int, str] | None = None
        for candidate in self._vocab:
            dist = self._levenshtein(topic, candidate)
            if best is None or dist < best[0]:
                best = (dist, candidate)
        if best and best[0] <= 2:
            return best[1]
        # TODO: 实现 embedding 最近邻 cos > 0.85 命中
        return None

    def _load_vocabulary(self, vocabulary_path: str) -> None:
        candidates = [Path(vocabulary_path)]
        candidates.append(Path(__file__).resolve().parent / vocabulary_path)
        vocab: list[str] = []
        for path in candidates:
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("- "):
                    item = stripped[2:].strip().strip("'\"").lower().replace(" ", "_")
                    if item:
                        vocab.append(item)
            if vocab:
                break
        if "other" not in vocab:
            vocab.append("other")
        seen: set[str] = set()
        self._vocab = []
        for item in vocab:
            if item not in seen:
                seen.add(item)
                self._vocab.append(item)
        self._vocab_set = set(self._vocab)

    def _levenshtein(self, a: str, b: str) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            cur = [i]
            for j, cb in enumerate(b, start=1):
                cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (0 if ca == cb else 1)))
            prev = cur
        return prev[-1]

    @property
    def vocabulary(self) -> list[str]:
        return list(self._vocab)


__all__ = ["EntityNormalizer", "TimeNormalizer", "TopicNormalizer", "EntityCluster"]
