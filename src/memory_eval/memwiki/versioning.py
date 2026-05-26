from __future__ import annotations

"""
MemWiki v4 多版本管理（§四）。

关键概念：
- 每个 :class:`WikiEntry` 维护 versions list，每个 version 有
  ``[valid_from_session, valid_to_session)`` 半开区间；
- ``valid_to_session is None`` 表示当前 latest；
- 状态变更（同 SPO 不同 O）通过 :meth:`detect_state_change` 识别，触发新版本创建；
- 时间锚点查询（latest / explicit / fuzzy）由 :meth:`query_by_time` 路由。

与 v3 ``deprecated_by`` 字段向后兼容：旧版本同时设置 ``deprecated_by =
new_version_id``，便于不支持多版本的检索路径继续使用。
"""

from dataclasses import dataclass

from memory_eval.memwiki.schema import AtomicFact, WikiEntry, WikiVersion


@dataclass
class TimeQuery:
    """检索时的时间过滤条件。"""

    query_type: str           # "latest" | "explicit" | "fuzzy"
    target_session: int | None = None
    target_iso_date: str | None = None


class VersionManager:
    """
    管理 :class:`WikiEntry.versions` 链，提供"检测变更 + 闭旧开新 + 时间查询"
    三套接口。
    """

    def __init__(self) -> None:
        self._version_counter: dict[str, int] = {}

    # ------------------------------------------------------------------
    # 状态变更检测（v3 §4.3 SPO 模式）
    # ------------------------------------------------------------------

    def detect_state_change(self, page: WikiEntry, new_record: dict, llm_cfg=None) -> bool:
        """
        判断 new_record 是否对 page 中已有 (subject, predicate) 给出了新的 object。

        Args:
            page: 已存在的 entry，扫描其 atomic_facts。
            new_record: 新到达的 record dict（至少包含 ``text``、``id``、``session_id``）。
            llm_cfg: 可选 LLM 配置；若提供，则在 SPO 表层冲突时调 LLM 二次确认
                （区分"状态变更"与"并列偏好"）。

        Returns:
            True 表示需要触发新版本创建。
        """
        # TODO: 调用 NER + 简单关系抽取，从 new_record["text"] 抽出
        #       new_facts = list[AtomicFact]
        new_facts: list[AtomicFact] = []  # placeholder
        for nf in new_facts:
            for of in page.atomic_facts:
                if (
                    nf.subject == of.subject
                    and nf.predicate == of.predicate
                    and nf.object != of.object
                ):
                    if llm_cfg is None:
                        return True
                    # TODO: 调用 build_state_change_detection_prompt + _chat_json
                    #       LLM 二次确认是否真为状态变更
                    return True
        return False

    # ------------------------------------------------------------------
    # 创建新版本（关键：闭旧开新 + 单调）
    # ------------------------------------------------------------------

    def create_new_version(
        self,
        page: WikiEntry,
        new_content: str,
        session: int,
        new_atomic_facts: list[AtomicFact] | None = None,
        new_source_record_ids: list[str] | None = None,
    ) -> WikiVersion:
        """
        关闭当前 latest 版本于 ``session - 0`` （valid_to=session），
        新建版本 valid_from=session。

        若旧版本 valid_from > session，则视为乱序写入，抛出
        :class:`ValueError`。
        """
        cur = page.current_version
        if cur is not None:
            if cur.valid_from_session > session:
                raise ValueError(
                    f"Out-of-order version: existing valid_from={cur.valid_from_session}"
                    f" > new session={session}"
                )
            cur.valid_to_session = session

        version_id = self._next_version_id(page.entry_id)
        new_ver = WikiVersion(
            version_id=version_id,
            valid_from_session=session,
            valid_to_session=None,
            content=new_content,
            atomic_facts=list(new_atomic_facts or []),
            source_record_ids=list(new_source_record_ids or []),
        )
        page.versions.append(new_ver)
        page.last_updated_session = session
        # v3 兼容：旧版本同时打 deprecated_by 标记
        if cur is not None:
            page.deprecated_by = version_id
        return new_ver

    def _next_version_id(self, entry_id: str) -> str:
        n = self._version_counter.get(entry_id, 0) + 1
        self._version_counter[entry_id] = n
        return f"{entry_id}::v{n}"

    # ------------------------------------------------------------------
    # 时间锚点查询
    # ------------------------------------------------------------------

    def query_by_time(self, page: WikiEntry, time_query: TimeQuery) -> WikiVersion | None:
        """
        根据 ``time_query.query_type`` 路由：
        - ``"latest"`` → 返回 ``valid_to_session is None`` 的版本
        - ``"explicit"`` → 返回 ``valid_from <= target_session < valid_to`` 的版本
        - ``"fuzzy"`` → 返回 latest（fuzzy 由调用方在 RRF 后用 LLM 选择）
        """
        if not page.versions:
            return None
        qt = time_query.query_type
        if qt == "latest" or qt == "fuzzy":
            return page.current_version
        if qt == "explicit":
            target = time_query.target_session
            if target is None:
                return page.current_version
            for ver in page.versions:
                lo = ver.valid_from_session
                hi = ver.valid_to_session if ver.valid_to_session is not None else float("inf")
                if lo <= target < hi:
                    return ver
            return None
        return page.current_version

    def list_versions(self, page: WikiEntry) -> list[WikiVersion]:
        """完整版本链（用于 ?explain=true 溯源）。"""
        return list(page.versions)


__all__ = ["VersionManager", "TimeQuery"]
