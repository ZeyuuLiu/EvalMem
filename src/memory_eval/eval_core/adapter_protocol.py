from __future__ import annotations

from typing import Any, Dict, List, Protocol

from memory_eval.eval_core.models import AdapterTrace


class EvalAdapterProtocol(Protocol):
    """
    Runtime adapter protocol used by evaluator engine.
    评估引擎与外部记忆系统交互时使用的统一适配器协议。
    """

    def ingest_conversation(self, sample_id: str, conversation: List[Dict[str, Any]]) -> Any:
        """
        Build runtime context for one sample.
        为单个样本构建运行时上下文（可缓存系统状态）。
        """
        ...

    def build_trace_for_query(self, run_ctx: Any, query: str, oracle_context: str, top_k: int) -> AdapterTrace:
        """
        Return all evaluator-required observations for one query.
        返回当前 query 的评估所需观测值（memory_view/retrieval/answers）。
        """
        ...


class EncodingAdapterProtocol(Protocol):
    """
    Dedicated adapter protocol for encoding probe.
    编码探针专用适配器协议：显式约束“从记忆系统读取全量记忆库并做匹配”。
    """

    def export_full_memory(self, run_ctx: Any) -> List[Dict[str, Any]]:
        """
        Export the full memory corpus M from the underlying memory system.
        导出底层记忆系统的全量记忆库 M（可序列化视图）。
        """
        ...

    def find_memory_records(
        self,
        run_ctx: Any,
        query: str,
        f_key: List[str],
        memory_corpus: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Adapter-side traversal / matching over memory corpus.
        由适配器层实现脚本遍历/匹配数据库，返回与 Q / F_key 相关记录。
        """
        ...

    def hybrid_retrieve_candidates(
        self,
        run_ctx: Any,
        query: str,
        f_key: List[str],
        evidence_texts: List[str],
        top_n: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Optional high-recall retrieval for encoding judgement.
        可选：编码层高召回候选检索（语义+关键词融合）接口。
        """
        ...


class RetrievalAdapterProtocol(Protocol):
    """
    Dedicated adapter protocol for retrieval probe.
    检索探针专用适配器协议：显式约束原始检索结果 C_original 的导出。
    """

    def retrieve_original(
        self,
        run_ctx: Any,
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Return ordered retrieval list C_original from underlying memory system.
        返回来自底层记忆系统的有序检索结果 C_original。
        """
        ...


class GenerationAdapterProtocol(Protocol):
    """
    Dedicated adapter protocol for generation probe.
    生成探针专用适配器协议：使用完美上下文 C_oracle 生成 A_oracle。
    """

    def generate_oracle_answer(
        self,
        run_ctx: Any,
        query: str,
        oracle_context: str,
    ) -> str:
        """
        Ask the underlying memory-system model to answer with oracle context.
        调用底层记忆系统模型，在完美上下文下生成答案 A_oracle。
        """
        ...

    def generate_online_answer(
        self,
        run_ctx: Any,
        query: str,
        top_k: int = 5,
    ) -> str:
        """
        Optional online answer generated from system's normal retrieval path.
        可选：使用系统正常检索路径生成 A_online。
        """
        ...
