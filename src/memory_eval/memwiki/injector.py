from __future__ import annotations

"""
MemWiki auxiliary-index injection.

The injector never performs query-time retrieval.  It builds ordinary
retrieval-friendly records and asks the adapter to ingest them into the same
runtime context used by the original memory-system retrieval path.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

from memory_eval.memwiki.aux_record import AuxRecord, normalize_aux_records
from memory_eval.memwiki.builder import MemWikiBuilder
from memory_eval.memwiki.config import MemWikiConfig


@dataclass(frozen=True)
class InjectionReport:
    sample_id: str
    aux_record_count: int
    injected: bool
    adapter_class: str
    mode: str = "memwiki_aux_index"
    warnings: list[str] = field(default_factory=list)
    build_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemWikiInjector:
    """Build and inject MemWiki aux records for one adapter runtime context."""

    def __init__(
        self,
        config: MemWikiConfig | None = None,
        *,
        llm_cfg: Any | None = None,
        max_aux_records: int | None = None,
    ) -> None:
        self.config = config or MemWikiConfig()
        self.llm_cfg = llm_cfg
        self.max_aux_records = max_aux_records
        self.last_build_metrics: dict[str, Any] = {}

    def build_aux_records(self, adapter: Any, run_ctx: Any, sample_id: str) -> list[AuxRecord]:
        builder = MemWikiBuilder(config=self.config, llm_cfg=self.llm_cfg)
        records = builder.build_aux_records(
            adapter=adapter,
            run_ctx=run_ctx,
            sample_id=sample_id,
            max_records=self.max_aux_records,
        )
        self.last_build_metrics = dict(builder.metrics)
        return records

    def inject(self, adapter: Any, run_ctx: Any, sample_id: str) -> InjectionReport:
        warnings: list[str] = []
        aux_records = self.build_aux_records(adapter, run_ctx, sample_id)
        memory_records = normalize_aux_records(aux_records)
        if not memory_records:
            return InjectionReport(
                sample_id=sample_id,
                aux_record_count=0,
                injected=False,
                adapter_class=adapter.__class__.__name__,
                warnings=["MemWiki produced no auxiliary records."],
                build_metrics=dict(self.last_build_metrics),
            )

        ingest_fn = getattr(adapter, "ingest_aux_records", None)
        if not callable(ingest_fn):
            return InjectionReport(
                sample_id=sample_id,
                aux_record_count=len(memory_records),
                injected=False,
                adapter_class=adapter.__class__.__name__,
                warnings=["Adapter does not expose ingest_aux_records()."],
                build_metrics=dict(self.last_build_metrics),
            )

        try:
            ingest_fn(run_ctx, memory_records)
        except Exception as exc:
            return InjectionReport(
                sample_id=sample_id,
                aux_record_count=len(memory_records),
                injected=False,
                adapter_class=adapter.__class__.__name__,
                warnings=[f"Adapter aux ingestion failed: {exc}"],
                build_metrics=dict(self.last_build_metrics),
            )

        if isinstance(run_ctx, dict):
            reports = run_ctx.setdefault("memwiki_injection_reports", [])
            reports.append(
                {
                    "sample_id": sample_id,
                    "aux_record_count": len(memory_records),
                    "mode": "memwiki_aux_index",
                    "build_metrics": dict(self.last_build_metrics),
                }
            )
        return InjectionReport(
            sample_id=sample_id,
            aux_record_count=len(memory_records),
            injected=True,
            adapter_class=adapter.__class__.__name__,
            warnings=warnings,
            build_metrics=dict(self.last_build_metrics),
        )


__all__ = ["InjectionReport", "MemWikiInjector"]
