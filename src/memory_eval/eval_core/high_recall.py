from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol


@dataclass(frozen=True)
class HighRecallRequest:
    query: str
    f_key: List[str] = field(default_factory=list)
    evidence_texts: List[str] = field(default_factory=list)
    memory_corpus: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HighRecallResponse:
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class EncodingHighRecallRetriever(Protocol):
    def retrieve(self, request: HighRecallRequest) -> HighRecallResponse:
        ...
