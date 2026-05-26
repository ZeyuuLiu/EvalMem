from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..generators.base import BaseGenerator
from ..core.tree import BaseTree
from ..tools.base import BaseTool


class BaseChatAgent(ABC):
    def __init__(
        self,
        generator: BaseGenerator,
        tree: BaseTree | None = None,
        tools: list[BaseTool] | None = None,
    ):
        self.generator = generator
        self.tree = tree
        self.tools = tools or []

    @abstractmethod
    def request(
        self,
        system_prompt: str,
        user_prompt: str,
        max_iter: int = 3,
        limit_tokens: int | None = None,
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError
