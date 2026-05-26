from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .config import GeneratorConfig


class BaseGenerator(ABC):
    """
    Base generator for text or multimodal models.
    """

    def __init__(self, config: GeneratorConfig | dict[str, Any]):
        if isinstance(config, dict):
            self.config = GeneratorConfig(**config)
        else:
            self.config = config

    @abstractmethod
    def generate_single(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        schema: dict[str, Any] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Return: {"text": str, "json": dict|None, "response": dict}
        """
        raise NotImplementedError

    @abstractmethod
    def generate_batch(
        self,
        prompts: list[str] | None = None,
        messages_list: list[list[dict[str, str]]] | None = None,
        schema: dict[str, Any] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return: [{"text": str, "json": dict|None, "response": dict}, ...]
        """
        raise NotImplementedError
