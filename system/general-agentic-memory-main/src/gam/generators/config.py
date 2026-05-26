from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Any

class GeneratorConfig(BaseModel):
    """Base configuration for all generators."""
    model_name: str = Field(..., description="Name of the model to use")
    temperature: float = Field(0.0, description="Sampling temperature")
    top_p: float = Field(1.0, description="Top-p sampling parameter")
    max_tokens: int = Field(512, description="Maximum number of tokens to generate")
    system_prompt: Optional[str] = Field(None, description="Default system prompt")
    extra_params: dict[str, Any] = Field(default_factory=dict, description="Additional parameters")

class OpenAIGeneratorConfig(GeneratorConfig):
    """Configuration for OpenAIGenerator."""
    api_key: str = Field("empty", description="OpenAI API key")
    base_url: str = Field("http://localhost:8000/v1", description="OpenAI base URL")
    thread_count: Optional[int] = Field(None, description="Number of threads for batch generation")
    timeout: float = Field(60.0, description="Request timeout in seconds")
    enable_thinking: bool = Field(True, description="Enable thinking mode for supported models")

class SGLangGeneratorConfig(GeneratorConfig):
    """Configuration for SGLangGenerator."""
    # Override defaults for SGLang
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.8, description="Top-p sampling parameter")
    
    # SGLang specific fields
    model_name_or_path: Optional[str] = Field(None, description="Model path or name")
    context_length: int = Field(32768, description="Maximum context length")
    tp_size: int = Field(1, description="Tensor parallelism size")
    dp_size: int = Field(1, description="Data parallelism size")
    ep_size: int = Field(1, description="Expert parallelism size")
    random_seed: int = Field(42, description="Random seed")
    enable_thinking: bool = Field(False, description="Enable thinking mode")
    top_k: int = Field(20, description="Top-k sampling parameter")
    n: int = Field(1, description="Number of completions to generate")
    batch_size: int = Field(128, description="Batch size for generation")
    auto_init: bool = Field(True, description="Automatically initialize the engine")
