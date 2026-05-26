"""
统一 LLM 客户端：OpenAI 兼容 chat completions 接口。

提供：
- chat_json: 调 LLM 并解析 JSON 响应（带 retry + negative hint）
- chat_text: 调 LLM 拿纯文本
- 缓存（基于 prompt+model+temperature 的 SHA256）

设计：
- 失败时按指数 backoff 重试
- JSON 解析失败时把上次响应作为 negative example 重试
- 全部调用经过本模块以便审计 / 计费
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests

logger = logging.getLogger(__name__)


# =====================================================================
# 异常
# =====================================================================

class LLMCallError(Exception):
    pass


class LLMJSONParseError(LLMCallError):
    pass


# =====================================================================
# 主客户端
# =====================================================================

@dataclass
class LLMClient:
    api_key: str
    base_url: str
    timeout_sec: int = 60
    max_retries: int = 3
    retry_backoff_sec: int = 2
    cache_dir: Path | None = None
    on_call: Callable[[str, str], None] | None = None  # (prompt, response_text) hook

    def _cache_key(self, model: str, temperature: float, prompt: str, max_tokens: int) -> str:
        h = hashlib.sha256()
        h.update(model.encode("utf-8"))
        h.update(f"{temperature:.3f}".encode("utf-8"))
        h.update(str(max_tokens).encode("utf-8"))
        h.update(prompt.encode("utf-8"))
        return h.hexdigest()

    def _cache_get(self, key: str) -> str | None:
        if self.cache_dir is None:
            return None
        f = self.cache_dir / f"{key}.txt"
        if f.exists():
            return f.read_text(encoding="utf-8")
        return None

    def _cache_put(self, key: str, value: str) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        f = self.cache_dir / f"{key}.txt"
        f.write_text(value, encoding="utf-8")

    def _post(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str | None = None,
    ) -> str:
        """单次 HTTP POST，不做缓存与重试。"""
        url = self.base_url.rstrip("/") + "/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise LLMCallError(f"HTTP error: {e}")

        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return str(content).strip()
        except (KeyError, IndexError, ValueError) as e:
            raise LLMCallError(f"unexpected response shape: {e}; body={resp.text[:300]}")

    def chat_text(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
        use_cache: bool = True,
    ) -> str:
        """调 LLM 拿文本，带缓存与 retry。"""
        cache_key = self._cache_key(model, temperature, prompt, max_tokens)
        if use_cache:
            cached = self._cache_get(cache_key)
            if cached is not None:
                logger.debug(f"cache hit: {cache_key[:8]}")
                if self.on_call:
                    self.on_call(prompt, cached)
                return cached

        last_err = None
        for attempt in range(self.max_retries):
            try:
                content = self._post(model, prompt, temperature, max_tokens, system_prompt)
                if use_cache:
                    self._cache_put(cache_key, content)
                if self.on_call:
                    self.on_call(prompt, content)
                return content
            except LLMCallError as e:
                last_err = e
                logger.warning(f"LLM attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_backoff_sec * (2 ** attempt))
        raise LLMCallError(f"LLM failed after {self.max_retries} attempts: {last_err}")

    def chat_json(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
        use_cache: bool = True,
    ) -> dict:
        """调 LLM 并解析 JSON。失败时把上次响应作为 negative example 重试。"""
        if system_prompt is None:
            system_prompt = "Return ONLY a strict JSON object. No markdown, no explanation."

        last_response = None
        last_err = None
        for attempt in range(self.max_retries):
            current_prompt = prompt
            if last_response is not None and attempt > 0:
                current_prompt = (
                    prompt
                    + f"\n\n[上次输出格式无法解析为 JSON，请严格输出一个 JSON 对象。"
                    + f"上次错误样本（前 200 字）：{last_response[:200]}]"
                )
            try:
                content = self.chat_text(
                    current_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                    use_cache=use_cache and attempt == 0,
                )
                parsed = parse_json_loose(content)
                return parsed
            except LLMJSONParseError as e:
                last_response = e.args[0] if e.args else str(e)
                last_err = e
                logger.warning(f"JSON parse fail attempt {attempt + 1}/{self.max_retries}: {e}")
            except LLMCallError as e:
                last_err = e
                logger.warning(f"LLM call fail attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_backoff_sec * (2 ** attempt))
        raise LLMCallError(f"chat_json failed after {self.max_retries} attempts: {last_err}")


# =====================================================================
# JSON 容错解析
# =====================================================================

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def parse_json_loose(text: str) -> dict:
    """
    宽松 JSON 解析：
    1. 直接 json.loads
    2. 去掉 ```json ... ``` 代码块包装
    3. 尝试找第一个 { 与最后一个 }，截取再解析
    """
    if not text:
        raise LLMJSONParseError(text)

    cleaned = text.strip()

    # 1. direct
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # 2. strip code fence
    m = _CODE_FENCE_RE.match(cleaned)
    if m:
        inner = m.group(1).strip()
        try:
            result = json.loads(inner)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        cleaned = inner

    # 3. find first { and last }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        snippet = cleaned[start: end + 1]
        try:
            result = json.loads(snippet)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    raise LLMJSONParseError(text)


# =====================================================================
# 工厂
# =====================================================================

def build_client(cfg, cache_subdir: str = "llm_responses") -> LLMClient:
    """
    从 DatasetConfig 构造默认客户端。
    cfg 须为 builder.config.DatasetConfig 实例。
    """
    cache_path = cfg.paths.cache_root / cache_subdir
    return LLMClient(
        api_key=cfg.llm.api_key,
        base_url=cfg.llm.base_url,
        timeout_sec=cfg.llm.request_timeout_sec,
        max_retries=cfg.llm.max_retries,
        retry_backoff_sec=cfg.llm.retry_backoff_sec,
        cache_dir=cache_path,
    )


__all__ = [
    "LLMClient",
    "LLMCallError",
    "LLMJSONParseError",
    "parse_json_loose",
    "build_client",
]
