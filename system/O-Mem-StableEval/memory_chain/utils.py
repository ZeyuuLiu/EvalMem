#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Personal AI team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import os
from typing import Any, Callable, Iterable


def ensure_directory_exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


class StableEvalJSONError(RuntimeError):
    pass


def _strip_code_fence(text: str) -> str:
    s = str(text or "").strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if not lines:
        return s
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    if lines and lines[0].strip().lower() == "json":
        lines = lines[1:]
    return "\n".join(lines).strip()


def _extract_balanced_json(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        raise StableEvalJSONError("empty model content")
    starts = [i for i, ch in enumerate(s) if ch in "{["]
    for start in starts:
        opener = s[start]
        closer = "}" if opener == "{" else "]"
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(s)):
            ch = s[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "\"":
                    in_string = False
                continue
            if ch == "\"":
                in_string = True
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return s[start : idx + 1]
    raise StableEvalJSONError("no balanced json payload found")


def parse_json_response(content: Any) -> Any:
    raw = _strip_code_fence(str(content or ""))
    if not raw:
        raise StableEvalJSONError("empty model content")
    try:
        return json.loads(raw)
    except Exception:
        payload = _extract_balanced_json(raw)
        return json.loads(payload)


def require_keys(obj: Any, required_keys: Iterable[str], label: str = "json object") -> Any:
    if not isinstance(obj, dict):
        raise StableEvalJSONError(f"{label} is not a dict")
    missing = [k for k in required_keys if k not in obj]
    if missing:
        raise StableEvalJSONError(f"{label} missing keys: {missing}")
    return obj


async def call_llm_json_with_retries(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    retries: int = 5,
    retry_delay_sec: float = 1.0,
    validator: Callable[[Any], Any] | None = None,
    label: str = "json call",
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            kwargs = {"model": model, "messages": messages, "temperature": temperature}
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            response = await client.chat.completions.create(**kwargs)
            parsed = parse_json_response(response.choices[0].message.content)
            if validator is not None:
                parsed = validator(parsed)
            return parsed
        except Exception as exc:
            last_exc = exc
            print(f"[StableEval] {label} failed on attempt {attempt}/{retries}: {exc}")
            if attempt < retries:
                await asyncio.sleep(retry_delay_sec)
    raise StableEvalJSONError(f"{label} failed after {retries} attempts: {last_exc}")


async def call_llm_text_with_retries(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    retries: int = 5,
    retry_delay_sec: float = 1.0,
    label: str = "text call",
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            kwargs = {"model": model, "messages": messages, "temperature": temperature}
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            response = await client.chat.completions.create(**kwargs)
            return response
        except Exception as exc:
            last_exc = exc
            print(f"[StableEval] {label} failed on attempt {attempt}/{retries}: {exc}")
            if attempt < retries:
                await asyncio.sleep(retry_delay_sec)
    raise RuntimeError(f"{label} failed after {retries} attempts: {last_exc}")


def call_llm_json_sync_with_retries(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    retries: int = 5,
    validator: Callable[[Any], Any] | None = None,
    label: str = "sync json call",
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            kwargs = {"model": model, "messages": messages, "temperature": temperature}
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            response = client.chat.completions.create(**kwargs)
            parsed = parse_json_response(response.choices[0].message.content)
            if validator is not None:
                parsed = validator(parsed)
            return parsed
        except Exception as exc:
            last_exc = exc
            print(f"[StableEval] {label} failed on attempt {attempt}/{retries}: {exc}")
    raise StableEvalJSONError(f"{label} failed after {retries} attempts: {last_exc}")

