# -*- coding: utf-8 -*-
"""
OpenAI-compatible Generator

A simple generator using OpenAI-compatible API endpoints.
"""
from __future__ import annotations

import json
import time
import os
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from tqdm import tqdm

from openai import OpenAI

from .base import BaseGenerator
from .config import OpenAIGeneratorConfig


class OpenAIGenerator(BaseGenerator):
    """
    使用 OpenAI 兼容端点的生成器
    支持标准的 Chat Completions API
    """
    config: OpenAIGeneratorConfig
    
    def __init__(self, config: OpenAIGeneratorConfig | Dict[str, Any]):
        if isinstance(config, dict):
            config = OpenAIGeneratorConfig(**config)
        super().__init__(config)
        
        # 使用 config 对象中的值
        self.model_name = self.config.model_name
        self.api_key = self.config.api_key
        self.base_url = self.config.base_url
        self.temperature = self.config.temperature
        self.top_p = self.config.top_p
        self.max_tokens = self.config.max_tokens
        self.thread_count = self.config.thread_count
        self.system_prompt = self.config.system_prompt
        self.timeout = self.config.timeout
        self.enable_thinking = self.config.enable_thinking

        # 设置环境变量
        if self.api_key is not None:
            os.environ["OPENAI_API_KEY"] = self.api_key
        if self.base_url is not None:
            os.environ["OPENAI_BASE_URL"] = self.base_url

        # 初始化 OpenAI 客户端
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url.rstrip("/"))
        self._cclient = (
            self._client.with_options(timeout=self.timeout)
            if hasattr(self._client, "with_options")
            else self._client
        )

    def _build_messages(
        self,
        prompt: Optional[str],
        messages: Optional[List[Dict[str, str]]],
    ) -> List[Dict[str, str]]:
        """构建消息列表"""
        if (prompt is None) and (not messages):
            raise ValueError("Either prompt or messages is required.")
        if (prompt is not None) and messages:
            raise ValueError("Pass either prompt or messages, not both.")
        
        # 构造 messages
        if messages is None:
            messages = [{"role": "user", "content": prompt}]  # type: ignore[arg-type]
        
        # 添加系统提示
        if self.system_prompt and not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        
        return messages

    def generate_single(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        schema: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        生成单个响应
        
        Args:
            prompt: 文本提示
            messages: 消息列表 (二选一)
            schema: JSON Schema，用于结构化输出
            extra_params: 额外参数
        
        Returns:
            {"text": str, "json": dict|None, "response": dict}
        """
        msgs = self._build_messages(prompt, messages)

        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": msgs,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

        if 'qwen' in self.model_name.lower():
            params['extra_body'] = {
                'chat_template_kwargs': {
                    'enable_thinking': self.enable_thinking
                }
            }
        
        # 如果提供了 schema，通过 extra_body 传递 guided_json
        if schema is not None:
            params['extra_body'] = params.get('extra_body', {})
            params['extra_body']['guided_json'] = schema
            params['response_format'] = {
                    'type': 'json_schema',
                    'json_schema': {
                        'name': 'output',
                        'schema': schema,
                    }
                }
        
        if extra_params:
            params.update(extra_params)
        
        out: Dict[str, Any] = {"text": None, "json": None, "response": None}

        # 重试机制
        max_retries = 20
        for attempt in range(max_retries):
            try:
                resp = self._cclient.chat.completions.create(**params)
                
                # 获取响应文本
                if (
                    resp
                    and hasattr(resp, "choices")
                    and isinstance(resp.choices, list)
                    and len(resp.choices) > 0
                ):
                    text = resp.choices[0].message.content or ""
                else:
                    raise ValueError(f"API 返回的 choices 为空或格式不正确")
                
                out["text"] = text # .split('</think>')[-1].strip()
                out["response"] = resp.model_dump()
                
                # 如果提供了 schema，尝试解析 JSON
                if schema is not None:
                    out["json"] = self._extract_json(text)
                
                break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Request failed, retry ({attempt + 1}/{max_retries}): {str(e)}")
                    print('=' * 50)
                    # 不要打印整个 msgs，因为可能非常长
                    msg_len = len(str(msgs))
                    print(f"Messages length: {msg_len}")
                    if msg_len > 1000:
                        print(f"Messages (truncated): {str(msgs)[:500]} ... {str(msgs)[-500:]}")
                    else:
                        print(msgs)
                    print('*' * 50)
                    time.sleep(20)  # 指数退避
                else:
                    print(f"请求最终失败: {str(e)}")
                    raise
        
        return out
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """从文本中提取 JSON"""
        try:
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(text[json_start:json_end])
        except json.JSONDecodeError:
            pass
        return None

    def generate_batch(
        self,
        prompts: Optional[List[str]] = None,
        messages_list: Optional[List[List[Dict[str, str]]]] = None,
        schema: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量生成响应
        """
        if (prompts is None) and (not messages_list):
            raise ValueError("Either prompts or messages_list is required.")
        if (prompts is not None) and messages_list:
            raise ValueError("Pass either prompts or messages_list, not both.")

        if prompts is not None:
            if isinstance(prompts, str):
                prompts = [prompts]
            messages_list = [[{"role": "user", "content": p}] for p in prompts]

        thread_count = self.thread_count or cpu_count()

        def _worker(msgs):
            return self.generate_single(
                messages=msgs,
                schema=schema,
                extra_params=extra_params,
            )

        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            results = list(
                tqdm(executor.map(_worker, messages_list), total=len(messages_list))
            )
        return results

    @classmethod
    def from_config(cls, config) -> "OpenAIGenerator":
        """从配置创建实例"""
        if hasattr(config, "__dict__"):
            return cls(config.__dict__)
        return cls(config)
