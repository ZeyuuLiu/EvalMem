# -*- coding: utf-8 -*-
"""
SGLang Generator

A high-performance generator using SGLang Engine for local model inference.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from .base import BaseGenerator
from .config import SGLangGeneratorConfig


class SGLangGenerator(BaseGenerator):
    """
    使用 SGLang Engine 的生成器
    支持高性能批量推理
    """
    config: SGLangGeneratorConfig

    def __init__(self, config: SGLangGeneratorConfig | Dict[str, Any]):
        if isinstance(config, dict):
            config = SGLangGeneratorConfig(**config)
        super().__init__(config)
        
        # 模型配置
        self.model_name_or_path = self.config.model_name_or_path or self.config.model_name
        if not self.model_name_or_path:
            raise ValueError("model_name_or_path or model_name is required in config")
        
        self.context_length = self.config.context_length
        self.tp_size = self.config.tp_size
        self.dp_size = self.config.dp_size
        self.ep_size = self.config.ep_size
        self.random_seed = self.config.random_seed
        self.enable_thinking = self.config.enable_thinking
        
        # 采样参数
        self.temperature = self.config.temperature
        self.top_p = self.config.top_p
        self.top_k = self.config.top_k
        self.max_tokens = self.config.max_tokens
        self.n = self.config.n
        
        # 批量处理配置
        self.batch_size = self.config.batch_size
        self.system_prompt = self.config.system_prompt
        
        # 延迟初始化 engine
        self._engine = None
        self._tokenizer = None
        
        # 是否在初始化时加载模型
        if self.config.auto_init:
            self._init_engine()

    def _init_engine(self):
        """初始化 SGLang Engine"""
        try:
            import sglang as sgl
        except ImportError:
            raise ImportError(
                "sglang is required for SGLangGenerator. "
                "Install it with: pip install sglang"
            )
        
        print(f"Current Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        print(f"Initializing SGLangGenerator with Model: {self.model_name_or_path}")
        print(f"Context Length: {self.context_length}, TP Size: {self.tp_size}, "
              f"DP Size: {self.dp_size}, EP Size: {self.ep_size}")
        print(f"Random Seed: {self.random_seed}, Batch Size: {self.batch_size}, "
              f"Enable Thinking: {self.enable_thinking}")
        
        self._engine = sgl.Engine(
            model_path=self.model_name_or_path,
            context_length=self.context_length,
            tp_size=self.tp_size,
            dp_size=self.dp_size,
            ep_size=self.ep_size,
            random_seed=self.random_seed,
        )
        self._tokenizer = self._engine.tokenizer_manager.tokenizer

    @property
    def engine(self):
        """获取 engine，如果未初始化则自动初始化"""
        if self._engine is None:
            self._init_engine()
        return self._engine

    @property
    def tokenizer(self):
        """获取 tokenizer，如果未初始化则自动初始化"""
        if self._tokenizer is None:
            self._init_engine()
        return self._tokenizer

    def _get_sampling_params(
        self, 
        schema: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """构建采样参数"""
        params = {
            "n": self.n,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_new_tokens": self.max_tokens,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
        }
        
        # 如果提供了 JSON Schema，添加 guided_json
        if schema is not None:
            params["json_schema"] = json.dumps(schema) if isinstance(schema, dict) else schema
        
        # 合并额外参数
        if extra_params:
            params.update(extra_params)
        
        return params

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
            messages = [{"role": "user", "content": prompt}]
        
        # 添加系统提示
        if self.system_prompt and not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        
        return messages

    def _apply_chat_template(
        self, 
        messages_list: List[List[Dict[str, str]]]
    ) -> List[str]:
        """应用聊天模板将消息转换为输入文本"""
        input_texts = []
        for messages in messages_list:
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            input_texts.append(input_text)
        return input_texts

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

    def generate_single(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        schema: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        生成单个响应
        """
        msgs = self._build_messages(prompt, messages)
        input_texts = self._apply_chat_template([msgs])
        
        sampling_params = self._get_sampling_params(schema, extra_params)
        
        outputs = self.engine.generate(
            input_texts,
            sampling_params=sampling_params,
        )
        
        text = outputs[0]["text"]
        
        out: Dict[str, Any] = {
            "text": text,
            "json": None,
            "response": outputs[0],
        }
        
        # 如果提供了 schema，尝试解析 JSON
        if schema is not None:
            out["json"] = self._extract_json(text)
        
        return out

    def generate_batch(
        self,
        prompts: Optional[List[str]] = None,
        messages_list: Optional[List[List[Dict[str, str]]]] = None,
        schema: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        批量生成响应
        """
        if (prompts is None) and (not messages_list):
            raise ValueError("Either prompts or messages_list is required.")
        if (prompts is not None) and messages_list:
            raise ValueError("Pass either prompts or messages_list, not both.")

        # 构建消息列表
        if prompts is not None:
            if isinstance(prompts, str):
                prompts = [prompts]
            messages_list = [self._build_messages(p, None) for p in prompts]
        else:
            # 为每个消息列表添加系统提示
            messages_list = [self._build_messages(None, msgs) for msgs in messages_list]

        sampling_params = self._get_sampling_params(schema, extra_params)
        results: List[Dict[str, Any]] = []
        
        # 分批处理
        num_batches = (len(messages_list) + self.batch_size - 1) // self.batch_size
        iterator = range(0, len(messages_list), self.batch_size)
        
        if show_progress:
            iterator = tqdm(iterator, desc="Generating", total=num_batches)
        
        for i in iterator:
            batch_messages = messages_list[i:i + self.batch_size]
            input_texts = self._apply_chat_template(batch_messages)
            
            outputs = self.engine.generate(
                input_texts,
                sampling_params=sampling_params,
            )
            
            for j, output in enumerate(outputs):
                text = output["text"]
                result: Dict[str, Any] = {
                    "text": text,
                    "json": None,
                    "response": output,
                }
                
                # 如果提供了 schema，尝试解析 JSON
                if schema is not None:
                    result["json"] = self._extract_json(text)
                
                results.append(result)
        
        return results

    def shutdown(self):
        """关闭 engine 释放资源"""
        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None
            self._tokenizer = None

    def __del__(self):
        """析构时自动关闭 engine"""
        self.shutdown()

    @classmethod
    def from_config(cls, config) -> "SGLangGenerator":
        """从配置创建实例"""
        if hasattr(config, "__dict__"):
            return cls(config.__dict__)
        return cls(config)
