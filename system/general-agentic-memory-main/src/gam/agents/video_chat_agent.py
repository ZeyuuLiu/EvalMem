# -*- coding: utf-8 -*-
"""
VideoChatAgent - Video-based Chat Agent

基于 Video GAM 的聊天 Agent，使用 OpenAI Function Calling 格式进行多轮探索。
最终答案使用 <answer> XML 标签输出，便于解析。

使用示例:
```python
from gam import VideoChatAgent, VideoGAMTree, LocalWorkspace
from gam.generators import OpenAIGenerator

# 创建 workspace
workspace = LocalWorkspace(root_path="./my_video_gam")

# 加载 Video GAM
tree = VideoGAMTree.from_disk(Path("./my_video_gam"), workspace)

# 创建 generator（主 LLM，需支持 Function Calling）
generator = OpenAIGenerator(config)

# 创建多模态 generator（用于 inspect_video 工具，需支持图片输入）
video_generator = OpenAIGenerator(multimodal_config)

# 创建 video chat agent
agent = VideoChatAgent(
    generator=generator,
    tree=tree,
    workspace=workspace,
    video_generator=video_generator,
)

# 问答
result = agent.request(
    system_prompt="You are a helpful assistant.",
    user_prompt="What happens in the first segment?",
)
print(result.answer)
```

## 工作流程

1. 工具调用：使用 OpenAI Function Calling
2. 文本工具：ls, cat, grep 用于浏览 GAM 目录和阅读 README/SUBTITLES
3. 视频工具：inspect_video 用于多模态视觉分析
4. 最终答案：使用 <answer> XML 标签输出 JSON
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .chat_agent import BaseChatAgent
from ..generators.base import BaseGenerator
from ..core.tree import BaseTree, VideoGAMTree
from ..tools.base import BaseTool
from ..tools.result import ToolResult
from ..tools.ls_tool import LsTool
from ..tools.cat_tool import CatTool
from ..tools.grep_tool import GrepTool
from ..tools.inspect_video_tool import InspectVideoTool
from ..schemas.chat_schemas import ChatResult
from ..prompts.chat_prompts import VIDEO_EXPLORATION_GUIDE, SUMMARIZE_PROMPT

if TYPE_CHECKING:
    from ..workspaces.base import BaseWorkspace


class VideoChatAgent(BaseChatAgent):
    """
    Video-based Chat Agent
    
    探索 Video GAM 知识库并回答问题的 Agent。
    - 工具调用：使用 OpenAI Function Calling
    - 最终答案：使用 <answer> XML 标签输出
    
    支持的工具:
    1. ls: 列出目录结构
    2. cat: 读取文件内容（README.md, SUBTITLES.md 等）
    3. grep: 在文本中搜索关键词
    4. inspect_video: 使用多模态 LLM 分析视频片段视觉内容
    """
    
    def __init__(
        self,
        generator: BaseGenerator,
        tree: BaseTree | None = None,
        workspace: Optional["BaseWorkspace"] = None,
        tools: list[BaseTool] | None = None,
        video_generator: BaseGenerator | None = None,
        max_iterations: int = 10,
        verbose: bool = True,
        video_fps: float = 1.0,
        video_max_resolution: int = 480,
    ):
        """
        初始化 VideoChatAgent
        
        Args:
            generator: 主 LLM generator 实例（用于 ReAct 循环，需支持 Function Calling）
            tree: Video GAM 树实例（可选，用于内存模式）
            workspace: Workspace 实例（用于执行 Linux 命令和定位视频文件）
            tools: 工具列表（可选，默认自动创建 ls, cat, grep, inspect_video）
            video_generator: 多模态 LLM generator（用于 inspect_video 工具，需支持图片输入）。
                            如果不提供，则使用主 generator（需确保其支持多模态）
            max_iterations: 最大迭代次数
            verbose: 是否打印详细信息
            video_fps: 视频分析时的采样帧率（默认 1.0 fps）
            video_max_resolution: 视频分析时的最大分辨率（默认 480）
        """
        self.workspace = workspace
        
        # 视频 generator：优先使用专用的 video_generator，否则复用主 generator
        self._video_generator = video_generator or generator
        
        # 默认创建工具
        default_tools = [
            LsTool(workspace=workspace),
            CatTool(workspace=workspace),
            GrepTool(workspace=workspace),
            InspectVideoTool(
                workspace=workspace,
                generator=self._video_generator,
                fps=video_fps,
                max_resolution=video_max_resolution,
            ),
        ]
        
        # 合并用户提供的工具
        if tools is not None:
            default_tools.extend(tools)
        tools = default_tools
        
        super().__init__(generator, tree, tools)
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # 注册工具
        self._tools_by_name: Dict[str, BaseTool] = {}
        if tools:
            for tool in tools:
                self._tools_by_name[tool.name] = tool
    
    def register_tool(self, tool: BaseTool) -> None:
        """注册新工具"""
        self._tools_by_name[tool.name] = tool
        if tool not in self.tools:
            self.tools.append(tool)
    
    def get_registered_tools(self) -> List[str]:
        """获取已注册的工具名称列表"""
        return list(self._tools_by_name.keys())
    
    def get_tools_spec(self) -> List[Dict[str, Any]]:
        """
        获取所有工具的 OpenAI Function Calling 规范
        
        可以直接传给 OpenAI API:
            client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=agent.get_tools_spec()
            )
        """
        return [tool.spec() for tool in self._tools_by_name.values()]
    
    def request(
        self,
        system_prompt: str,
        user_prompt: str,
        max_iter: int | None = None,
        limit_tokens: int | None = None,
        output_dir: Path | None = None,
        action_callback: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        运行 video chat agent 回答问题

        流程：
        1. 构建初始 messages（system + user）
        2. 循环调用 LLM，处理 tool_calls（OpenAI Function Calling）
        3. 当模型输出 <answer> 标签时，解析并结束
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户问题
            max_iter: 最大迭代轮数（默认使用 self.max_iterations）
            limit_tokens: 限制 token 数（未使用）
            output_dir: 输出目录（未使用）
            action_callback: 可选的回调函数 callback(event_type, data)，
                             用于实时推送 agent 动作，如工具调用等
            **kwargs: 额外参数
            
        Returns:
            ChatResult 包含答案、来源和过程信息
        """
        if max_iter is None:
            max_iter = self.max_iterations
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Video Chat: {user_prompt[:100]}...")
            print(f"Available tools: {list(self._tools_by_name.keys())}")
            print(f"{'='*60}\n")
        
        result = ChatResult(question=user_prompt)
        
        # ========== 构建初始 messages ==========
        messages: List[Dict[str, Any]] = []
        
        # 系统提示词：包含视频知识库概览和探索指南
        readme_content = self._get_knowledge_base_overview()
        system_content = self._build_system_prompt(system_prompt, readme_content)
        messages.append({"role": "system", "content": system_content})
        
        # 用户问题
        messages.append({"role": "user", "content": user_prompt})
        
        # 获取工具规范（OpenAI Function Calling 格式）
        tools_spec = self.get_tools_spec()
        
        # 保存轨迹
        result.trajectory = json.dumps(messages, ensure_ascii=False, indent=2)
        
        if self.verbose:
            print(f"[Init] Messages initialized, {len(tools_spec)} tools available")
        
        # ========== 主循环 ==========
        for round_num in range(max_iter):
            if self.verbose:
                print(f"\n--- Round {round_num + 1}/{max_iter} ---")
            
            # 通知回调：新一轮开始
            if action_callback:
                action_callback("round", {"round": round_num + 1, "max_rounds": max_iter})
            
            try:
                # 检查消息历史长度并根据需要总结 (Summarize history if too long)
                summarize_threshold = kwargs.get("summarize_threshold", 60000)
                total_content_len = sum(len(str(m.get("content", ""))) for m in messages)
                if total_content_len > summarize_threshold and len(messages) > 3:
                    if self.verbose:
                        print(f"  [Summarize] History length ({total_content_len}) exceeds threshold ({summarize_threshold}), summarizing...")
                    
                    # 调用总结 Prompt
                    summary_msgs = messages + [{"role": "user", "content": SUMMARIZE_PROMPT}]
                    summary_response = self.generator.generate_single(messages=summary_msgs)
                    summary_text = summary_response.get("text", "(Summary failed)")
                    
                    # 重新构建 messages：保留系统 Prompt 和原始用户 Prompt
                    # messages[0] 是 system, messages[1] 是 initial user prompt
                    messages = [
                        messages[0],
                        messages[1],
                        {
                            "role": "user",
                            "content": (
                                "## PREVIOUS RESEARCH SUMMARY\n\n"
                                "The conversation history was too long and has been summarized below to save context space:\n\n"
                                f"{summary_text}\n\n"
                                "Please continue your research based on this summary and your original goal. "
                                "Use tools as needed to gather more information."
                            )
                        }
                    ]
                    
                    # 在轨迹中记录总结事件 (Replace history with summary in trajectory as well)
                    result.trajectory = f"--- Conversation Summary (Previous History {total_content_len} chars replaced) ---\n"
                    result.trajectory += f"System: {messages[0].get('content', '')[:100]}...\n"
                    result.trajectory += f"User: {messages[1].get('content', '')}\n\n"
                    result.trajectory += f"## SUMMARY OF PREVIOUS STEPS:\n{summary_text}\n"
                    result.trajectory += "---------------------------------------------------\n"
                    
                    if self.verbose:
                        print(f"  [Summarize] History replaced with summary ({len(summary_text)} chars).")

                # Step 1: 调用 LLM（带 tools 参数）
                if action_callback:
                    action_callback("thinking", {"message": "Thinking..."})
                response = self.generator.generate_single(
                    messages=messages,
                    extra_params={"tools": tools_spec},
                )
                
                # 解析响应
                raw_response = response.get("response", {})
                choice = raw_response.get("choices", [{}])[0] if raw_response else {}
                message = choice.get("message", {})
                
                # 获取 assistant 的回复内容和工具调用
                assistant_content = message.get("content", "")
                tool_calls = message.get("tool_calls") or []  # 处理 None 的情况
                
                # Step 2: 将 assistant 消息追加到 messages
                assistant_message: Dict[str, Any] = {"role": "assistant"}
                if assistant_content:
                    assistant_message["content"] = assistant_content
                if tool_calls:
                    assistant_message["tool_calls"] = tool_calls
                messages.append(assistant_message)
                
                # 更新轨迹
                result.trajectory += f"\n\n[Round {round_num + 1} - Assistant]: {assistant_content or '(tool calls only)'}"
                if tool_calls:
                    result.trajectory += f"\n  Tool calls: {[tc.get('function', {}).get('name') for tc in tool_calls]}"
                
                if self.verbose:
                    print(f"  Assistant: {len(assistant_content or '')} chars, {len(tool_calls)} tool calls")
                
                # Step 3: 检查是否输出了 <answer>（结束条件）
                if assistant_content:
                    answer_data = self._extract_answer(assistant_content)
                    if answer_data:
                        result.answer = answer_data.get("answer", "")
                        result.sources = answer_data.get("sources", result.files_read)
                        result.confidence = answer_data.get("confidence", 0.5)
                        result.notes = answer_data.get("notes", "")
                        if self.verbose:
                            print("  Found <answer>, ending loop.")
                        break
                
                # Step 4: 如果没有 <answer> 也没有 tool_calls，可能模型直接给出了文本答案
                finish_reason = choice.get("finish_reason", "")
                if finish_reason == "stop" and not tool_calls and not self._extract_answer(assistant_content or ""):
                    # 模型没用 <answer> 标签，但给出了文本回答
                    if self.verbose:
                        print("  No <answer> tag but stop, treating content as answer...")
                    result.answer = assistant_content or ""
                    result.sources = result.files_read
                    result.confidence = 0.6
                    break
                
                # Step 5: 执行工具调用
                if tool_calls:
                    for tc in tool_calls:
                        tool_id = tc.get("id", "")
                        tool_name = tc.get("function", {}).get("name", "")
                        tool_args_str = tc.get("function", {}).get("arguments", "{}")
                        
                        try:
                            tool_args = json.loads(tool_args_str)
                        except json.JSONDecodeError:
                            tool_args = {}
                        
                        if self.verbose:
                            print(f"    Executing {tool_name}: {str(tool_args)[:80]}...")
                        
                        # 通知回调：工具调用
                        if action_callback:
                            action_callback("tool_call", {
                                "tool": tool_name,
                                "args": tool_args,
                                "display": self._format_tool_display(tool_name, tool_args),
                            })
                        
                        tool_result = self._execute_tool(tool_name, tool_args)
                        result_str = self._format_tool_result(tool_result)
                        
                        # 将工具结果以 role="tool" 返回给模型
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": result_str,
                        })
                        
                        result.trajectory += f"\n\n[Tool {tool_name} Result]:\n{result_str[:1000]}..."
                        self._update_stats(tool_name, tool_result, result)
                    
                    if self.verbose:
                        print(f"  Executed {len(tool_calls)} tool call(s)")
                        
            except Exception as e:
                if self.verbose:
                    print(f"Error in round {round_num + 1}: {e}")
                    import traceback
                    traceback.print_exc()
                continue
        
        # 循环结束但没有答案（达到 max_iter）
        if not result.answer:
            if self.verbose:
                print("  Max rounds reached without answer, requesting final answer...")
            if action_callback:
                action_callback("thinking", {"message": "Generating final answer..."})
            
            messages.append({
                "role": "user",
                "content": "You have reached the maximum number of exploration rounds. Please provide your final answer NOW in <answer> tags based on what you have found.",
            })
            
            response = self.generator.generate_single(messages=messages)
            llm_output = response.get("text", "")
            result.trajectory += f"\n\n[Final]: {llm_output}"
            
            answer_data = self._extract_answer(llm_output)
            if answer_data:
                result.answer = answer_data.get("answer", "")
                result.sources = answer_data.get("sources", result.files_read)
                result.confidence = answer_data.get("confidence", 0.5)
                result.notes = answer_data.get("notes", "")
            else:
                result.answer = llm_output or "Unable to find sufficient information to answer the question."
                result.sources = result.files_read
                result.confidence = 0.3
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Video Chat Complete")
            print(f"  Answer: {result.answer[:200]}..." if len(result.answer) > 200 else f"  Answer: {result.answer}")
            print(f"  Sources: {result.sources}")
            print(f"  Confidence: {result.confidence}")
            print(f"{'='*60}\n")
        
        return result
    
    # ========== 辅助方法 ==========
    
    def _build_system_prompt(self, user_system_prompt: str, readme_content: str) -> str:
        """构建系统提示词"""
        parts = []
        
        if user_system_prompt:
            parts.append(user_system_prompt)
        
        parts.append(VIDEO_EXPLORATION_GUIDE)
        
        if readme_content:
            parts.append(
                f"## KNOWLEDGE BASE OVERVIEW\n\n"
                f"Below is the directory structure and summary of the video knowledge base. "
                f"{readme_content}"
            )
        
        return "\n\n".join(parts)
    
    def _get_knowledge_base_overview(self) -> str:
        """获取视频知识库概览
        
        优先使用根目录 README（由 VideoGAMAgent 构建时已包含标题、摘要和片段概览），
        仅在 README 为空时 fallback 到 tree_view。
        """
        if self.tree is None:
            return "(No video knowledge base loaded)"
        
        # 优先使用根 README，它已经包含了完整的概览和片段信息
        root_readme = ""
        if hasattr(self.tree, "root") and self.tree.root:
            root_readme = getattr(self.tree.root, "summary", "") or ""
        
        if root_readme:
            return root_readme
        
        # Fallback: 根 README 为空时，用 tree_view 生成目录结构
        if hasattr(self.tree, "tree_view"):
            actual_tree = self.tree.tree_view("/", depth=3)
        else:
            actual_tree = "(Unable to get directory structure)"
        
        return "### Directory Structure:\n```\n" + actual_tree + "\n```"
    
    def _extract_answer(self, text: str) -> Optional[Dict]:
        """从 LLM 输出中提取 <answer> 内容
        
        支持两种情况：
        1. 完整的 <answer>...</answer> 标签对
        2. 只有 <answer> 开头但缺少 </answer> 闭合标签（模型提前停止）
        """
        # 优先匹配完整的 <answer>...</answer>
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if answer_match:
            return self._extract_json(answer_match.group(1))
        
        # Fallback: 只有 <answer> 没有 </answer>，取 <answer> 之后的所有内容
        answer_match = re.search(r'<answer>(.*)', text, re.DOTALL)
        if answer_match:
            return self._extract_json(answer_match.group(1))
        
        return None
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """从文本中提取 JSON"""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        return None
    
    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> List[ToolResult]:
        """执行工具"""
        tool = self._tools_by_name.get(tool_name)
        if tool is None:
            return [ToolResult(
                path="",
                content=f"Error: Tool '{tool_name}' not found. Available tools: {list(self._tools_by_name.keys())}",
                score=0.0,
                rank=0,
                meta={"error": True},
            )]
        
        try:
            return tool.execute(**args)
        except Exception as e:
            if self.verbose:
                print(f"  Error executing tool {tool_name}: {e}")
            return [ToolResult(
                path="",
                content=f"Error executing {tool_name}: {str(e)}",
                score=0.0,
                rank=0,
                meta={"error": True},
            )]
    
    def _format_tool_result(self, results: List[ToolResult]) -> str:
        """格式化工具结果"""
        if not results:
            return "No results."
        
        if results[0].meta.get("error"):
            return results[0].content
        
        lines = []
        for r in results[:10]:
            if r.path:
                lines.append(f"=== {r.path} ===")
            lines.append(r.content)
            lines.append("")
        
        return "\n".join(lines).strip()
    
    def _update_stats(
        self,
        tool_name: str,
        tool_result: List[ToolResult],
        result: ChatResult,
    ) -> None:
        """更新统计信息"""
        for r in tool_result:
            if r.meta.get("error"):
                continue
            
            if tool_name == "ls":
                if r.meta.get("type") == "directory" and r.path not in result.dirs_explored:
                    result.dirs_explored.append(r.path)
            elif tool_name in ("cat", "grep"):
                if r.meta.get("type") == "file" and r.path not in result.files_read:
                    result.files_read.append(r.path)
            elif tool_name == "inspect_video":
                if r.meta.get("type") == "video_inspection" and r.path not in result.files_read:
                    result.files_read.append(r.path)
            # 通用：任何返回 file 类型结果的工具
            elif r.meta.get("type") == "file" and r.path:
                if r.path not in result.files_read:
                    result.files_read.append(r.path)
    
    def _format_tool_display(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """将工具调用格式化为可读的命令行形式"""
        if tool_name == "ls":
            paths = tool_args.get("paths", tool_args.get("path", []))
            if isinstance(paths, str):
                paths = [paths]
            return f"ls {' '.join(paths)}" if paths else "ls /"
        elif tool_name == "cat":
            paths = tool_args.get("paths", tool_args.get("path", []))
            if isinstance(paths, str):
                paths = [paths]
            return f"cat {' '.join(paths)}" if paths else "cat"
        elif tool_name == "grep":
            pattern = tool_args.get("pattern", "")
            path = tool_args.get("path", tool_args.get("paths", "/"))
            if isinstance(path, list):
                path = " ".join(path)
            return f"grep '{pattern}' {path}"
        elif tool_name == "inspect_video":
            segment = tool_args.get("target_segment", "")
            query = tool_args.get("query", "")
            return f"inspect_video {segment} '{query[:50]}...'" if len(query) > 50 else f"inspect_video {segment} '{query}'"
        else:
            args_str = " ".join(f"{k}={v}" for k, v in tool_args.items())
            return f"{tool_name} {args_str}" if args_str else tool_name
    
    # ========== 便捷方法 ==========
    
    def chat(self, question: str, max_rounds: int = None, action_callback: Any | None = None) -> ChatResult:
        """简化的问答接口"""
        return self.request(
            system_prompt="",
            user_prompt=question,
            max_iter=max_rounds,
            action_callback=action_callback,
        )
    
    def read_file(self, path: str) -> str:
        """读取文件内容（便捷方法）"""
        cat_tool = self._tools_by_name.get("cat")
        if cat_tool:
            results = cat_tool.execute(paths=[path])
            if results and not results[0].meta.get("error"):
                return results[0].content
        return ""
    
    def list_directory(self, path: str = "/") -> List[str]:
        """列出目录内容（便捷方法）"""
        ls_tool = self._tools_by_name.get("ls")
        if ls_tool:
            results = ls_tool.execute(paths=[path])
            if results and not results[0].meta.get("error"):
                return results[0].content.split("\n")
        return []
