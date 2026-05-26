# -*- coding: utf-8 -*-
"""
InspectVideoTool - 视频片段视觉分析工具

使用多模态 LLM 分析视频片段的视觉内容。
通过 workspace 定位视频文件，通过 generator 进行视觉理解。

用法:
    inspect_video(target_segment="seg_0001", query="What objects are visible?")
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import BaseTool
from .result import ToolResult

if TYPE_CHECKING:
    from ..workspaces.base import BaseWorkspace
    from ..generators.base import BaseGenerator


class InspectVideoTool(BaseTool):
    """
    视频片段视觉分析工具
    
    通过多模态 LLM 分析视频片段内容，回答关于视觉内容的问题。
    视频片段位于 workspace 的 segments/ 目录下。
    
    使用场景:
    - 当文本描述 (README.md) 不足以回答关于视觉细节的问题时
    - 需要验证视频中的具体视觉信息时
    """
    
    name = "inspect_video"
    description = (
        "Inspect the visual content of a video segment using a multi-modal LLM. "
        "Use this tool when text descriptions (README.md, SUBTITLES.md) are insufficient "
        "to answer questions about visual details in a specific video segment. "
        "Example: inspect_video(target_segment='seg_0001', query='What color is the car?')"
    )

    def __init__(
        self,
        workspace: Optional["BaseWorkspace"] = None,
        generator: Optional["BaseGenerator"] = None,
        fps: float = 1.0,
        max_resolution: int = 480,
    ):
        """
        初始化视频分析工具
        
        Args:
            workspace: Workspace 实例，用于定位视频文件
            generator: 多模态 LLM Generator 实例（需支持图片输入）
            fps: 视频采样帧率（默认 1.0 fps）
            max_resolution: 帧图片最大分辨率（默认 480）
        """
        super().__init__(workspace=workspace)
        self.generator = generator
        self.fps = fps
        self.max_resolution = max_resolution

    def spec(self) -> Dict[str, Any]:
        """返回 OpenAI Function Calling 兼容的工具规范"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_segment": {
                            "type": "string",
                            "description": (
                                "The ID of the video segment to inspect. "
                                "For example: 'seg_0001'. Must match a directory under segments/."
                            ),
                        },
                        "query": {
                            "type": "string",
                            "description": (
                                "The question to ask about the video segment's visual content. "
                                "For example: 'What objects are visible in this segment?'"
                            ),
                        },
                    },
                    "required": ["target_segment", "query"],
                },
            },
        }

    def execute(
        self,
        target_segment: str = "",
        query: str = "",
        **kwargs: Any,
    ) -> List[ToolResult]:
        """
        分析视频片段的视觉内容
        
        Args:
            target_segment: 片段 ID，如 "seg_0001"
            query: 关于视觉内容的问题
            
        Returns:
            List[ToolResult] 包含多模态 LLM 的分析结果
        """
        if self.workspace is None:
            return [ToolResult(
                path="",
                content="Error: Workspace not initialized",
                score=0.0,
                rank=0,
                meta={"error": True},
            )]
        
        if self.generator is None:
            return [ToolResult(
                path="",
                content="Error: Multi-modal generator not initialized for video inspection",
                score=0.0,
                rank=0,
                meta={"error": True},
            )]
        
        if not target_segment or not query:
            return [ToolResult(
                path="",
                content="Error: Both target_segment and query are required",
                score=0.0,
                rank=0,
                meta={"error": True},
            )]
        
        # 定位视频片段文件
        video_path = os.path.join(
            str(self.workspace.root_path), "segments", target_segment, "video.mp4"
        )
        
        if not os.path.exists(video_path):
            return [ToolResult(
                path=f"segments/{target_segment}/video.mp4",
                content=f"Error: Video segment not found at segments/{target_segment}/video.mp4",
                score=0.0,
                rank=0,
                meta={"error": True},
            )]
        
        try:
            from ..utils import get_frame_indices, prepare_video_input, get_video_property
            
            # 获取视频属性
            video_info = get_video_property(video_path)
            duration = float(video_info["duration"])
            
            # 提取视频帧
            frame_indices = get_frame_indices(video_path, 0, duration, fps=self.fps)
            frame_base64_list, timestamps = prepare_video_input(
                video_path, frame_indices, self.max_resolution, max_workers=4
            )
            
            # 构建多模态消息
            user_content = []
            for frame_base64, timestamp in zip(frame_base64_list, timestamps):
                user_content.extend([
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_base64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": f"<{timestamp:.1f} seconds>",
                    },
                ])
            user_content.append({
                "type": "text",
                "text": (
                    f"Based on the video frames above, please answer the following question:\n"
                    f"{query}"
                ),
            })
            
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that analyzes video content. "
                        "Answer questions based on the provided video frames accurately and concisely."
                    ),
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ]
            
            # 调用多模态 LLM
            out = self.generator.generate_single(messages=messages)
            generated_text = out.get("text", "")
            
            if not generated_text:
                return [ToolResult(
                    path=f"segments/{target_segment}/video.mp4",
                    content="Error: No response from multi-modal LLM",
                    score=0.0,
                    rank=0,
                    meta={"error": True},
                )]
            
            return [ToolResult(
                path=f"segments/{target_segment}/video.mp4",
                content=generated_text,
                score=1.0,
                rank=0,
                meta={"type": "video_inspection", "segment": target_segment},
            )]
            
        except Exception as e:
            return [ToolResult(
                path=f"segments/{target_segment}/video.mp4",
                content=f"Error inspecting video segment: {str(e)}",
                score=0.0,
                rank=0,
                meta={"error": True},
            )]
