# -*- coding: utf-8 -*-
"""
GAM Web Application

A Flask-based web interface for building, managing and browsing GAM.

Uses TextGAMAgent for chunking and organizing content.
Uses TextChatAgent for Q&A over the GAM knowledge base.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from flask import Flask

from .helpers import DEFAULT_OUTPUT_BASE
from .routes import (
    browse_bp,
    pages_bp,
    pipeline_bp,
    research_bp,
    video_pipeline_bp,
    video_research_bp,
    long_horizontal_bp,
    custom_api_bp,
)


def create_app(
    generator=None,
    video_generator=None,
    segmentor=None,
    output_base: str = "",
) -> Flask:
    """
    创建 Flask 应用

    Args:
        generator: LLM generator 实例（可选，如果提供则支持自动组织）
        video_generator: 多模态 LLM generator（用于 VideoChatAgent 的 inspect_video，可选）
        segmentor: 视频片段描述 LLM generator（用于 VideoGAMAgent 的 segmentor，可选）
        output_base: Pipeline 输出根目录（可选，默认使用 DEFAULT_OUTPUT_BASE）
    """
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max (for video)

    # Store shared state in app config
    app.config["GENERATOR"] = generator
    app.config["VIDEO_GENERATOR"] = video_generator
    app.config["SEGMENTOR"] = segmentor
    app.config["OUTPUT_BASE"] = output_base or DEFAULT_OUTPUT_BASE
    app.config["GAM_CACHE"]: Dict = {}

    # Register blueprints – text
    app.register_blueprint(pages_bp)
    app.register_blueprint(pipeline_bp)
    app.register_blueprint(browse_bp)
    app.register_blueprint(research_bp)

    # Register blueprints – video
    app.register_blueprint(video_pipeline_bp)
    app.register_blueprint(video_research_bp)

    # Register blueprints – long horizontal
    app.register_blueprint(long_horizontal_bp)

    # Register blueprints – custom api
    app.register_blueprint(custom_api_bp)

    return app


def run_server(
    generator=None,
    video_generator=None,
    segmentor=None,
    output_base: str = "",
    host: str = "0.0.0.0",
    port: int = 5000,
    debug: bool = True,
):
    """
    运行 GAM Web 平台

    Args:
        generator: LLM generator 实例（必需，用于 TextGAMAgent / TextChatAgent / VideoGAMAgent）
        video_generator: 多模态 LLM generator（用于 VideoChatAgent 的 inspect_video，可选）
        segmentor: 视频片段描述 LLM generator（用于 VideoGAMAgent 的 segmentor，可选）
        output_base: Pipeline 输出根目录（可选，默认使用 DEFAULT_OUTPUT_BASE）
        host: 主机地址
        port: 端口号
        debug: 是否开启调试模式
    """
    app = create_app(
        generator=generator,
        video_generator=video_generator,
        segmentor=segmentor,
        output_base=output_base,
    )
    actual_output_base = app.config["OUTPUT_BASE"]

    print(f"\n{'='*60}")
    print(f"🧠 GAM Platform - Text & Video Pipeline")
    print(f"{'='*60}")
    print(f"")
    print(f"🔗 Text Platform: http://{host}:{port}/")
    print(f"🎬 Video Platform: http://{host}:{port}/video")
    print(f"🔄 Long Horizontal: http://{host}:{port}/long-horizontal")
    print(f"")
    print(f"📦 Default output: {actual_output_base}/[timestamp]/")
    print(f"   ├── chunks_*/  # TextGAMAgent chunks 输出")
    print(f"   ├── gam/       # TextGAMAgent GAM 输出")
    print(f"   └── video_*/   # VideoGAMAgent 输出")
    print(f"")
    if generator:
        print(f"✅ LLM enabled: Full pipeline available")
        print(f"   - TextGAMAgent: 智能文本切分 + 目录组织")
        print(f"   - TextChatAgent: 智能文本问答")
        print(f"   - VideoGAMAgent: 视频分析 + GAM 构建")
        print(f"   - VideoChatAgent: 视频知识库问答")
    else:
        print(f"⚠️  No LLM configured!")
        print(f"   Pipeline requires an LLM generator.")
        print(f"   Please provide a generator when calling run_server().")
    print(f"")
    print(f"Press Ctrl+C to stop.")
    print(f"{'='*60}\n")

    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    # 尝试导入 generator（需要用户配置）
    generator = None
    try:
        from gam.generators.openai_generator import OpenAIGenerator
        from gam.generators.config import OpenAIGeneratorConfig

        # 这里可以从环境变量读取配置
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            config = OpenAIGeneratorConfig(api_key=api_key)
            generator = OpenAIGenerator(config)
            print("✅ Using OpenAI generator from environment variable")
        else:
            print("⚠️  OPENAI_API_KEY not found in environment variables")
            print("   Running without LLM generator. Some features will be unavailable.")
    except Exception as e:
        print(f"⚠️  Could not initialize generator: {e}")
        print("   Running without LLM generator. Some features will be unavailable.")

    # 运行服务器
    run_server(generator=generator)
