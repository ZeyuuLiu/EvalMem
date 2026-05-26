#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Video GAM — Request (Q&A) Example

使用 Workflow("video") 对已构建的 Video GAM 进行问答。

Usage:
    python request.py [gam_path] [question]

Examples:
    python request.py                                                          # 使用默认路径和问题
    python request.py ./output/chunk_build_gam                                 # 指定 GAM 路径
    python request.py ./output/chunk_build_gam "视频中发生了什么？"                # 指定路径和问题
"""

import sys
from pathlib import Path
from datetime import datetime

from gam import Workflow


def main():
    # ----------------------------------------------------------------
    # 1. 解析参数
    # ----------------------------------------------------------------
    default_gam = Path(__file__).parent / "output" / "chunk_build_gam"
    default_question = "总结一下这个视频的主要内容。"

    gam_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_gam
    question = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else default_question

    if not gam_path.exists():
        print(f"❌ GAM 不存在: {gam_path}")
        print(f"\nUsage: python {Path(__file__).name} [gam_path] [question]")
        print(f"💡 提示: 请先使用 add.py 构建 Video GAM")
        return

    # ----------------------------------------------------------------
    # 2. 创建 Workflow  (只需要这一步！)
    # ----------------------------------------------------------------
    wf = Workflow(
        "video",
        gam_dir=gam_path,
        # LLM config — set via env vars GAM_MODEL, GAM_API_BASE, GAM_API_KEY
        # or pass explicitly here:
        # model="gpt-4o",
        # api_base="https://api.openai.com/v1",
        # api_key="sk-xxx",
        max_tokens=4096,
        temperature=0.3,
        # video_model="gpt-4o",
        # video_api_base="https://api.openai.com/v1",
        video_fps=1.0,
        video_max_resolution=480,
        max_iterations=20,
        verbose=True,
    )

    print(f"📂 GAM 路径: {gam_path}")
    print(f"🤖 模型: {wf.model}")
    print(f"\n📂 GAM 结构:")
    print(wf.get_tree_view(depth=3))
    print(f"\n📋 问题: {question}\n")

    # ----------------------------------------------------------------
    # 3. 问答
    # ----------------------------------------------------------------
    start = datetime.now()
    result = wf.request(question)
    duration = (datetime.now() - start).total_seconds()

    # ----------------------------------------------------------------
    # 4. 显示结果
    # ----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📝 答案:")
    print("=" * 80)
    print(result.answer or "(未生成答案)")
    print("=" * 80)

    print(f"\n📚 来源 ({len(result.sources)}):")
    for i, src in enumerate(result.sources, 1):
        print(f"   {i}. {src}")

    print(f"\n📖 读取的文件 ({len(result.files_read)}):")
    for i, f in enumerate(result.files_read, 1):
        print(f"   {i}. {f}")

    if hasattr(result, "confidence"):
        print(f"\n✅ 置信度: {result.confidence:.2%}")

    print(f"\n⏱️  耗时: {duration:.2f} 秒")


if __name__ == "__main__":
    main()
