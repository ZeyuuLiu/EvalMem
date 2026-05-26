#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Video GAM — Add Example

使用 Workflow("video") 从视频目录构建 GAM 记忆系统。

输入目录应包含：
- video.mp4      (必需)
- subtitles.srt  (可选)
- metadata.json  (可选)

Usage:
    python add.py [input_dir] [output_dir]

Examples:
    python add.py                                         # 使用默认路径
    python add.py ./my_video_dir                          # 指定输入目录
    python add.py ./my_video_dir ./my_output              # 指定输入和输出
"""

import sys
from pathlib import Path
from datetime import datetime

from gam import Workflow


def main():
    # ----------------------------------------------------------------
    # 1. 解析参数
    # ----------------------------------------------------------------
    default_input = Path(__file__).parent / "input"
    default_output = Path(__file__).parent / "output" / "chunk_build_gam"

    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_input
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else default_output

    if not input_path.exists():
        print(f"❌ 输入目录不存在: {input_path}")
        print(f"\nUsage: python {Path(__file__).name} [input_dir] [output_dir]")
        return

    # ----------------------------------------------------------------
    # 2. 创建 Workflow  (只需要这一步！)
    # ----------------------------------------------------------------
    wf = Workflow(
        "video",
        gam_dir=output_path,
        # LLM config — set via env vars GAM_MODEL, GAM_API_BASE, GAM_API_KEY
        # or pass explicitly here:
        # model="gpt-4o",
        # api_base="https://api.openai.com/v1",
        # api_key="sk-xxx",
        max_tokens=4096,
        temperature=0.3,
        # segmentor_model="gpt-4o-mini",
        # segmentor_api_base="https://api.openai.com/v1",
        verbose=True,
    )

    print(f"📹 输入目录: {input_path}")
    print(f"📁 输出目录: {output_path}")
    print(f"🤖 模型: {wf.model}")
    print()

    # ----------------------------------------------------------------
    # 3. 构建 Video GAM
    # ----------------------------------------------------------------
    start = datetime.now()
    result = wf.add(input_path=input_path, caption_with_subtitles=True)
    duration = (datetime.now() - start).total_seconds()

    # ----------------------------------------------------------------
    # 4. 显示结果
    # ----------------------------------------------------------------
    print(f"\n✅ 构建完成！耗时: {duration:.2f} 秒")
    print(f"\n📂 目录结构:")
    print(wf.get_tree_view(depth=5))
    print(f"\n💡 提示: 使用 request.py 对这个 Video GAM 进行问答")


if __name__ == "__main__":
    main()
