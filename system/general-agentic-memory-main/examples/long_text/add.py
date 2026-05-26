#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Text GAM — Add Example

使用 Workflow("text") 从文档构建 GAM 记忆系统。

Usage:
    python add.py [input_file] [output_dir]

Examples:
    python add.py                                         # 使用默认路径
    python add.py ./my_document.pdf                       # 指定输入文件
    python add.py ./my_document.pdf ./my_output           # 指定输入和输出
"""

import sys
from pathlib import Path
from datetime import datetime

from gam import Workflow


def main():
    # ----------------------------------------------------------------
    # 1. 解析参数
    # ----------------------------------------------------------------
    default_input = Path(__file__).parent.parent.parent / "gam_test" / "data" / "GAM.pdf"
    default_output = Path(__file__).parent / "output" / "GAM"

    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_input
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else default_output

    if not input_path.exists():
        print(f"❌ 输入文件不存在: {input_path}")
        print(f"\nUsage: python {Path(__file__).name} [input_file] [output_dir]")
        return

    # ----------------------------------------------------------------
    # 2. 创建 Workflow  (只需要这一步！)
    # ----------------------------------------------------------------
    wf = Workflow(
        "text",
        gam_dir=output_path,
        # LLM config — set via env vars GAM_MODEL, GAM_API_BASE, GAM_API_KEY
        # or pass explicitly here:
        # model="gpt-4o-mini",
        # api_base="https://api.openai.com/v1",
        # api_key="sk-xxx",
        max_tokens=4096,
        temperature=0.3,
        use_chunking=True,
        memory_workers=4,
        verbose=True,
    )

    print(f"📄 输入文件: {input_path}")
    print(f"📁 输出目录: {output_path}")
    print(f"🤖 模型: {wf.model}")
    print()

    # ----------------------------------------------------------------
    # 3. 构建 GAM
    # ----------------------------------------------------------------
    start = datetime.now()
    result = wf.add(input_file=input_path)
    duration = (datetime.now() - start).total_seconds()

    # ----------------------------------------------------------------
    # 4. 显示结果
    # ----------------------------------------------------------------
    print(f"\n✅ 构建完成！耗时: {duration:.2f} 秒")
    print(f"   - 总 chunks: {len(result.memorized_chunks)}")
    print(f"   - 创建文件: {len(result.created_files)}")
    print(f"\n📂 目录结构:")
    print(wf.get_tree_view(depth=5))
    print(f"\n💡 提示: 使用 request.py 对这个 GAM 进行问答")


if __name__ == "__main__":
    main()
