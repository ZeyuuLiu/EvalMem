# -*- coding: utf-8 -*-
"""
GAM CLI - Command Line Interface

提供命令行工具：
- gam-add: 向 GAM 添加内容（支持 text / video 两种类型）
- gam-request: 基于 GAM 进行问答（支持 text / video 两种类型）
- gam-skill-prompt: 输出 Skill 描述文本（用于注入到其他 Agent 的 system prompt）

使用示例:
    # 添加文件到 text GAM
    gam-add --type text --gam-dir ./my_gam --input paper.pdf

    # 添加视频到 video GAM
    gam-add --type video --gam-dir ./my_video_gam --input ./video_dir

    # 基于 text GAM 问答
    gam-request --type text --gam-dir ./my_gam --question "What is the main conclusion?"

    # 基于 video GAM 问答
    gam-request --type video --gam-dir ./my_video_gam --question "What happens in the video?"

    # 获取 skill prompt
    gam-skill-prompt --gam-dir ./my_gam
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional


def _build_generator(args):
    """根据命令行参数构建 GAM Agent Generator 实例"""
    from .generators.openai_generator import OpenAIGenerator
    from .generators.config import OpenAIGeneratorConfig

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Error: --api-key 或环境变量 OPENAI_API_KEY 必须提供", file=sys.stderr)
        sys.exit(1)

    config = OpenAIGeneratorConfig(
        model_name=args.model,
        base_url=args.api_base,
        api_key=api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    return OpenAIGenerator(config)


def _build_chat_generator(args):
    """根据命令行参数构建 Chat Agent Generator 实例（回退到 GAM Agent 配置）"""
    chat_model = getattr(args, "chat_model", None)
    chat_api_base = getattr(args, "chat_api_base", None)
    chat_api_key = getattr(args, "chat_api_key", None)

    if not chat_model and not chat_api_base and not chat_api_key:
        return _build_generator(args)

    from .generators.openai_generator import OpenAIGenerator
    from .generators.config import OpenAIGeneratorConfig

    config = OpenAIGeneratorConfig(
        model_name=chat_model or args.model,
        base_url=chat_api_base or args.api_base,
        api_key=chat_api_key or args.api_key or os.environ.get("OPENAI_API_KEY", ""),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    return OpenAIGenerator(config)


def _build_workspace(gam_dir: str):
    """构建 LocalWorkspace"""
    from .workspaces.local_workspace import LocalWorkspace

    gam_path = Path(gam_dir).resolve()
    gam_path.mkdir(parents=True, exist_ok=True)
    return LocalWorkspace(root_path=str(gam_path))


def _build_tree(gam_dir: str, workspace):
    """构建或加载 GAMTree"""
    from .core.tree import GAMTree

    gam_path = Path(gam_dir).resolve()
    gam_path.mkdir(parents=True, exist_ok=True)

    try:
        tree = GAMTree.from_disk(gam_path, workspace)
    except Exception:
        tree = GAMTree.create(gam_path, name=gam_path.name)
    return tree


def _build_video_tree(gam_dir: str, workspace):
    """构建或加载 VideoGAMTree"""
    from .core.tree import VideoGAMTree

    gam_path = Path(gam_dir).resolve()
    gam_path.mkdir(parents=True, exist_ok=True)

    try:
        tree = VideoGAMTree.from_disk(gam_path, workspace)
    except Exception:
        tree = VideoGAMTree.create_empty(gam_path, name=gam_path.name)
    return tree


# ========== 共享参数 ==========

def _add_common_args(parser: argparse.ArgumentParser):
    """添加通用的 LLM/workspace 参数"""
    parser.add_argument(
        "--type", "-t",
        type=str,
        required=True,
        choices=["text", "video"],
        help="GAM 类型：text（文本）或 video（视频）",
    )
    parser.add_argument(
        "--gam-dir",
        type=str,
        required=True,
        help="GAM 目录路径（会自动创建）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("GAM_MODEL", "gpt-4o-mini"),
        help="LLM 模型名称（默认: gpt-4o-mini，可通过 GAM_MODEL 环境变量设置）",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=os.environ.get("GAM_API_BASE", "https://api.openai.com/v1"),
        help="API base URL（默认: https://api.openai.com/v1，可通过 GAM_API_BASE 环境变量设置）",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("GAM_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
        help="API key（可通过 GAM_API_KEY 或 OPENAI_API_KEY 环境变量设置）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("GAM_MAX_TOKENS", "4096")),
        help="LLM 最大生成 token 数（默认: 4096）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("GAM_TEMPERATURE", "0.3")),
        help="LLM 温度（默认: 0.3）",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否打印详细信息（默认: 开启）",
    )


# ========== gam-add ==========

def cli_add():
    """
    CLI 入口: gam-add

    根据 --type 参数选择添加文本或视频内容到 GAM 知识库。
    """
    parser = argparse.ArgumentParser(
        prog="gam-add",
        description="向 GAM 知识库添加内容（支持 text / video）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  # 添加单个文件（text）
  gam-add --type text --gam-dir ./my_gam --input paper.pdf

  # 添加多个文件
  gam-add --type text --gam-dir ./my_gam --input paper1.pdf --input paper2.txt

  # 添加文本内容
  gam-add --type text --gam-dir ./my_gam --content "Some text to remember"

  # 同时添加文件和文本
  gam-add --type text --gam-dir ./my_gam --input paper.pdf --content "Additional notes"

  # 不分块，将每个输入作为整体
  gam-add --type text --gam-dir ./my_gam --input paper.pdf --no-chunking

  # 添加视频
  gam-add --type video --gam-dir ./my_video_gam --input ./video_dir

  # 视频：指定 segmentor 使用不同模型
  gam-add --type video --gam-dir ./my_video_gam --input ./video_dir \\
      --model gpt-4o --segmentor-model gpt-4o-mini

  # 视频：不使用字幕进行描述生成
  gam-add --type video --gam-dir ./my_video_gam --input ./video_dir --no-caption-subtitles
""",
    )

    _add_common_args(parser)

    # text add 参数
    parser.add_argument(
        "--input", "-i",
        type=str,
        action="append",
        default=None,
        help="输入文件/目录路径（可多次指定；text 类型为文件路径，video 类型为视频目录路径）",
    )
    parser.add_argument(
        "--content", "-c",
        type=str,
        action="append",
        default=None,
        help="[text] 直接输入文本内容（可多次指定）",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="",
        help="[text] 可选的上下文/说明信息",
    )
    parser.add_argument(
        "--chunking",
        dest="use_chunking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="[text] 是否启用智能分块（默认: 开启）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="[text] 可选：保存原始 chunk 的目录",
    )
    parser.add_argument(
        "--force-reorganize",
        action="store_true",
        default=False,
        help="[text] 强制触发目录结构重组",
    )
    parser.add_argument(
        "--memory-workers",
        type=int,
        default=4,
        help="[text] 并行生成 memory 的工作线程数（默认: 4）",
    )

    # video add 参数
    parser.add_argument(
        "--segmentor-model",
        type=str,
        default=None,
        help="[video] Segmentor LLM 模型名称（默认与 --model 相同）",
    )
    parser.add_argument(
        "--segmentor-api-base",
        type=str,
        default=None,
        help="[video] Segmentor API base URL（默认与 --api-base 相同）",
    )
    parser.add_argument(
        "--segmentor-api-key",
        type=str,
        default=None,
        help="[video] Segmentor API key（默认与 --api-key 相同）",
    )
    parser.add_argument(
        "--caption-subtitles",
        dest="caption_with_subtitles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="[video] 描述生成时是否包含字幕信息（默认: 开启）",
    )

    args = parser.parse_args()

    if args.type == "text":
        _do_text_add(args, parser)
    else:
        _do_video_add(args, parser)


def _do_text_add(args, parser):
    """执行 text add 逻辑"""
    if not args.input and not args.content:
        parser.error("text 类型至少需要提供 --input 或 --content")

    generator = _build_generator(args)
    workspace = _build_workspace(args.gam_dir)
    tree = _build_tree(args.gam_dir, workspace)

    from .agents.text_gam_agent import TextGAMAgent

    agent = TextGAMAgent(
        generator=generator,
        tree=tree,
        workspace=workspace,
        use_chunking=args.use_chunking,
        auto_save=True,
        verbose=args.verbose,
        memory_workers=args.memory_workers,
    )

    input_files = [Path(f) for f in args.input] if args.input else None
    contents = args.content if args.content else None

    result = agent.add(
        input_file=input_files,
        content=contents,
        context=args.context,
        output_dir=args.output_dir,
        force_reorganize=args.force_reorganize,
    )

    if args.verbose:
        print(f"\n{'='*60}")
        print("✅ gam-add (text) 完成!")
        print(f"GAM 目录: {Path(args.gam_dir).resolve()}")
        if hasattr(result, 'created_files'):
            print(f"创建文件数: {len(result.created_files)}")
        if hasattr(result, 'new_directories'):
            print(f"新建目录数: {len(result.new_directories)}")
        print(f"{'='*60}\n")


def _do_video_add(args, parser):
    """执行 video add 逻辑"""
    if not args.input:
        parser.error("video 类型需要提供 --input（视频目录路径）")
    if len(args.input) > 1:
        parser.error("video 类型仅支持一个 --input（视频目录路径）")

    generator = _build_generator(args)

    if args.segmentor_model:
        from .generators.openai_generator import OpenAIGenerator
        from .generators.config import OpenAIGeneratorConfig

        seg_config = OpenAIGeneratorConfig(
            model_name=args.segmentor_model,
            base_url=args.segmentor_api_base or args.api_base,
            api_key=args.segmentor_api_key or args.api_key or os.environ.get("OPENAI_API_KEY", ""),
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        segmentor = OpenAIGenerator(seg_config)
    else:
        segmentor = generator

    workspace = _build_workspace(args.gam_dir)
    tree = _build_video_tree(args.gam_dir, workspace)

    from .agents.video_gam_agent import VideoGAMAgent

    agent = VideoGAMAgent(
        planner=generator,
        segmentor=segmentor,
        workspace=workspace,
        tree=tree,
    )

    input_path = Path(args.input[0]).resolve()
    result = agent.add(
        input_path=input_path,
        verbose=args.verbose,
        caption_with_subtitles=args.caption_with_subtitles,
    )

    if args.verbose:
        print(f"\n{'='*60}")
        print("✅ gam-add (video) 完成!")
        print(f"GAM 目录: {Path(args.gam_dir).resolve()}")
        print(f"视频分段数: {result.segment_num}")
        print(f"{'='*60}\n")


# ========== gam-request ==========

def cli_request():
    """
    CLI 入口: gam-request

    根据 --type 参数选择基于文本或视频 GAM 进行问答。
    """
    parser = argparse.ArgumentParser(
        prog="gam-request",
        description="基于 GAM 知识库进行问答（支持 text / video）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  # text 基本问答
  gam-request --type text --gam-dir ./my_gam --question "What is the main conclusion?"

  # text 指定系统提示
  gam-request --type text --gam-dir ./my_gam -q "Summarize" --system-prompt "You are a research assistant."

  # text 增加探索轮数
  gam-request --type text --gam-dir ./my_gam -q "详细分析实验结果" --max-iter 20

  # video 基本问答
  gam-request --type video --gam-dir ./my_video_gam --question "What happens in the video?"

  # video 使用专门的视觉分析模型
  gam-request --type video --gam-dir ./my_video_gam -q "Describe the scene" \\
      --video-model gpt-4o

  # video 调整视频分析参数
  gam-request --type video --gam-dir ./my_video_gam -q "What color is the car?" \\
      --video-fps 2.0 --video-max-resolution 720

  # 输出 JSON 格式（text / video 均支持）
  gam-request --type text --gam-dir ./my_gam -q "What methods were used?" --json
""",
    )

    _add_common_args(parser)

    # Chat Agent 配置（可单独指定，默认回退到 GAM Agent 配置）
    parser.add_argument(
        "--chat-model",
        type=str,
        default=os.environ.get("GAM_CHAT_MODEL"),
        help="Chat Agent LLM 模型名称（默认与 --model 相同，可通过 GAM_CHAT_MODEL 环境变量设置）",
    )
    parser.add_argument(
        "--chat-api-base",
        type=str,
        default=os.environ.get("GAM_CHAT_API_BASE"),
        help="Chat Agent API base URL（默认与 --api-base 相同，可通过 GAM_CHAT_API_BASE 环境变量设置）",
    )
    parser.add_argument(
        "--chat-api-key",
        type=str,
        default=os.environ.get("GAM_CHAT_API_KEY"),
        help="Chat Agent API key（默认与 --api-key 相同，可通过 GAM_CHAT_API_KEY 环境变量设置）",
    )

    # 共享 request 参数
    parser.add_argument(
        "--question", "-q",
        type=str,
        required=True,
        help="用户问题",
    )
    parser.add_argument(
        "--system-prompt", "-s",
        type=str,
        default="",
        help="系统提示词（可选）",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=10,
        help="最大探索迭代轮数（默认: 10）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="以 JSON 格式输出完整结果",
    )

    # video request 特有参数
    parser.add_argument(
        "--video-model",
        type=str,
        default=None,
        help="[video] 多模态视觉分析 LLM 模型名称（用于 inspect_video 工具，默认与 --model 相同）",
    )
    parser.add_argument(
        "--video-api-base",
        type=str,
        default=None,
        help="[video] 多模态视觉分析 LLM API base URL（默认与 --api-base 相同）",
    )
    parser.add_argument(
        "--video-api-key",
        type=str,
        default=None,
        help="[video] 多模态视觉分析 LLM API key（默认与 --api-key 相同）",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=1.0,
        help="[video] 视频分析采样帧率（默认: 1.0 fps）",
    )
    parser.add_argument(
        "--video-max-resolution",
        type=int,
        default=480,
        help="[video] 视频分析最大分辨率（默认: 480）",
    )

    args = parser.parse_args()

    if args.type == "text":
        _do_text_request(args)
    else:
        _do_video_request(args)


def _do_text_request(args):
    """执行 text request 逻辑"""
    chat_gen = _build_chat_generator(args)
    workspace = _build_workspace(args.gam_dir)
    tree = _build_tree(args.gam_dir, workspace)

    from .agents.text_chat_agent import TextChatAgent

    agent = TextChatAgent(
        generator=chat_gen,
        tree=tree,
        workspace=workspace,
        max_iterations=args.max_iter,
        verbose=args.verbose,
    )

    result = agent.request(
        system_prompt=args.system_prompt,
        user_prompt=args.question,
        max_iter=args.max_iter,
    )

    _print_request_result(result, args)


def _do_video_request(args):
    """执行 video request 逻辑"""
    chat_gen = _build_chat_generator(args)

    if args.video_model:
        from .generators.openai_generator import OpenAIGenerator
        from .generators.config import OpenAIGeneratorConfig

        video_config = OpenAIGeneratorConfig(
            model_name=args.video_model,
            base_url=args.video_api_base or args.api_base,
            api_key=args.video_api_key or args.api_key or os.environ.get("OPENAI_API_KEY", ""),
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        video_generator = OpenAIGenerator(video_config)
    else:
        video_generator = None

    workspace = _build_workspace(args.gam_dir)
    tree = _build_video_tree(args.gam_dir, workspace)

    from .agents.video_chat_agent import VideoChatAgent

    agent = VideoChatAgent(
        generator=chat_gen,
        tree=tree,
        workspace=workspace,
        video_generator=video_generator,
        max_iterations=args.max_iter,
        verbose=args.verbose,
        video_fps=args.video_fps,
        video_max_resolution=args.video_max_resolution,
    )

    result = agent.request(
        system_prompt=args.system_prompt,
        user_prompt=args.question,
        max_iter=args.max_iter,
    )

    _print_request_result(result, args)


def _print_request_result(result, args):
    """统一的 request 结果输出"""
    if args.json:
        import json
        output = {
            "question": result.question,
            "answer": result.answer,
            "sources": result.sources,
            "confidence": result.confidence,
            "notes": result.notes,
            "files_read": result.files_read,
            "dirs_explored": result.dirs_explored,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"📝 问题: {result.question}")
        print(f"{'='*60}")
        print(f"\n💡 回答:\n{result.answer}")
        if result.sources:
            print(f"\n📚 来源: {', '.join(result.sources)}")
        if result.confidence:
            print(f"\n🎯 置信度: {result.confidence}")
        print(f"\n{'='*60}")


# ========== gam-skill-prompt ==========

def cli_skill_prompt():
    """
    CLI 入口: gam-skill-prompt

    输出 GAM 的 Skill 描述文本，可直接注入到其他 Agent 的 system prompt 中。
    """
    parser = argparse.ArgumentParser(
        prog="gam-skill-prompt",
        description="输出 GAM Skill 描述文本（用于注入到 Agent system prompt）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  # 输出 skill prompt（直接打印到终端）
  gam-skill-prompt

  # 指定 GAM 目录（会替换到示例中的路径）
  gam-skill-prompt --gam-dir ./my_gam

  # 保存到文件
  gam-skill-prompt --gam-dir ./my_gam > skill.txt

  # 在 Python 中使用
  # from gam import get_skill_prompt
  # skill = get_skill_prompt(gam_dir="./my_gam")
  # system_prompt = f"You are a research agent.\\n\\n{skill}"
""",
    )

    parser.add_argument(
        "--gam-dir",
        type=str,
        default="/path/to/gam",
        help="GAM 目录路径（会替换到 skill prompt 中的示例路径，默认: /path/to/gam）",
    )

    args = parser.parse_args()

    from .prompts.skill_prompts import get_skill_prompt
    print(get_skill_prompt(gam_dir=args.gam_dir))
