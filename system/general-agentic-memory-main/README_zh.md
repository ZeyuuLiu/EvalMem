# GAM (General Agentic Memory via Deep Research in An Agent File System)

[English](README.md) | 中文版

一个高度模块化的智能体文件系统框架，为大语言模型（LLM）提供结构化的记忆与操作环境集。支持 **文本** 和 **视频** 两种模态，提供 **Python SDK**、**CLI 命令行**、**REST API** 和 **Web 平台** 四种调用方式。

## 特性

### 1. GAM 核心特性
* 📝 **智能分块**: 基于 LLM 的文本智能切分（Chunking），自动识别语义边界。
* 🧠 **记忆生成**: 为每个文本块生成结构化记忆摘要（Memory + TLDR）。
* 📂 **层级组织**: 自动将记忆组织为层级化目录结构（Taxonomy）。
* ➕ **增量添加**: 向已有 GAM 追加新内容，无需重建。
* 🐳 **多环境支持**: 支持本地文件系统（Local）和 Docker 容器两种工作空间。
* 🔌 **灵活 LLM 后端**: 兼容 OpenAI、SGLang 等多种推理后端。

### 2. 适用任务场景
* 📄 **长文本**: 为超长文档构建层级化记忆，支持探索式智能问答。
* 🎥 **长视频**: 自动探测、分段并描述视频内容，构建长视频记忆层级。
* 🎞️ **长时程 (智能体轨迹)**: 对长序列智能体轨迹（如复杂推理步骤、工具调用日志）进行高效压缩与组织，使智能体能够在长时间的操作中有效管理上下文。

### 3. 多样化实现方式
* 🐍 **Python SDK**: 高层级 Python SDK，方便集成到智能体工作流中。
* 💻 **CLI 命令行**: 提供 `gam-add` 和 `gam-request` 统一命令行工具。
* 🚀 **REST API**: 高性能 RESTful API（FastAPI + Uvicorn），自动生成 OpenAPI 文档，内置请求校验与 CORS 支持。
* 🌐 **Web 平台**: 基于 Flask 的可视化管理与交互界面。

## 快速开始

### 安装

```bash
# 全部功能安装
pip install -e ".[all]"
```

### 使用方式概览

GAM 支持通过 Python SDK、CLI 命令行、REST API 或 Web 界面进行操作。

#### 1. Python SDK (Workflow API)
```python
from gam import Workflow
wf = Workflow("text", gam_dir="./my_gam", model="gpt-4o-mini", api_key="sk-xxx")
wf.add(input_file="paper.pdf")
result = wf.request("主要结论是什么？")
print(result.answer)
```

#### 2. CLI 命令行
```bash
# 添加内容
gam-add --type text --gam-dir ./my_gam --input paper.pdf
# 问答查询
gam-request --type text --gam-dir ./my_gam --question "主要结论是什么？"
```

#### 3. REST API
```bash
# 启动 REST API 服务（FastAPI + Uvicorn）
python examples/run_api.py --port 5001
# 交互式文档：http://localhost:5001/docs
# 查看调用示例
python examples/rest_api_client.py
```

#### 4. Web 界面
```bash
python examples/run_web.py --model gpt-4o-mini --api-key sk-xxx
```

### 环境配置

GAM Agent（记忆构建）和 Chat Agent（问答）支持独立配置：

```bash
# GAM Agent（记忆构建）
export GAM_API_KEY="sk-your-api-key"
export GAM_MODEL="gpt-4o-mini"
export GAM_API_BASE="https://api.openai.com/v1"

# Chat Agent（问答）—— 未设置时回退到 GAM Agent 配置
export GAM_CHAT_API_KEY="sk-your-chat-api-key"
export GAM_CHAT_MODEL="gpt-4o"
export GAM_CHAT_API_BASE="https://api.openai.com/v1"
```

## 使用指南

有关各组件的详细使用说明，请参阅以下文档：

* 🐍 **[Python SDK 用法](./examples/docs/sdk_usage.md)**: `Workflow` API、底层组件及高级用法。
* 💻 **[CLI 命令行用法](./examples/docs/cli_usage.md)**: `gam-add` 和 `gam-request` 命令详解。
* 🚀 **[REST API 用法](./examples/docs/rest_api_usage.md)**: RESTful API 调用与程序化集成。
* 🌐 **[Web 平台用法](./examples/docs/web_usage.md)**: 可视化管理界面的安装与启动。

## 示例

参见 [`examples/`](./examples/) 目录：

| 场景 | 说明 |
|---|---|
| [`long_text/`](./examples/long_text/) | 文本 GAM 构建与问答示例 |
| [`long_video/`](./examples/long_video/) | 视频 GAM 构建与问答示例 |
| [`long_horizon/`](./examples/long_horizon/) | 长时程智能体轨迹压缩示例（search/memorize/recall） |

## 研究代码

[`research/`](./research/) 目录包含 [GAM 论文](https://arxiv.org/abs/2511.18423)的原始代码，包括基准评测脚本（LoCoMo、HotpotQA、RULER、NarrativeQA）以及双智能体（Memorizer + Researcher）实现：

```bash
cd research
pip install -e .
```

```python
from gam_research import MemoryAgent, ResearchAgent
```

详见 [Research README](./research/README.md)。

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。
