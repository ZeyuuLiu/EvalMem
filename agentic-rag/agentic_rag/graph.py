"""LangGraph ReAct agent with retrieval tool."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agentic_rag.config import Settings
from agentic_rag.embeddings import build_embeddings
from agentic_rag.tools import build_retriever_tool

SYSTEM_PROMPT = """You are a helpful assistant. The user has uploaded documents (e.g. novels, notes) into a knowledge base.

Rules:
1. For ANY question about 人物是谁、角色、剧情、关系、设定、章节内容、专有名词 — you MUST call `search_knowledge_base` first with a short query that includes key names or terms (fix typos/split words in the user message when forming the query).
2. Base your answer primarily on the retrieved passages. Quote or paraphrase them. If retrieval is empty or irrelevant, say 知识库中未找到相关片段 and then you may briefly use general knowledge, clearly marking it as 非知识库内容.
3. Do NOT invent plot or character facts when retrieval failed; say you cannot find it in the uploaded texts.

Reply in the same language as the user (Chinese if they write Chinese)."""


def build_llm(settings: Settings) -> ChatOpenAI:
    # Qwen3 默认 enable_thinking=True 时，工具调用常以内嵌 XML 出现在 content 里，
    # 无法被解析为 tool_calls，LangGraph 不会执行检索。关闭思考模式以走标准 function calling。
    return ChatOpenAI(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        model=settings.vllm_model_name,
        temperature=0.2,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
        model_kwargs={"parallel_tool_calls": False},
    )


def build_agent(settings: Settings):
    """Returns a compiled LangGraph runnable (invoke / stream)."""
    llm = build_llm(settings)
    embeddings = build_embeddings(settings)
    tool = build_retriever_tool(settings, embeddings)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("messages"),
        ]
    )
    return create_react_agent(llm, [tool], prompt=prompt)
