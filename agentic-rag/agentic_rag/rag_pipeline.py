"""先检索再生成：不依赖 LLM 的 function calling（避免 Qwen3 只输出 <tool_call> XML 导致无法执行工具）。"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from agentic_rag.config import Settings
from agentic_rag.embeddings import build_embeddings
from agentic_rag.graph import build_llm
from agentic_rag.qdrant_store import get_vector_store

_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是阅读助手。请**仅根据**下面【上下文】回答用户问题。\n"
            "若上下文中没有相关信息，请明确说明「上传的文档中未找到相关内容」，不要编造人物或情节。\n\n"
            "【上下文】\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def run_rag_turn(settings: Settings, question: str) -> str:
    llm = build_llm(settings)
    embeddings = build_embeddings(settings)
    store = get_vector_store(settings, embeddings)
    retriever = store.as_retriever(search_kwargs={"k": settings.retrieve_k})
    docs = retriever.invoke(question)
    if not docs:
        return "知识库中未检索到相关片段。"
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    chain = _RAG_PROMPT | llm
    out = chain.invoke({"context": context, "question": question})
    return (out.content or "").strip() if hasattr(out, "content") else str(out)
