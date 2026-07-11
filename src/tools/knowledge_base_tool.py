"""
RAG 知识库检索工具

提供向量化的知识库检索能力，Agent 可通过此工具查询私有知识库，
实现基于私有数据的精准问答。支持多知识库联合检索、相似度过滤。
"""
import os
import json
import logging
from typing import Optional

from langchain.tools import tool
from coze_coding_utils.log.write_log import request_context
from coze_coding_utils.runtime_ctx.context import new_context

logger = logging.getLogger(__name__)


def _search_knowledge_base(
    query: str,
    knowledge_base_ids: list[str],
    top_k: int = 5,
    similarity_threshold: float = 0.7,
) -> list[dict]:
    """
    知识库检索核心逻辑（可被多个 tool 复用）

    检索流程：
    1. 根据 knowledge_base_ids 加载对应的 FAISS 索引
    2. 将 query 向量化
    3. 执行相似度搜索
    4. 过滤低于阈值的结果
    5. 返回排序后的文档片段
    """
    try:
        from agents.services.rag_engine import rag_engine

        results = rag_engine.search(
            query=query,
            knowledge_base_ids=knowledge_base_ids,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
        return results
    except Exception as e:
        logger.error(f"Knowledge base search failed: {e}")
        return []


@tool
def knowledge_base_search(
    query: str,
    knowledge_base_ids: str = "",
    top_k: int = 5,
    similarity_threshold: float = 0.7,
) -> str:
    """
    Search the private knowledge base for relevant documents.
    Use this tool when the user asks questions that may require internal/private domain knowledge,
    such as product details, company policies, industry regulations, FAQ answers, etc.

    Args:
        query: The search query text describing what information to find.
        knowledge_base_ids: Comma-separated list of knowledge base IDs to search in.
                           If empty, searches all available knowledge bases.
        top_k: Maximum number of results to return (default: 5).
        similarity_threshold: Minimum similarity score 0.0-1.0 (default: 0.7).

    Returns:
        Retrieved document snippets from the knowledge base, ranked by relevance.
    """
    ctx = request_context.get() or new_context(method="knowledge_base_search")

    # Parse knowledge_base_ids from comma-separated string
    kb_ids = [kb.strip() for kb in knowledge_base_ids.split(",") if kb.strip()]

    results = _search_knowledge_base(
        query=query,
        knowledge_base_ids=kb_ids,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )

    if not results:
        return "未在知识库中找到相关信息，建议使用通用模型能力回答。"

    # Format results for the agent
    output_parts = []
    for i, doc in enumerate(results, 1):
        source = doc.get("source", "unknown")
        content = doc.get("content", "")
        score = doc.get("score", 0.0)
        output_parts.append(
            f"[{i}] (来源: {source}, 相似度: {score:.2f})\n{content}"
        )

    return "\n\n---\n\n".join(output_parts)


@tool
def list_knowledge_bases() -> str:
    """
    List all available knowledge bases with their IDs, names, and document counts.
    Use this tool to discover which knowledge bases are available before performing a search.

    Returns:
        A formatted list of available knowledge bases.
    """
    ctx = request_context.get() or new_context(method="list_knowledge_bases")

    try:
        from agents.services.rag_engine import rag_engine

        kb_list = rag_engine.list_available_bases()
        if not kb_list:
            return "当前没有可用的知识库。"

        output_parts = []
        for kb in kb_list:
            kb_id = kb.get("id", "")
            kb_name = kb.get("name", "")
            doc_count = kb.get("doc_count", 0)
            output_parts.append(f"- {kb_name} (ID: {kb_id}, 文档数: {doc_count})")

        return "可用知识库列表:\n" + "\n".join(output_parts)
    except Exception as e:
        logger.error(f"List knowledge bases failed: {e}")
        return f"获取知识库列表失败: {e}"
