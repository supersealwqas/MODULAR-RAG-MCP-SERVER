"""query_knowledge_hub MCP Tool 模块。

调用 HybridSearch + Reranker 执行知识库查询，
返回带引用标注的 Markdown 响应和结构化引用信息。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.reranker import Reranker
from src.core.response.response_builder import ResponseBuilder
from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)

# 全局实例（延迟初始化）
_hybrid_search: Optional[HybridSearch] = None
_reranker: Optional[Reranker] = None


def _get_hybrid_search(settings: Settings) -> HybridSearch:
    """获取 HybridSearch 实例（延迟创建）。"""
    global _hybrid_search
    if _hybrid_search is None:
        _hybrid_search = HybridSearch(settings)
    return _hybrid_search


def _get_reranker(settings: Settings) -> Reranker:
    """获取 Reranker 实例（延迟创建）。"""
    global _reranker
    if _reranker is None:
        _reranker = Reranker(settings)
    return _reranker


async def query_knowledge_hub(
    query: str,
    top_k: int = 10,
    collection: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """查询知识库，返回最相关的文档片段。

    参数:
        query: 查询文本
        top_k: 返回结果数量（默认 10）
        collection: 限定检索集合（可选）
        settings: Settings 实例（可选，不传则自动加载）

    返回:
        MCP Tool 响应字典，包含 Markdown 文本和结构化引用
    """
    logger.info("query_knowledge_hub 被调用: query=%s, top_k=%d, collection=%s", query, top_k, collection)

    # 加载配置
    if settings is None:
        from src.core.settings import load_settings
        settings = load_settings()

    # 构建过滤条件
    filters = {}
    if collection:
        filters["collection"] = collection

    # 执行检索
    trace = TraceContext()
    hybrid_search = _get_hybrid_search(settings)

    try:
        results = hybrid_search.search(
            query=query,
            top_k=top_k,
            filters=filters if filters else None,
            trace=trace,
        )
    except Exception as e:
        logger.error("HybridSearch 检索失败: %s", e, exc_info=True)
        return ResponseBuilder.build_empty(query)

    # 无结果时返回友好提示
    if not results:
        logger.info("查询 '%s' 未找到结果", query)
        return ResponseBuilder.build_empty(query)

    # Reranker 精排
    reranker = _get_reranker(settings)
    try:
        rerank_result = reranker.rerank(
            query=query,
            candidates=results,
            top_k=top_k,
            trace=trace,
        )
        results = rerank_result["results"]
    except Exception as e:
        logger.warning("Reranker 失败，使用原始排序: %s", e)

    # 构建响应
    response = ResponseBuilder.build(results, query, include_images=True)
    logger.info("查询 '%s' 返回 %d 条结果", query, len(results))

    return response
