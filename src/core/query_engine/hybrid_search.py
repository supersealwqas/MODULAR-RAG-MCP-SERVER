"""HybridSearch 模块。

编排 Dense + Sparse + Fusion 的完整混合检索流程，
集成 Metadata 过滤逻辑。
内部流程：query_processor.process() → dense.retrieve + sparse.retrieve → fusion.fuse() → metadata_filter → Top-K
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.fusion import Fusion
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import ProcessedQuery, RetrievalResult

logger = logging.getLogger(__name__)


class HybridSearch:
    """混合检索编排器：Dense + Sparse + Fusion。

    属性:
        top_k: 默认返回结果数
    """

    def __init__(
        self,
        settings: Settings,
        query_processor: Optional[QueryProcessor] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        fusion: Optional[Fusion] = None,
    ) -> None:
        """初始化 HybridSearch。

        参数:
            settings: 全局配置对象
            query_processor: QueryProcessor 实例（可选，不传则延迟创建）
            dense_retriever: DenseRetriever 实例（可选，不传则延迟创建）
            sparse_retriever: SparseRetriever 实例（可选，不传则延迟创建）
            fusion: Fusion 实例（可选，不传则使用默认配置）
        """
        self._settings = settings
        self._query_processor = query_processor
        self._dense_retriever = dense_retriever
        self._sparse_retriever = sparse_retriever
        self._fusion = fusion or Fusion()
        self.top_k = settings.retrieval.top_k

    def _get_query_processor(self) -> QueryProcessor:
        """获取 QueryProcessor 实例（延迟创建）。"""
        if self._query_processor is None:
            self._query_processor = QueryProcessor(self._settings)
        return self._query_processor

    def _get_dense_retriever(self) -> DenseRetriever:
        """获取 DenseRetriever 实例（延迟创建）。"""
        if self._dense_retriever is None:
            self._dense_retriever = DenseRetriever(self._settings)
        return self._dense_retriever

    def _get_sparse_retriever(self) -> SparseRetriever:
        """获取 SparseRetriever 实例（延迟创建）。"""
        if self._sparse_retriever is None:
            self._sparse_retriever = SparseRetriever(self._settings)
        return self._sparse_retriever

    def _apply_metadata_filters(
        self,
        candidates: List[RetrievalResult],
        filters: Dict[str, Any],
    ) -> List[RetrievalResult]:
        """后置 Metadata 过滤。

        参数:
            candidates: 候选结果列表
            filters: 过滤条件字典

        返回:
            满足过滤条件的结果列表
        """
        if not filters:
            return candidates

        filtered = []
        for result in candidates:
            match = True
            for key, value in filters.items():
                meta_value = result.metadata.get(key)
                if meta_value != value:
                    match = False
                    break
            if match:
                filtered.append(result)

        return filtered

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[TraceContext] = None,
    ) -> List[RetrievalResult]:
        """执行混合检索。

        流程：
        1. QueryProcessor 提取关键词
        2. DenseRetriever 向量检索
        3. SparseRetriever BM25 检索
        4. Fusion RRF 融合
        5. Metadata 后置过滤
        6. Top-K 截断

        参数:
            query: 用户查询文本
            top_k: 返回结果数（默认使用配置值）
            filters: 元数据过滤条件
            trace: 可选的追踪上下文

        返回:
            按相关性降序排列的 RetrievalResult 列表
        """
        if not query or not query.strip():
            return []

        k = top_k if top_k is not None else self.top_k
        start_time = time.time()

        # 1. 查询预处理
        query_processor = self._get_query_processor()
        processed = query_processor.process(query, filters=filters, trace=trace)

        # 2. Dense 检索（容错：失败时降级为空列表）
        dense_retriever = self._get_dense_retriever()
        try:
            dense_results = dense_retriever.retrieve(
                query, top_k=k, trace=trace,
            )
        except Exception as e:
            logger.warning("DenseRetriever 失败，降级到空结果: %s", e)
            dense_results = []

        # 3. Sparse 检索（容错：失败时降级为空列表）
        sparse_retriever = self._get_sparse_retriever()
        try:
            sparse_results = sparse_retriever.retrieve(
                processed.keywords, top_k=k, trace=trace,
            )
        except Exception as e:
            logger.warning("SparseRetriever 失败，降级到空结果: %s", e)
            sparse_results = []

        # 4. RRF 融合
        rankings = []
        if dense_results:
            rankings.append(dense_results)
        if sparse_results:
            rankings.append(sparse_results)

        if not rankings:
            logger.warning("Dense 和 Sparse 均无结果")
            return []

        fused_results = self._fusion.fuse(rankings, top_k=k, trace=trace)

        # 5. Metadata 后置过滤
        if filters:
            fused_results = self._apply_metadata_filters(fused_results, filters)

        # 6. Top-K 截断
        results = fused_results[:k]

        total_ms = (time.time() - start_time) * 1000

        if trace:
            trace.record_stage(
                "hybrid_search",
                query_length=len(query),
                dense_count=len(dense_results),
                sparse_count=len(sparse_results),
                fused_count=len(fused_results),
                final_count=len(results),
                elapsed_ms=round(total_ms, 2),
            )

        logger.debug(
            "HybridSearch: query='%s' → dense=%d, sparse=%d, final=%d (%.1fms)",
            query[:50], len(dense_results), len(sparse_results), len(results), total_ms,
        )

        return results
