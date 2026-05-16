"""SparseRetriever 模块。

组合 BM25Indexer（关键词检索）+ VectorStore（获取完整文本），
完成稀疏检索召回。
将 BM25 结果转换为 RetrievalResult 供下游 Fusion/Reranker 使用。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class SparseRetriever:
    """稀疏检索器：BM25 关键词检索 + VectorStore 获取完整文本。

    属性:
        top_k: 默认返回结果数
    """

    def __init__(
        self,
        settings: Settings,
        bm25_indexer: Optional[BM25Indexer] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ) -> None:
        """初始化 SparseRetriever。

        参数:
            settings: 全局配置对象
            bm25_indexer: BM25Indexer 实例（可选，不传则延迟创建）
            vector_store: VectorStore 实例（可选，不传则延迟创建）
        """
        self._settings = settings
        self._bm25_indexer = bm25_indexer
        self._vector_store = vector_store
        self.top_k = settings.retrieval.top_k

    def _get_bm25_indexer(self) -> BM25Indexer:
        """获取 BM25Indexer 实例（延迟创建 + 自动加载索引）。"""
        if self._bm25_indexer is None:
            self._bm25_indexer = BM25Indexer()
            self._bm25_indexer.load()
        return self._bm25_indexer

    def _get_vector_store(self) -> BaseVectorStore:
        """获取 VectorStore 实例（延迟创建）。"""
        if self._vector_store is None:
            from src.libs.vector_store.vector_store_factory import VectorStoreFactory
            self._vector_store = VectorStoreFactory.create(self._settings.vector_store)
        return self._vector_store

    def retrieve(
        self,
        keywords: List[str],
        top_k: Optional[int] = None,
        trace: Optional[TraceContext] = None,
    ) -> List[RetrievalResult]:
        """执行稀疏检索。

        流程：keywords → bm25_indexer.query → vector_store.get_by_ids → RetrievalResult 列表

        参数:
            keywords: 查询关键词列表（来自 QueryProcessor.process().keywords）
            top_k: 返回结果数（默认使用配置值）
            trace: 可选的追踪上下文

        返回:
            按 BM25 分数降序排列的 RetrievalResult 列表
        """
        if not keywords:
            return []

        k = top_k if top_k is not None else self.top_k
        start_time = time.time()

        # 1. BM25 检索 chunk_ids + scores
        bm25_indexer = self._get_bm25_indexer()
        bm25_results = bm25_indexer.query(keywords, top_k=k)

        bm25_ms = (time.time() - start_time) * 1000

        if not bm25_results:
            if trace:
                trace.record_stage(
                    "sparse_retrieval",
                    method="bm25",
                    keyword_count=len(keywords),
                    result_count=0,
                    bm25_ms=round(bm25_ms, 2),
                    lookup_ms=0,
                    elapsed_ms=round(bm25_ms, 2),
                )
            return []

        # 2. 从 VectorStore 获取完整文本和 metadata
        chunk_ids = [cid for cid, _ in bm25_results]
        scores = {cid: score for cid, score in bm25_results}

        lookup_start = time.time()
        vector_store = self._get_vector_store()
        records = vector_store.get_by_ids(chunk_ids)
        lookup_ms = (time.time() - lookup_start) * 1000

        # 3. 组装 RetrievalResult（保持 BM25 排序）
        id_to_record = {r["id"]: r for r in records}
        results: List[RetrievalResult] = []
        for cid in chunk_ids:
            record = id_to_record.get(cid)
            if record:
                results.append(RetrievalResult(
                    chunk_id=cid,
                    score=scores[cid],
                    text=record.get("text", ""),
                    metadata=record.get("metadata", {}),
                ))

        total_ms = (time.time() - start_time) * 1000

        if trace:
            trace.record_stage(
                "sparse_retrieval",
                method="bm25",
                keyword_count=len(keywords),
                result_count=len(results),
                bm25_ms=round(bm25_ms, 2),
                lookup_ms=round(lookup_ms, 2),
                elapsed_ms=round(total_ms, 2),
            )

        logger.debug(
            "SparseRetriever: keywords=%s → %d results (%.1fms)",
            keywords[:3], len(results), total_ms,
        )

        return results
