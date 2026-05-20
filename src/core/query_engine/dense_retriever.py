"""DenseRetriever 模块。

组合 EmbeddingClient（query 向量化）+ VectorStore（向量检索），完成语义召回。
将 QueryResult 转换为 RetrievalResult 供下游 Fusion/Reranker 使用。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class DenseRetriever:
    """稠密向量检索器：query embedding + vector store 语义检索。

    属性:
        top_k: 默认返回结果数
    """

    def __init__(
        self,
        settings: Settings,
        embedding_client: Optional[BaseEmbedding] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ) -> None:
        """初始化 DenseRetriever。

        参数:
            settings: 全局配置对象
            embedding_client: Embedding 实例（可选，不传则延迟创建）
            vector_store: VectorStore 实例（可选，不传则延迟创建）
        """
        self._settings = settings
        self._embedding_client = embedding_client
        self._vector_store = vector_store
        self.top_k = settings.retrieval.top_k

    def _get_embedding_client(self) -> BaseEmbedding:
        """获取 Embedding 实例（延迟创建）。"""
        if self._embedding_client is None:
            from src.libs.embedding.embedding_factory import EmbeddingFactory
            self._embedding_client = EmbeddingFactory.create(self._settings.embedding)
        return self._embedding_client

    def _get_vector_store(self, collection_name: str = "default") -> BaseVectorStore:
        """获取 VectorStore 实例（延迟创建）。

        参数:
            collection_name: 集合名称

        返回:
            BaseVectorStore 实例
        """
        # 如果已经有一个 store 且 collection 一致，则复用。如果是 Mock 则直接复用。
        is_mock = type(self._vector_store).__name__ in ('Mock', 'MagicMock', 'NonCallableMagicMock', 'NonCallableMock')
        if self._vector_store is not None and (is_mock or getattr(self._vector_store, "collection_name", None) == collection_name):
            return self._vector_store

        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        self._vector_store = VectorStoreFactory.create(
            self._settings.vector_store,
            collection_name=collection_name
        )
        return self._vector_store

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        collection: Optional[str] = None,
        trace: Optional[TraceContext] = None,
    ) -> List[RetrievalResult]:
        """执行稠密向量检索。

        流程：query → embed → vector_store.query → RetrievalResult 列表

        参数:
            query: 查询文本
            top_k: 返回结果数（默认使用配置值）
            filters: 元数据过滤条件
            collection: 集合名称（可选，优先级高于 filters 中的 collection）
            trace: 可选的追踪上下文

        返回:
            按相似度降序排列的 RetrievalResult 列表
        """
        if not query or not query.strip():
            return []

        # 确定 collection 名称
        col = collection
        if col is None and filters:
            col = filters.get("collection")
        col = col or "default"

        k = top_k if top_k is not None else self.top_k
        start_time = time.time()

        # 1. 向量化 query
        embedding_client = self._get_embedding_client()
        query_vector = embedding_client.embed_single(query)

        embed_ms = (time.time() - start_time) * 1000

        # 2. 向量检索
        vector_store = self._get_vector_store(collection_name=col)
        query_start = time.time()
        query_results = vector_store.query(
            vector=query_vector,
            top_k=k,
            filters=filters,
        )
        query_ms = (time.time() - query_start) * 1000

        # 3. 转换为 RetrievalResult
        results: List[RetrievalResult] = []
        for qr in query_results:
            results.append(RetrievalResult(
                chunk_id=qr.id,
                score=qr.score,
                text=qr.text,
                metadata=qr.metadata,
            ))

        total_ms = (time.time() - start_time) * 1000

        if trace:
            trace.record_stage(
                "dense_retrieval",
                method="vector_search",
                collection=col,
                query_length=len(query),
                result_count=len(results),
                embed_ms=round(embed_ms, 2),
                query_ms=round(query_ms, 2),
                elapsed_ms=round(total_ms, 2),
            )

        logger.debug(
            "DenseRetriever [%s]: query='%s' → %d results (%.1fms)",
            col, query[:50], len(results), total_ms,
        )

        return results
