"""ChromaDB 向量存储实现模块。

基于 ChromaDB 实现的向量存储后端，支持本地持久化目录。
提供 upsert、query、delete、count 等操作。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    QueryResult,
    VectorRecord,
)
from src.libs.vector_store.vector_store_factory import register_vector_store


@register_vector_store("chroma")
class ChromaStore(BaseVectorStore):
    """ChromaDB 向量存储实现。

    使用 ChromaDB 作为后端，支持本地持久化存储。
    每个 collection 对应 ChromaDB 中的一个 collection。

    特性:
        - 本地持久化：数据存储在指定目录，重启后可恢复
        - 自动创建 collection：首次写入时自动创建
        - 元数据过滤：支持按 metadata 字段过滤查询结果
    """

    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: str = "data/db/chroma",
        **kwargs,
    ):
        """初始化 ChromaStore 实例。

        参数:
            collection_name: ChromaDB collection 名称（默认 "default"）
            persist_directory: 持久化目录路径（默认 "data/db/chroma"）
            **kwargs: 其他参数
        """
        super().__init__(collection_name, **kwargs)
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None

    def _get_collection(self):
        """延迟获取 ChromaDB collection 实例。

        首次调用时初始化 client 和 collection，后续复用。

        返回:
            ChromaDB collection 对象
        """
        if self._collection is not None:
            return self._collection

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise ImportError(
                "请安装 chromadb 库: uv pip install chromadb"
            )

        self._client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    @staticmethod
    def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """将嵌套结构序列化为 ChromaDB 兼容格式。

        ChromaDB 只支持 str/int/float/bool 值，嵌套的 dict/list 需要 JSON 序列化。

        参数:
            metadata: 原始 metadata 字典

        返回:
            ChromaDB 兼容的 metadata 字典
        """
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif value is None:
                sanitized[key] = ""
            elif isinstance(value, (dict, list)):
                # 嵌套结构序列化为 JSON 字符串
                sanitized[key] = json.dumps(value, ensure_ascii=False)
            else:
                # 其他类型转为字符串
                sanitized[key] = str(value)
        return sanitized

    @staticmethod
    def _deserialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """将 ChromaDB 返回的 metadata 反序列化。

        尝试将 JSON 字符串还原为 dict/list。

        参数:
            metadata: ChromaDB 返回的 metadata

        返回:
            反序列化后的 metadata
        """
        if not metadata:
            return metadata

        deserialized = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                # 尝试 JSON 反序列化
                try:
                    deserialized[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    deserialized[key] = value
            else:
                deserialized[key] = value
        return deserialized

    def upsert(self, records: List[VectorRecord], **kwargs) -> int:
        """插入或更新向量记录到 ChromaDB。

        参数:
            records: 待插入/更新的向量记录列表
            **kwargs: 其他参数

        返回:
            成功插入/更新的记录数

        异常:
            ValueError: 记录列表为空时抛出
        """
        if not records:
            return 0

        collection = self._get_collection()

        ids = [r.id for r in records]
        embeddings = [r.vector for r in records]
        documents = [r.text for r in records]
        # ChromaDB 要求 metadata 非空，空 dict 时添加占位字段
        # 嵌套结构需要序列化为 JSON 字符串
        metadatas = [
            self._sanitize_metadata(r.metadata) if r.metadata else {"_has_metadata": False}
            for r in records
        ]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        return len(records)

    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[QueryResult]:
        """根据向量进行相似度查询。

        参数:
            vector: 查询向量
            top_k: 返回的最大结果数（默认 10）
            filters: 元数据过滤条件（如 {"source": "doc1.pdf"}）
            **kwargs: 其他参数

        返回:
            按相似度降序排列的查询结果列表

        异常:
            RuntimeError: 查询失败时抛出
        """
        collection = self._get_collection()

        # 构建 where 过滤条件
        where = None
        if filters:
            if len(filters) == 1:
                key, value = next(iter(filters.items()))
                where = {key: value}
            else:
                where = {
                    "$and": [{k: v} for k, v in filters.items()]
                }

        try:
            results = collection.query(
                query_embeddings=[vector],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            raise RuntimeError(
                f"ChromaDB 查询失败 (collection={self.collection_name}): {e}"
            ) from e

        # 解析结果
        query_results = []
        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)
            documents = results["documents"][0] if results["documents"] else [""] * len(ids)
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)

            for i, doc_id in enumerate(ids):
                # ChromaDB 返回的是距离（越小越相似），转换为分数（越大越相似）
                score = 1.0 - distances[i]
                meta = metadatas[i] or {}
                # 移除占位字段并反序列化 JSON
                meta.pop("_has_metadata", None)
                meta = self._deserialize_metadata(meta)
                query_results.append(QueryResult(
                    id=doc_id,
                    score=score,
                    text=documents[i],
                    metadata=meta,
                ))

        return query_results

    def delete(self, ids: List[str], **kwargs) -> int:
        """删除指定 ID 的记录。

        参数:
            ids: 待删除的记录 ID 列表
            **kwargs: 其他参数

        返回:
            成功删除的记录数
        """
        if not ids:
            return 0

        collection = self._get_collection()
        before = collection.count()
        collection.delete(ids=ids)
        after = collection.count()
        return before - after

    def count(self, **kwargs) -> int:
        """获取集合中的记录总数。

        返回:
            记录总数
        """
        collection = self._get_collection()
        return collection.count()

    def get_by_ids(self, ids: List[str], **kwargs) -> List[Dict[str, Any]]:
        """根据 ID 批量获取记录。

        参数:
            ids: 记录 ID 列表
            **kwargs: 其他参数

        返回:
            记录字典列表，每个字典包含 id、text、metadata 字段
        """
        if not ids:
            return []

        collection = self._get_collection()
        results = collection.get(
            ids=ids,
            include=["documents", "metadatas"],
        )

        records = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                meta = results["metadatas"][i] if results["metadatas"] else {}
                meta.pop("_has_metadata", None)
                meta = self._deserialize_metadata(meta)
                records.append({
                    "id": doc_id,
                    "text": results["documents"][i] if results["documents"] else "",
                    "metadata": meta,
                })

        return records
