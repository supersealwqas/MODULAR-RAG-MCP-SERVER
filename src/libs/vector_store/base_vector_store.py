"""向量存储的抽象基类模块。

定义所有 VectorStore 实现必须遵循的接口规范。
支持向量的插入（upsert）和相似度查询（query）操作。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VectorRecord:
    """向量记录数据类，表示一条待存储的向量数据。

    属性:
        id: 记录唯一标识
        vector: 向量数据（浮点数列表）
        text: 原始文本内容
        metadata: 附加元数据（如来源、页码等）
    """
    id: str
    vector: List[float]
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """查询结果数据类，表示一条相似度匹配结果。

    属性:
        id: 记录唯一标识
        score: 相似度分数（越高越相似）
        text: 原始文本内容
        metadata: 附加元数据
    """
    id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseVectorStore(ABC):
    """所有向量存储的抽象基类。

    子类必须实现 `upsert` 和 `query` 方法。
    提供 `delete` 和 `count` 可选方法的默认实现。
    """

    def __init__(self, collection_name: str = "default", **kwargs):
        """初始化向量存储实例。

        参数:
            collection_name: 集合/表名称
            **kwargs: 其他提供者特定参数
        """
        self.collection_name = collection_name

    @abstractmethod
    def upsert(self, records: List[VectorRecord], **kwargs) -> int:
        """插入或更新向量记录。

        参数:
            records: 待插入/更新的向量记录列表
            **kwargs: 提供者特定参数

        返回:
            成功插入/更新的记录数
        """
        ...

    @abstractmethod
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
            top_k: 返回的最大结果数
            filters: 元数据过滤条件（如 {"source": "doc1.pdf"}）
            **kwargs: 提供者特定参数

        返回:
            按相似度降序排列的查询结果列表
        """
        ...

    def delete(self, ids: List[str], **kwargs) -> int:
        """删除指定 ID 的记录。

        参数:
            ids: 待删除的记录 ID 列表
            **kwargs: 提供者特定参数

        返回:
            成动删除的记录数
        """
        raise NotImplementedError("此存储后端未实现 delete 方法")

    def count(self, **kwargs) -> int:
        """获取集合中的记录总数。

        返回:
            记录总数
        """
        raise NotImplementedError("此存储后端未实现 count 方法")
