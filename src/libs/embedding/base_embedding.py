"""Embedding 提供者的抽象基类模块。

定义所有 Embedding 实现必须遵循的接口规范。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseEmbedding(ABC):
    """所有 Embedding 实现的抽象基类。

    子类必须实现 `embed` 方法来完成实际的向量化调用。
    提供 `embed_single` 便捷方法用于单文本嵌入场景。
    """

    def __init__(self, model: str, dimensions: int = 1536, api_key: str = "", **kwargs):
        """初始化 Embedding 实例。

        参数:
            model: 嵌入模型名称（如 "text-embedding-ada-002"）
            dimensions: 输出向量维度
            api_key: API 密钥（可选，部分本地模型不需要）
            **kwargs: 其他提供者特定参数
        """
        self.model = model
        self.dimensions = dimensions
        self.api_key = api_key

    @abstractmethod
    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """批量将文本转换为向量。

        参数:
            texts: 待嵌入的文本列表
            **kwargs: 提供者特定参数

        返回:
            向量列表，每个向量是浮点数列表，长度等于 dimensions
        """
        ...

    def embed_single(self, text: str, **kwargs) -> List[float]:
        """单文本嵌入的便捷方法。

        参数:
            text: 待嵌入的文本
            **kwargs: 提供者特定参数

        返回:
            浮点数列表表示的向量
        """
        results = self.embed([text], **kwargs)
        return results[0]
