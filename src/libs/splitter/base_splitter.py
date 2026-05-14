"""文本切分器的抽象基类模块。

定义所有 Splitter 实现必须遵循的接口规范。
支持不同的切分策略：递归切分、语义切分、固定长度切分等。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseSplitter(ABC):
    """所有文本切分器的抽象基类。

    子类必须实现 `split_text` 方法来完成实际的文本切分逻辑。
    提供 `split_texts` 便捷方法用于批量切分场景。
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, **kwargs):
        """初始化切分器实例。

        参数:
            chunk_size: 每个文本块的最大字符数
            chunk_overlap: 相邻文本块之间的重叠字符数
            **kwargs: 其他提供者特定参数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split_text(self, text: str, **kwargs) -> List[str]:
        """将单个文本切分为多个文本块。

        参数:
            text: 待切分的文本
            **kwargs: 提供者特定参数

        返回:
            切分后的文本块列表
        """
        ...

    def split_texts(self, texts: List[str], **kwargs) -> List[str]:
        """批量切分多个文本。

        参数:
            texts: 待切分的文本列表
            **kwargs: 提供者特定参数

        返回:
            所有文本切分后的文本块列表（已展平）
        """
        all_chunks = []
        for text in texts:
            chunks = self.split_text(text, **kwargs)
            all_chunks.extend(chunks)
        return all_chunks
