"""Loader 抽象基类模块。

定义所有文档加载器必须遵循的接口规范。
加载器负责将原始文件解析为标准 Document 对象。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.core.types import Document


class BaseLoader(ABC):
    """所有文档加载器的抽象基类。

    子类必须实现 `load` 方法，将原始文件解析为 Document 对象。
    加载器只负责"格式统一 + 结构抽取 + 引用收集"，不负责切分。
    """

    @abstractmethod
    def load(self, path: str, collection: str = "default") -> Document:
        """加载文件并解析为标准 Document 对象。

        参数:
            path: 文件路径
            collection: 目标集合名称（用于图片存储路径等）

        返回:
            Document 对象，metadata 至少包含 source_path

        异常:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不支持或解析失败
        """
        ...
