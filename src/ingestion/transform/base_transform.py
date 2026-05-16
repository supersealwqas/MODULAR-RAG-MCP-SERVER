"""Transform 抽象基类模块。

定义所有 Transform 实现必须遵循的接口规范。
Transform 负责对 Chunk 进行增强/去噪/元数据注入等处理。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk


class BaseTransform(ABC):
    """所有 Transform 实现的抽象基类。

    子类必须实现 `transform` 方法来完成实际的 Chunk 处理逻辑。
    """

    @abstractmethod
    def transform(self, chunks: List[Chunk], trace: Optional[TraceContext] = None) -> List[Chunk]:
        """对 Chunk 列表进行变换处理。

        参数:
            chunks: 待处理的 Chunk 列表
            trace: 可选的追踪上下文，用于记录处理阶段数据

        返回:
            处理后的 Chunk 列表（数量可以变化，但通常保持一致）
        """
        ...
