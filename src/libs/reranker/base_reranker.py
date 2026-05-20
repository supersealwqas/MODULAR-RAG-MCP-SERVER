"""重排序器的抽象基类模块。

定义所有 Reranker 实现必须遵循的接口规范。
重排序器用于对检索结果进行二次排序，提升相关性。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Candidate:
    """候选文档数据类，表示待重排序的文档。

    属性:
        id: 文档唯一标识
        text: 文档文本内容
        score: 初始相关性分数（来自检索阶段）
        metadata: 附加元数据（如来源、页码等）
    """
    id: str
    text: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RankedCandidate:
    """重排序后的候选文档数据类。

    属性:
        id: 文档唯一标识
        text: 文档文本内容
        rerank_score: 重排序后的相关性分数
        original_score: 原始相关性分数（来自检索阶段）
        metadata: 附加元数据
    """
    id: str
    text: str
    rerank_score: float
    original_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseReranker(ABC):
    """所有重排序器的抽象基类。

    子类必须实现 `rerank` 方法来完成实际的重排序逻辑。
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Candidate],
        top_k: Optional[int] = None,
        **kwargs,
    ) -> List[RankedCandidate]:
        """对候选文档进行重排序。

        参数:
            query: 查询文本
            candidates: 待重排序的候选文档列表
            top_k: 返回的最大结果数（None 表示返回全部）
            **kwargs: 提供者特定参数

        返回:
            按重排序分数降序排列的 RankedCandidate 列表
        """
        ...


class NoneReranker(BaseReranker):
    """空重排序器，保持原始顺序不变。

    用于禁用重排序的场景，作为默认回退实现。
    """

    def rerank(
        self,
        query: str,
        candidates: List[Candidate],
        top_k: Optional[int] = None,
        **kwargs,
    ) -> List[RankedCandidate]:
        """保持原始顺序，将 Candidate 转换为 RankedCandidate。

        参数:
            query: 查询文本（未使用）
            candidates: 候选文档列表
            top_k: 返回的最大结果数

        返回:
            保持原始顺序的 RankedCandidate 列表
        """
        results = []
        for candidate in candidates:
            results.append(RankedCandidate(
                id=candidate.id,
                text=candidate.text,
                rerank_score=candidate.score,  # 保持原始分数
                original_score=candidate.score,
                metadata=candidate.metadata,
            ))

        if top_k is not None:
            if top_k <= 0:
                return []
            results = results[:top_k]

        return results
