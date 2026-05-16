"""Fusion 模块。

实现 RRF (Reciprocal Rank Fusion) 算法，将 Dense/Sparse 检索结果融合为统一排名。
RRF 公式：score(d) = Σ 1 / (k + rank_i(d))
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult

logger = logging.getLogger(__name__)

# RRF 默认常数 k（控制排名靠后结果的权重衰减速度）
_DEFAULT_RRF_K = 60


class Fusion:
    """RRF 融合器：将多个排名列表融合为统一排序。

    属性:
        k: RRF 常数（默认 60）
    """

    def __init__(self, k: int = _DEFAULT_RRF_K) -> None:
        """初始化 Fusion。

        参数:
            k: RRF 常数，越大则排名靠后结果的权重衰减越慢（默认 60）
        """
        self.k = k

    def fuse(
        self,
        rankings: List[List[RetrievalResult]],
        top_k: Optional[int] = None,
        trace: Optional[TraceContext] = None,
    ) -> List[RetrievalResult]:
        """使用 RRF 算法融合多个排名列表。

        参数:
            rankings: 多个排名列表，每个列表按相关性降序排列
            top_k: 返回结果数（默认返回所有结果）
            trace: 可选的追踪上下文

        返回:
            按 RRF 分数降序排列的 RetrievalResult 列表
        """
        if not rankings:
            return []

        # 过滤空列表
        valid_rankings = [r for r in rankings if r]
        if not valid_rankings:
            return []

        # 计算每个 chunk_id 的 RRF 分数
        # 同时保留第一次出现的 text 和 metadata
        rrf_scores: Dict[str, float] = {}
        chunk_info: Dict[str, RetrievalResult] = {}

        for ranking in valid_rankings:
            for rank, result in enumerate(ranking):
                cid = result.chunk_id
                # RRF 公式：1 / (k + rank)
                score_contribution = 1.0 / (self.k + rank)
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score_contribution

                # 保留第一次出现的完整信息
                if cid not in chunk_info:
                    chunk_info[cid] = result

        # 按 RRF 分数降序排序
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        # 构建结果列表
        results: List[RetrievalResult] = []
        for cid in sorted_ids:
            original = chunk_info[cid]
            results.append(RetrievalResult(
                chunk_id=cid,
                score=rrf_scores[cid],
                text=original.text,
                metadata=original.metadata,
            ))

        # 截断到 top_k
        if top_k is not None:
            results = results[:top_k]

        if trace:
            trace.record_stage(
                "fusion",
                method="rrf",
                k=self.k,
                input_rankings=len(valid_rankings),
                output_count=len(results),
            )

        logger.debug(
            "Fusion: %d rankings → %d results (k=%d)",
            len(valid_rankings), len(results), self.k,
        )

        return results
