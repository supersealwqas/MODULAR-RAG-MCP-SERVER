"""Fusion 模块。

实现 RRF (Reciprocal Rank Fusion) 算法，将 Dense/Sparse 检索结果融合为统一排名。
支持加权 RRF：score(d) = Σ w_i × 1 / (k + rank_i(d))
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
        weights: 各排名列表的权重（默认 None 表示等权）
    """

    def __init__(self, k: int = _DEFAULT_RRF_K, weights: Optional[List[float]] = None) -> None:
        """初始化 Fusion。

        参数:
            k: RRF 常数，越大则排名靠后结果的权重衰减越慢（默认 60）
            weights: 各排名列表的权重，长度应与 rankings 一致（默认 None 表示等权）
        """
        self.k = k
        self.weights = weights

    def fuse(
        self,
        rankings: List[List[RetrievalResult]],
        top_k: Optional[int] = None,
        weights: Optional[List[float]] = None,
        trace: Optional[TraceContext] = None,
    ) -> List[RetrievalResult]:
        """使用 RRF 算法融合多个排名列表。

        参数:
            rankings: 多个排名列表，每个列表按相关性降序排列
            top_k: 返回结果数（默认返回所有结果）
            weights: 动态权重列表（覆盖初始化时的 weights），长度应与 rankings 一致
            trace: 可选的追踪上下文

        返回:
            按 RRF 分数降序排列的 RetrievalResult 列表
        """
        if not rankings:
            return []

        # 确定使用的权重（优先使用调用时传入的权重）
        active_weights = weights if weights is not None else self.weights

        # 过滤空列表（同时维护权重映射）
        valid_rankings = []
        valid_weights = []
        for i, ranking in enumerate(rankings):
            if ranking:
                valid_rankings.append(ranking)
                # 获取对应权重，默认 1.0
                weight = 1.0
                if active_weights and i < len(active_weights):
                    weight = active_weights[i]
                valid_weights.append(weight)

        if not valid_rankings:
            return []

        # 计算每个 chunk_id 的加权 RRF 分数
        # 同时保留第一次出现的 text 和 metadata
        rrf_scores: Dict[str, float] = {}
        chunk_info: Dict[str, RetrievalResult] = {}

        for ranking, weight in zip(valid_rankings, valid_weights):
            for rank, result in enumerate(ranking):
                cid = result.chunk_id
                # 加权 RRF 公式：weight × 1 / (k + rank)
                score_contribution = weight * 1.0 / (self.k + rank)
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
                weights=valid_weights,
                input_rankings=len(valid_rankings),
                output_count=len(results),
            )

        logger.debug(
            "Fusion: %d rankings → %d results (k=%d, weights=%s)",
            len(valid_rankings), len(results), self.k, valid_weights,
        )

        return results
