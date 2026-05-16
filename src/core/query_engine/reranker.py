"""Reranker Core 层编排模块。

接入 libs.reranker 后端，对 HybridSearch 的候选结果进行精排。
后端异常或超时时回退到 fusion 排名，标记 fallback=True。

内部流程：
  List[RetrievalResult] → Candidate 转换 → backend.rerank() → RankedCandidate → RetrievalResult
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult
from src.libs.reranker.base_reranker import BaseReranker, Candidate, RankedCandidate
from src.libs.reranker.reranker_factory import RerankerFactory

logger = logging.getLogger(__name__)


class Reranker:
    """Reranker Core 层编排器。

    属性:
        top_k: 重排序后保留的默认结果数
    """

    def __init__(
        self,
        settings: Settings,
        backend: Optional[BaseReranker] = None,
    ) -> None:
        """初始化 Reranker。

        参数:
            settings: 全局配置对象
            backend: BaseReranker 实例（可选，不传则通过工厂创建）
        """
        self._settings = settings
        self._rerank_config = settings.rerank
        self._backend = backend
        self.top_k = self._rerank_config.top_k

    def _get_backend(self) -> BaseReranker:
        """获取 Reranker 后端实例（延迟创建）。"""
        if self._backend is None:
            self._backend = RerankerFactory.create(
                provider=self._rerank_config.provider,
            )
        return self._backend

    @staticmethod
    def _to_candidates(results: List[RetrievalResult]) -> List[Candidate]:
        """将 RetrievalResult 列表转换为 Candidate 列表。

        参数:
            results: 检索结果列表

        返回:
            Candidate 列表
        """
        return [
            Candidate(
                id=r.chunk_id,
                text=r.text,
                score=r.score,
                metadata=r.metadata,
            )
            for r in results
        ]

    @staticmethod
    def _to_retrieval_results(
        ranked: List[RankedCandidate],
        original_map: Dict[str, RetrievalResult],
    ) -> List[RetrievalResult]:
        """将 RankedCandidate 列表转换回 RetrievalResult 列表。

        保留原始 metadata，使用 rerank_score 作为 score。

        参数:
            ranked: 重排序后的候选列表
            original_map: 原始结果的 chunk_id → RetrievalResult 映射

        返回:
            RetrievalResult 列表
        """
        results = []
        for rc in ranked:
            original = original_map.get(rc.id)
            metadata = rc.metadata if rc.metadata else (original.metadata if original else {})
            results.append(RetrievalResult(
                chunk_id=rc.id,
                score=rc.rerank_score,
                text=rc.text,
                metadata=metadata,
            ))
        return results

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: Optional[int] = None,
        trace: Optional[TraceContext] = None,
    ) -> Dict[str, Any]:
        """对候选结果进行重排序。

        后端异常时回退到原始排序，标记 fallback=True。

        参数:
            query: 查询文本
            candidates: HybridSearch 返回的候选结果
            top_k: 返回的最大结果数（默认使用配置值）
            trace: 可选的追踪上下文

        返回:
            包含以下键的字典：
              - results: List[RetrievalResult]，重排序后的结果
              - fallback: bool，是否回退到原始排序
              - elapsed_ms: float，耗时（毫秒）
        """
        k = top_k if top_k is not None else self.top_k
        start_time = time.time()

        if not candidates:
            return {"results": [], "fallback": False, "elapsed_ms": 0.0}

        # 构建原始结果映射（用于回退时保留完整 metadata）
        original_map = {r.chunk_id: r for r in candidates}

        try:
            backend = self._get_backend()
            candidate_list = self._to_candidates(candidates)
            ranked = backend.rerank(query, candidate_list, top_k=k)
            results = self._to_retrieval_results(ranked, original_map)
            fallback = False
        except Exception as e:
            logger.warning("Reranker 后端异常，回退到原始排序: %s", e)
            results = candidates[:k]
            fallback = True

        elapsed_ms = (time.time() - start_time) * 1000

        if trace:
            trace.record_stage(
                "reranker",
                fallback=fallback,
                input_count=len(candidates),
                output_count=len(results),
                elapsed_ms=round(elapsed_ms, 2),
            )

        logger.debug(
            "Reranker: query='%s' → input=%d, output=%d, fallback=%s (%.1fms)",
            query[:50], len(candidates), len(results), fallback, elapsed_ms,
        )

        return {"results": results, "fallback": fallback, "elapsed_ms": round(elapsed_ms, 2)}
