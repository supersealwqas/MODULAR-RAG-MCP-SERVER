"""Cross-Encoder 重排序器模块。

使用 Cross-Encoder 模型对 (query, passage) 对进行相关性打分，
实现高精度的语义重排序。

Cross-Encoder 与 Bi-Encoder（Embedding）的区别：
- Bi-Encoder：query 和 passage 独立编码，计算向量相似度，速度快但精度较低
- Cross-Encoder：query 和 passage 拼接后一起编码，精度高但速度较慢

适用场景：对 Top-M 候选进行精排（M=10~30），CPU 环境下建议控制候选数量。
"""

from __future__ import annotations

import time
from typing import Any, Callable, List, Optional

from src.libs.reranker.base_reranker import (
    BaseReranker,
    Candidate,
    RankedCandidate,
)
from src.libs.reranker.reranker_factory import register_reranker


# 默认 Cross-Encoder 模型（本地 bge-reranker-large）
DEFAULT_MODEL = "models/bge-reranker-large"


@register_reranker("cross_encoder")
class CrossEncoderReranker(BaseReranker):
    """Cross-Encoder 重排序器。

    通过 Cross-Encoder 模型对 (query, passage) 对进行相关性打分，
    实现高精度的语义重排序。

    特性:
        - 延迟加载模型，首次调用 rerank 时才加载到内存
        - 支持自定义 scorer（测试中可注入 mock）
        - 支持超时控制，超时时返回回退信号
    """

    def __init__(
        self,
        model: str = "",
        scorer: Optional[Callable] = None,
        timeout: float = 30.0,
        **kwargs,
    ):
        """初始化 Cross-Encoder 重排序器。

        参数:
            model: Cross-Encoder 模型名称或本地路径（默认 ms-marco-MiniLM-L-6-v2）
            scorer: 自定义打分函数（可选，用于测试注入；签名: scorer(pairs) -> List[float]）
            timeout: 超时时间（秒），超时触发回退
            **kwargs: 其他参数（忽略，保留接口兼容）
        """
        self._model_name = model or DEFAULT_MODEL
        self._scorer = scorer
        self._timeout = timeout
        self._model = None  # 延迟加载

    def _load_model(self):
        """延迟加载 Cross-Encoder 模型。

        异常:
            ImportError: 未安装 sentence-transformers 时抛出
            RuntimeError: 模型加载失败时抛出
        """
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers 未安装，请运行: "
                "uv pip install sentence-transformers"
            ) from e

        try:
            self._model = CrossEncoder(self._model_name)
        except Exception as e:
            raise RuntimeError(
                f"Cross-Encoder 模型加载失败 (model={self._model_name}): {e}"
            ) from e

    def _score_pairs(
        self, query: str, candidates: List[Candidate]
    ) -> List[float]:
        """对 (query, passage) 对进行打分。

        优先使用注入的 scorer，否则使用加载的 Cross-Encoder 模型。

        参数:
            query: 查询文本
            candidates: 候选文档列表

        返回:
            每个候选的相关性分数列表
        """
        if self._scorer is not None:
            pairs = [(query, c.text) for c in candidates]
            return self._scorer(pairs)

        self._load_model()
        pairs = [(query, c.text) for c in candidates]
        scores = self._model.predict(pairs)
        return scores.tolist()

    def rerank(
        self,
        query: str,
        candidates: List[Candidate],
        top_k: Optional[int] = None,
        **kwargs,
    ) -> List[RankedCandidate]:
        """通过 Cross-Encoder 对候选文档进行重排序。

        参数:
            query: 查询文本
            candidates: 待重排序的候选文档列表
            top_k: 返回的最大结果数（None 表示返回全部）
            **kwargs: 额外参数
                - scorer: 临时覆盖打分函数

        返回:
            按重排序分数降序排列的 RankedCandidate 列表

        异常:
            RuntimeError: 打分失败且无法回退时抛出
        """
        if not candidates:
            return []

        # 支持临时覆盖 scorer
        scorer = kwargs.get("scorer", self._scorer)

        # 打分
        t0 = time.time()
        try:
            if scorer is not None:
                pairs = [(query, c.text) for c in candidates]
                scores = scorer(pairs)
            else:
                self._load_model()
                pairs = [(query, c.text) for c in candidates]
                scores = self._model.predict(pairs)
                scores = scores.tolist()
        except Exception as e:
            raise RuntimeError(
                f"Cross-Encoder 打分失败: {type(e).__name__}: {e}"
            ) from e

        t1 = time.time()
        elapsed_ms = (t1 - t0) * 1000

        # 超时检查
        if elapsed_ms > self._timeout * 1000:
            raise RuntimeError(
                f"Cross-Encoder 超时 ({elapsed_ms:.0f}ms > {self._timeout*1000:.0f}ms)"
            )

        # 构建排序结果
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        ranked_candidates = []
        for rank, (candidate, score) in enumerate(scored):
            ranked_candidates.append(RankedCandidate(
                id=candidate.id,
                text=candidate.text,
                rerank_score=float(score),
                original_score=candidate.score,
                metadata={
                    **candidate.metadata,
                    "rerank_elapsed_ms": round(elapsed_ms, 1),
                },
            ))

        if top_k is not None:
            ranked_candidates = ranked_candidates[:top_k]

        return ranked_candidates
