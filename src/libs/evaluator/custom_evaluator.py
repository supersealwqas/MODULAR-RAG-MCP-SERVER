"""自定义轻量评估器模块。

实现基础的检索质量指标：hit_rate 和 MRR（Mean Reciprocal Rank）。
"""

from __future__ import annotations

from typing import List, Optional

from src.libs.evaluator.base_evaluator import (
    BaseEvaluator,
    EvalCase,
    EvalReport,
    EvalResult,
)


def compute_hit_rate(retrieved_ids: List[str], golden_ids: List[str], k: int = 10) -> float:
    """计算 Hit Rate@K 指标。

    Hit Rate@K 表示在 Top-K 检索结果中是否包含至少一个相关文档。

    参数:
        retrieved_ids: 检索返回的文档 ID 列表
        golden_ids: 标注的相关文档 ID 列表
        k: 考虑的 Top-K 结果数

    返回:
        1.0 表示命中，0.0 表示未命中
    """
    if not golden_ids:
        return 0.0

    top_k = set(retrieved_ids[:k])
    golden = set(golden_ids)

    if top_k & golden:
        return 1.0
    return 0.0


def compute_mrr(retrieved_ids: List[str], golden_ids: List[str]) -> float:
    """计算 MRR（Mean Reciprocal Rank）指标。

    MRR 是第一个相关文档排名的倒数。如果没有相关文档出现在检索结果中，返回 0。

    参数:
        retrieved_ids: 检索返回的文档 ID 列表
        golden_ids: 标注的相关文档 ID 列表

    返回:
        MRR 值（0.0 到 1.0）
    """
    if not golden_ids:
        return 0.0

    golden = set(golden_ids)

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in golden:
            return 1.0 / rank

    return 0.0


def compute_precision_at_k(retrieved_ids: List[str], golden_ids: List[str], k: int = 10) -> float:
    """计算 Precision@K 指标。

    Precision@K 表示 Top-K 结果中相关文档的比例。

    参数:
        retrieved_ids: 检索返回的文档 ID 列表
        golden_ids: 标注的相关文档 ID 列表
        k: 考虑的 Top-K 结果数

    返回:
        Precision@K 值（0.0 到 1.0）
    """
    if not golden_ids or k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]

    # 如果 Top-K 为空，返回 0.0
    if not top_k:
        return 0.0

    golden = set(golden_ids)
    hits = sum(1 for doc_id in top_k if doc_id in golden)
    return hits / len(top_k)


class CustomEvaluator(BaseEvaluator):
    """自定义轻量评估器，实现基础检索质量指标。

    支持的指标：
    - hit_rate: Hit Rate@K（是否命中）
    - mrr: Mean Reciprocal Rank（平均倒数排名）
    - precision@K: Precision@K（精确率）
    """

    def __init__(self, k: int = 10):
        """初始化评估器。

        参数:
            k: 评估时考虑的 Top-K 结果数
        """
        self.k = k

    def evaluate(self, cases: List[EvalCase], **kwargs) -> EvalReport:
        """批量评估多个用例。

        对每个用例计算 hit_rate、mrr 和 precision@k 指标。

        参数:
            cases: 评估用例列表
            **kwargs: 提供者特定参数

        返回:
            EvalReport 评估报告，包含各用例结果和汇总指标
        """
        results = []

        for case in cases:
            metrics = {
                "hit_rate": compute_hit_rate(case.retrieved_ids, case.golden_ids, self.k),
                "mrr": compute_mrr(case.retrieved_ids, case.golden_ids),
                f"precision@{self.k}": compute_precision_at_k(
                    case.retrieved_ids, case.golden_ids, self.k
                ),
            }

            results.append(EvalResult(
                query=case.query,
                metrics=metrics,
                metadata=case.metadata,
            ))

        return EvalReport(results=results)
