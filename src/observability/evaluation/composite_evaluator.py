"""组合评估器模块。

将多个 Evaluator 组合并并行执行，汇总各评估器的指标结果。
支持配置驱动：通过 evaluation.backends 配置自动组合多个评估器。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from src.libs.evaluator.base_evaluator import (
    BaseEvaluator,
    EvalCase,
    EvalReport,
    EvalResult,
)


class CompositeEvaluator(BaseEvaluator):
    """组合评估器，并行执行多个 Evaluator 并汇总结果。

    支持的用法：
    1. 直接传入 evaluator 实例列表
    2. 通过 EvaluatorFactory 创建后组合

    属性:
        evaluators: 评估器实例列表
        max_workers: 并行执行的最大线程数
    """

    def __init__(
        self,
        evaluators: List[BaseEvaluator],
        max_workers: int = 4,
    ):
        """初始化组合评估器。

        参数:
            evaluators: 评估器实例列表
            max_workers: 并行执行的最大线程数，默认 4
        """
        if not evaluators:
            raise ValueError("evaluators 列表不能为空")
        self.evaluators = evaluators
        self.max_workers = max_workers

    def evaluate(self, cases: List[EvalCase], **kwargs) -> EvalReport:
        """并行执行所有评估器，合并指标结果。

        对每个评估器执行评估，然后合并所有指标到统一的 EvalReport。
        指标名称冲突时，后执行的评估器结果会覆盖先执行的。

        参数:
            cases: 评估用例列表
            **kwargs: 额外参数（传递给各评估器）

        返回:
            EvalReport 合并后的评估报告
        """
        if not cases:
            return EvalReport(results=[])

        # 并行执行各评估器
        all_reports: List[EvalReport] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_evaluator = {
                executor.submit(evaluator.evaluate, cases, **kwargs): evaluator
                for evaluator in self.evaluators
            }
            for future in as_completed(future_to_evaluator):
                evaluator = future_to_evaluator[future]
                try:
                    report = future.result()
                    all_reports.append(report)
                except Exception as e:
                    # 单个评估器失败时记录错误，不影响其他评估器
                    error_report = EvalReport(
                        results=[
                            EvalResult(
                                query=case.query,
                                metrics={},
                                metadata={"error": f"{evaluator.__class__.__name__}: {e}"},
                            )
                            for case in cases
                        ]
                    )
                    all_reports.append(error_report)

        # 合并所有报告
        return self._merge_reports(all_reports, cases)

    def _merge_reports(
        self, reports: List[EvalReport], cases: List[EvalCase]
    ) -> EvalReport:
        """合并多个评估报告。

        按用例索引对齐，合并各评估器的指标。
        指标名称冲突时，后面的报告覆盖前面的。

        参数:
            reports: 评估报告列表
            cases: 原始评估用例列表

        返回:
            EvalReport 合并后的报告
        """
        if not reports:
            return EvalReport(results=[])

        # 初始化合并结果
        merged_results: List[EvalResult] = []

        for i, case in enumerate(cases):
            merged_metrics: Dict[str, float] = {}
            merged_metadata: Dict[str, Any] = {}

            for report in reports:
                if i < len(report.results):
                    result = report.results[i]
                    merged_metrics.update(result.metrics)
                    merged_metadata.update(result.metadata)

            merged_results.append(
                EvalResult(
                    query=case.query,
                    metrics=merged_metrics,
                    metadata=merged_metadata,
                )
            )

        return EvalReport(results=merged_results)

    def evaluate_single(self, case: EvalCase, **kwargs) -> EvalResult:
        """评估单个用例，合并所有评估器的结果。

        参数:
            case: 评估用例
            **kwargs: 额外参数

        返回:
            EvalResult 合并后的结果
        """
        report = self.evaluate([case], **kwargs)
        return report.results[0] if report.results else EvalResult(
            query=case.query, metrics={}
        )
