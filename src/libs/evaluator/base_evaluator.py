"""评估器的抽象基类模块。

定义所有 Evaluator 实现必须遵循的接口规范。
评估器用于衡量 RAG 系统的检索和生成质量。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
 

@dataclass
class EvalCase:
    """评估用例数据类，表示一条评估样本。

    属性:
        query: 查询文本
        retrieved_ids: 检索返回的文档 ID 列表（按相关性降序）
        golden_ids: 标注的相关文档 ID 列表（ground truth）
        generated_answer: 生成的回答文本（可选，用于评估生成质量）
        reference_answer: 参考答案文本（可选）
        metadata: 附加元数据
    """
    query: str
    retrieved_ids: List[str]
    golden_ids: List[str]
    generated_answer: Optional[str] = None
    reference_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """评估结果数据类，表示一条评估用例的结果。

    属性:
        query: 查询文本
        metrics: 指标字典（如 {"hit_rate": 1.0, "mrr": 0.5}）
        metadata: 附加元数据
    """
    query: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    """评估报告数据类，表示批量评估的汇总结果。

    属性:
        results: 各评估用例的结果列表
        summary: 汇总指标（各指标的均值）
        total_cases: 评估用例总数
    """
    results: List[EvalResult]
    summary: Dict[str, float] = field(default_factory=dict)
    total_cases: int = 0

    def __post_init__(self):
        """初始化后计算汇总指标。"""
        self.total_cases = len(self.results)
        if self.results:
            self._compute_summary()

    def _compute_summary(self):
        """计算各指标的均值。"""
        if not self.results:
            return

        metric_keys = set()
        for result in self.results:
            metric_keys.update(result.metrics.keys())

        for key in metric_keys:
            values = [r.metrics.get(key, 0.0) for r in self.results]
            self.summary[key] = sum(values) / len(values)


class BaseEvaluator(ABC):
    """所有评估器的抽象基类。

    子类必须实现 `evaluate` 方法来完成实际的评估逻辑。
    """

    @abstractmethod
    def evaluate(self, cases: List[EvalCase], **kwargs) -> EvalReport:
        """批量评估多个用例。

        参数:
            cases: 评估用例列表
            **kwargs: 提供者特定参数

        返回:
            EvalReport 评估报告
        """
        ...

    def evaluate_single(self, case: EvalCase, **kwargs) -> EvalResult:
        """评估单个用例的便捷方法。

        参数:
            case: 评估用例
            **kwargs: 提供者特定参数

        返回:
            EvalResult 评估结果
        """
        report = self.evaluate([case], **kwargs)
        return report.results[0] if report.results else EvalResult(
            query=case.query, metrics={}
        )
