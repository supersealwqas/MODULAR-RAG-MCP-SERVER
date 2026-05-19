"""CompositeEvaluator 单元测试。

测试组合评估器的核心功能：
- 并行执行多个评估器
- 合并指标结果
- 异常处理和降级
- 空数据处理
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.libs.evaluator.base_evaluator import (
    BaseEvaluator,
    EvalCase,
    EvalReport,
    EvalResult,
)
from src.observability.evaluation.composite_evaluator import CompositeEvaluator


# ──────────────────────────────────────────────
# 辅助：Fake 评估器
# ──────────────────────────────────────────────


class FakeEvaluatorA(BaseEvaluator):
    """假评估器 A，返回 hit_rate 和 mrr。"""

    def __init__(self, hit_rate: float = 0.8, mrr: float = 0.6):
        self.hit_rate = hit_rate
        self.mrr = mrr

    def evaluate(self, cases: list[EvalCase], **kwargs) -> EvalReport:
        results = []
        for case in cases:
            results.append(EvalResult(
                query=case.query,
                metrics={"hit_rate": self.hit_rate, "mrr": self.mrr},
            ))
        return EvalReport(results=results)


class FakeEvaluatorB(BaseEvaluator):
    """假评估器 B，返回 faithfulness。"""

    def __init__(self, faithfulness: float = 0.9):
        self.faithfulness = faithfulness

    def evaluate(self, cases: list[EvalCase], **kwargs) -> EvalReport:
        results = []
        for case in cases:
            results.append(EvalResult(
                query=case.query,
                metrics={"faithfulness": self.faithfulness},
            ))
        return EvalReport(results=results)


class FailingEvaluator(BaseEvaluator):
    """总是抛出异常的评估器。"""

    def evaluate(self, cases: list[EvalCase], **kwargs) -> EvalReport:
        raise RuntimeError("评估器内部错误")


# ──────────────────────────────────────────────
# 测试用例
# ──────────────────────────────────────────────


@pytest.fixture
def sample_cases():
    """创建测试用的评估用例。"""
    return [
        EvalCase(
            query="如何配置 Ollama？",
            retrieved_ids=["chunk_001", "chunk_002"],
            golden_ids=["chunk_001"],
            generated_answer="配置 Ollama 需要设置 base_url。",
            reference_answer="Ollama 配置方法：设置 base_url。",
        ),
        EvalCase(
            query="什么是 RAG？",
            retrieved_ids=["chunk_010", "chunk_011"],
            golden_ids=["chunk_010"],
            generated_answer="RAG 是检索增强生成。",
            reference_answer="RAG 是一种结合检索和生成的技术。",
        ),
    ]


class TestCompositeEvaluatorBasic:
    """基本功能测试。"""

    def test_init_with_evaluators(self):
        """初始化时传入评估器列表。"""
        evaluator_a = FakeEvaluatorA()
        evaluator_b = FakeEvaluatorB()
        composite = CompositeEvaluator([evaluator_a, evaluator_b])

        assert len(composite.evaluators) == 2

    def test_init_empty_evaluators_raises(self):
        """空评估器列表应抛出 ValueError。"""
        with pytest.raises(ValueError, match="evaluators 列表不能为空"):
            CompositeEvaluator([])

    def test_evaluate_returns_merged_metrics(self, sample_cases):
        """evaluate() 应返回包含所有评估器指标的合并结果。"""
        evaluator_a = FakeEvaluatorA(hit_rate=0.8, mrr=0.6)
        evaluator_b = FakeEvaluatorB(faithfulness=0.9)
        composite = CompositeEvaluator([evaluator_a, evaluator_b])

        report = composite.evaluate(sample_cases)

        assert isinstance(report, EvalReport)
        assert len(report.results) == 2

        # 验证指标合并
        first_result = report.results[0]
        assert "hit_rate" in first_result.metrics
        assert "mrr" in first_result.metrics
        assert "faithfulness" in first_result.metrics
        assert first_result.metrics["hit_rate"] == 0.8
        assert first_result.metrics["mrr"] == 0.6
        assert first_result.metrics["faithfulness"] == 0.9

    def test_evaluate_preserves_query(self, sample_cases):
        """评估结果应保留原始 query。"""
        evaluator_a = FakeEvaluatorA()
        composite = CompositeEvaluator([evaluator_a])

        report = composite.evaluate(sample_cases)

        assert report.results[0].query == "如何配置 Ollama？"
        assert report.results[1].query == "什么是 RAG？"


class TestCompositeEvaluatorEdgeCases:
    """边界情况测试。"""

    def test_evaluate_empty_cases(self):
        """空用例列表应返回空报告。"""
        evaluator = FakeEvaluatorA()
        composite = CompositeEvaluator([evaluator])

        report = composite.evaluate([])

        assert isinstance(report, EvalReport)
        assert len(report.results) == 0

    def test_evaluate_single_evaluator(self):
        """单个评估器也能正常工作。"""
        evaluator = FakeEvaluatorA(hit_rate=0.7, mrr=0.5)
        composite = CompositeEvaluator([evaluator])

        cases = [
            EvalCase(
                query="测试问题",
                retrieved_ids=["chunk_001"],
                golden_ids=["chunk_001"],
                generated_answer="测试回答",
                reference_answer="参考答案",
            ),
        ]

        report = composite.evaluate(cases)

        assert len(report.results) == 1
        assert report.results[0].metrics["hit_rate"] == 0.7
        assert report.results[0].metrics["mrr"] == 0.5

    def test_evaluate_with_failing_evaluator(self):
        """单个评估器失败时，其他评估器结果仍保留。"""
        evaluator_a = FakeEvaluatorA(hit_rate=0.8, mrr=0.6)
        failing_evaluator = FailingEvaluator()
        composite = CompositeEvaluator([evaluator_a, failing_evaluator])

        cases = [
            EvalCase(
                query="测试问题",
                retrieved_ids=["chunk_001"],
                golden_ids=["chunk_001"],
                generated_answer="测试回答",
                reference_answer="参考答案",
            ),
        ]

        report = composite.evaluate(cases)

        assert len(report.results) == 1
        # FakeEvaluatorA 的指标应保留
        assert "hit_rate" in report.results[0].metrics
        assert report.results[0].metrics["hit_rate"] == 0.8
        # 错误信息应记录在 metadata 中
        assert "error" in report.results[0].metadata

    def test_metric_override_on_conflict(self):
        """指标名称冲突时，后面的评估器覆盖前面的。"""
        evaluator_a = FakeEvaluatorA(hit_rate=0.8, mrr=0.6)
        # 创建一个返回同名指标的评估器
        evaluator_override = MagicMock(spec=BaseEvaluator)
        evaluator_override.evaluate.return_value = EvalReport(
            results=[
                EvalResult(
                    query="测试",
                    metrics={"hit_rate": 0.95, "custom_metric": 1.0},
                )
            ]
        )
        composite = CompositeEvaluator([evaluator_a, evaluator_override])

        cases = [
            EvalCase(
                query="测试",
                retrieved_ids=["chunk_001"],
                golden_ids=["chunk_001"],
                generated_answer="回答",
                reference_answer="参考",
            ),
        ]

        report = composite.evaluate(cases)

        # hit_rate 应被覆盖为 0.95
        assert report.results[0].metrics["hit_rate"] == 0.95
        # mrr 应保留
        assert report.results[0].metrics["mrr"] == 0.6
        # custom_metric 应添加
        assert report.results[0].metrics["custom_metric"] == 1.0


class TestCompositeEvaluatorSingleCase:
    """单用例评估测试。"""

    def test_evaluate_single(self):
        """evaluate_single 应返回单个用例的合并结果。"""
        evaluator_a = FakeEvaluatorA(hit_rate=0.8, mrr=0.6)
        evaluator_b = FakeEvaluatorB(faithfulness=0.9)
        composite = CompositeEvaluator([evaluator_a, evaluator_b])

        case = EvalCase(
            query="测试问题",
            retrieved_ids=["chunk_001"],
            golden_ids=["chunk_001"],
            generated_answer="测试回答",
            reference_answer="参考答案",
        )

        result = composite.evaluate_single(case)

        assert isinstance(result, EvalResult)
        assert result.query == "测试问题"
        assert "hit_rate" in result.metrics
        assert "faithfulness" in result.metrics


class TestCompositeEvaluatorIntegration:
    """集成测试：与真实 Evaluator 组合。"""

    def test_with_custom_evaluator(self):
        """与 CustomEvaluator 组合使用。"""
        from src.libs.evaluator.custom_evaluator import CustomEvaluator

        custom_eval = CustomEvaluator(k=5)
        composite = CompositeEvaluator([custom_eval])

        cases = [
            EvalCase(
                query="测试",
                retrieved_ids=["a", "b", "c"],
                golden_ids=["a"],
                generated_answer="回答",
                reference_answer="参考",
            ),
        ]

        report = composite.evaluate(cases)

        assert len(report.results) == 1
        assert "hit_rate" in report.results[0].metrics
        assert "mrr" in report.results[0].metrics
        assert "precision@5" in report.results[0].metrics
