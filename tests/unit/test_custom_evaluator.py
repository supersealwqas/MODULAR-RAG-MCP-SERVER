"""测试 Evaluator 抽象接口、CustomEvaluator 和工厂。"""

import pytest

from src.libs.evaluator.base_evaluator import (
    BaseEvaluator,
    EvalCase,
    EvalReport,
    EvalResult,
)
from src.libs.evaluator.custom_evaluator import (
    CustomEvaluator,
    compute_hit_rate,
    compute_mrr,
    compute_precision_at_k,
)
from src.libs.evaluator.evaluator_factory import (
    EvaluatorFactory,
    register_evaluator,
    _EVALUATOR_REGISTRY,
)


# --- 指标函数测试 ---

@pytest.mark.unit
class TestComputeHitRate:
    """测试 hit_rate 计算函数。"""

    def test_hit(self):
        """检索结果包含相关文档时应返回 1.0。"""
        assert compute_hit_rate(["a", "b", "c"], ["b"]) == 1.0

    def test_miss(self):
        """检索结果不包含相关文档时应返回 0.0。"""
        assert compute_hit_rate(["a", "b", "c"], ["d"]) == 0.0

    def test_multiple_golden(self):
        """多个相关文档，命中一个即可。"""
        assert compute_hit_rate(["a", "b"], ["c", "d"]) == 0.0
        assert compute_hit_rate(["a", "b"], ["a", "d"]) == 1.0

    def test_k_limit(self):
        """应只考虑 Top-K 结果。"""
        assert compute_hit_rate(["a", "b", "c"], ["c"], k=2) == 0.0
        assert compute_hit_rate(["a", "b", "c"], ["c"], k=3) == 1.0

    def test_empty_golden(self):
        """无相关文档时应返回 0.0。"""
        assert compute_hit_rate(["a", "b"], []) == 0.0

    def test_empty_retrieved(self):
        """无检索结果时应返回 0.0。"""
        assert compute_hit_rate([], ["a"]) == 0.0

    def test_k_zero_or_negative(self):
        """k <= 0 时应返回 0.0。"""
        assert compute_hit_rate(["a", "b"], ["a"], k=0) == 0.0
        assert compute_hit_rate(["a", "b"], ["a"], k=-1) == 0.0

    def test_duplicate_retrieved_ids(self):
        """检索结果有重复时应正常处理。"""
        assert compute_hit_rate(["a", "a", "b"], ["a"]) == 1.0

    def test_duplicate_golden_ids(self):
        """相关文档有重复时应正常处理。"""
        assert compute_hit_rate(["a", "b"], ["a", "a"]) == 1.0


@pytest.mark.unit
class TestComputeMrr:
    """测试 MRR 计算函数。"""

    def test_first_rank(self):
        """相关文档排第一时 MRR = 1.0。"""
        assert compute_mrr(["a", "b", "c"], ["a"]) == 1.0

    def test_second_rank(self):
        """相关文档排第二时 MRR = 0.5。"""
        assert compute_mrr(["a", "b", "c"], ["b"]) == 0.5

    def test_third_rank(self):
        """相关文档排第三时 MRR = 1/3。"""
        assert compute_mrr(["a", "b", "c"], ["c"]) == pytest.approx(1.0 / 3.0)

    def test_not_found(self):
        """相关文档不在结果中时 MRR = 0.0。"""
        assert compute_mrr(["a", "b"], ["c"]) == 0.0

    def test_multiple_golden_first_match(self):
        """多个相关文档，取第一个匹配的排名。"""
        assert compute_mrr(["a", "b", "c"], ["b", "c"]) == 0.5

    def test_empty_golden(self):
        """无相关文档时 MRR = 0.0。"""
        assert compute_mrr(["a", "b"], []) == 0.0

    def test_empty_retrieved(self):
        """无检索结果时 MRR = 0.0。"""
        assert compute_mrr([], ["a"]) == 0.0

    def test_duplicate_golden_ids(self):
        """相关文档有重复时应正常处理。"""
        assert compute_mrr(["a", "b", "c"], ["b", "b"]) == 0.5


@pytest.mark.unit
class TestComputePrecisionAtK:
    """测试 Precision@K 计算函数。"""

    def test_all_relevant(self):
        """全部命中时 Precision = 1.0。"""
        assert compute_precision_at_k(["a", "b"], ["a", "b"], k=2) == 1.0

    def test_partial_hit(self):
        """部分命中时按比例计算。"""
        assert compute_precision_at_k(["a", "b", "c"], ["a", "d"], k=3) == pytest.approx(1.0 / 3.0)

    def test_none_relevant(self):
        """全部未命中时 Precision = 0.0。"""
        assert compute_precision_at_k(["a", "b"], ["c", "d"], k=2) == 0.0

    def test_k_limit(self):
        """应只考虑 Top-K 结果。"""
        assert compute_precision_at_k(["a", "b", "c"], ["c"], k=2) == 0.0
        assert compute_precision_at_k(["a", "b", "c"], ["c"], k=3) == pytest.approx(1.0 / 3.0)

    def test_k_zero_or_negative(self):
        """k <= 0 时应返回 0.0。"""
        assert compute_precision_at_k(["a", "b"], ["a"], k=0) == 0.0
        assert compute_precision_at_k(["a", "b"], ["a"], k=-1) == 0.0

    def test_empty_retrieved(self):
        """无检索结果时应返回 0.0。"""
        assert compute_precision_at_k([], ["a"], k=5) == 0.0


# --- 数据类测试 ---

@pytest.mark.unit
class TestEvalCase:
    """测试 EvalCase 数据类。"""

    def test_create_case(self):
        """应能创建包含所有字段的评估用例。"""
        case = EvalCase(
            query="What is RAG?",
            retrieved_ids=["doc1", "doc2"],
            golden_ids=["doc1"],
        )
        assert case.query == "What is RAG?"
        assert len(case.retrieved_ids) == 2
        assert len(case.golden_ids) == 1

    def test_optional_fields(self):
        """可选字段应有默认值。"""
        case = EvalCase(query="test", retrieved_ids=[], golden_ids=[])
        assert case.generated_answer is None
        assert case.reference_answer is None
        assert case.metadata == {}


@pytest.mark.unit
class TestEvalReport:
    """测试 EvalReport 数据类。"""

    def test_compute_summary(self):
        """应自动计算指标均值。"""
        results = [
            EvalResult(query="q1", metrics={"hit_rate": 1.0, "mrr": 0.5}),
            EvalResult(query="q2", metrics={"hit_rate": 0.0, "mrr": 0.3}),
        ]
        report = EvalReport(results=results)
        assert report.total_cases == 2
        assert report.summary["hit_rate"] == 0.5
        assert report.summary["mrr"] == 0.4

    def test_empty_report(self):
        """空报告应有合理的默认值。"""
        report = EvalReport(results=[])
        assert report.total_cases == 0
        assert report.summary == {}


# --- CustomEvaluator 测试 ---

@pytest.mark.unit
class TestCustomEvaluator:
    """测试 CustomEvaluator 实现。"""

    def test_evaluate_single_case(self):
        """应能评估单个用例。"""
        evaluator = CustomEvaluator(k=10)
        case = EvalCase(
            query="test",
            retrieved_ids=["a", "b", "c"],
            golden_ids=["b"],
        )
        report = evaluator.evaluate([case])
        assert report.total_cases == 1
        assert report.results[0].metrics["hit_rate"] == 1.0
        assert report.results[0].metrics["mrr"] == 0.5

    def test_evaluate_multiple_cases(self):
        """应能批量评估多个用例。"""
        evaluator = CustomEvaluator(k=10)
        cases = [
            EvalCase(query="q1", retrieved_ids=["a"], golden_ids=["a"]),
            EvalCase(query="q2", retrieved_ids=["b"], golden_ids=["c"]),
        ]
        report = evaluator.evaluate(cases)
        assert report.total_cases == 2
        assert report.summary["hit_rate"] == 0.5

    def test_evaluate_single_convenience(self):
        """evaluate_single 应返回单个结果。"""
        evaluator = CustomEvaluator(k=5)
        case = EvalCase(
            query="test",
            retrieved_ids=["a", "b"],
            golden_ids=["a"],
        )
        result = evaluator.evaluate_single(case)
        assert isinstance(result, EvalResult)
        assert result.metrics["hit_rate"] == 1.0

    def test_custom_k(self):
        """应支持自定义 k 值。"""
        evaluator = CustomEvaluator(k=2)
        case = EvalCase(
            query="test",
            retrieved_ids=["a", "b", "c"],
            golden_ids=["c"],
        )
        result = evaluator.evaluate_single(case)
        assert result.metrics["hit_rate"] == 0.0  # c 不在 Top-2 中

    def test_metrics_keys(self):
        """结果应包含所有指标键。"""
        evaluator = CustomEvaluator(k=5)
        case = EvalCase(query="test", retrieved_ids=["a"], golden_ids=["a"])
        result = evaluator.evaluate_single(case)
        assert "hit_rate" in result.metrics
        assert "mrr" in result.metrics
        assert "precision@5" in result.metrics

    def test_empty_retrieved_ids(self):
        """空检索结果应返回全 0 指标。"""
        evaluator = CustomEvaluator(k=5)
        case = EvalCase(query="test", retrieved_ids=[], golden_ids=["a"])
        result = evaluator.evaluate_single(case)
        assert result.metrics["hit_rate"] == 0.0
        assert result.metrics["mrr"] == 0.0
        assert result.metrics["precision@5"] == 0.0

    def test_empty_golden_ids(self):
        """空相关文档应返回全 0 指标。"""
        evaluator = CustomEvaluator(k=5)
        case = EvalCase(query="test", retrieved_ids=["a", "b"], golden_ids=[])
        result = evaluator.evaluate_single(case)
        assert result.metrics["hit_rate"] == 0.0
        assert result.metrics["mrr"] == 0.0

    def test_evaluate_empty_cases(self):
        """空测试用例列表应返回空报告。"""
        evaluator = CustomEvaluator(k=5)
        report = evaluator.evaluate([])
        assert report.total_cases == 0
        assert report.summary == {}
        assert len(report.results) == 0


# --- EvaluatorFactory 测试 ---

@pytest.mark.unit
class TestEvaluatorFactory:
    """测试 EvaluatorFactory 的路由逻辑。"""

    def test_create_custom_provider(self):
        """应能创建内置的 CustomEvaluator。"""
        evaluator = EvaluatorFactory.create(provider="custom")
        assert isinstance(evaluator, CustomEvaluator)

    def test_create_default_provider(self):
        """默认提供者应为 custom。"""
        evaluator = EvaluatorFactory.create()
        assert isinstance(evaluator, CustomEvaluator)

    def test_unknown_provider_raises(self):
        """未注册的提供者应抛出 ValueError。"""
        with pytest.raises(ValueError, match="未知的评估器提供者.*unknown"):
            EvaluatorFactory.create(provider="unknown")

    def test_case_insensitive_provider(self):
        """提供者名称匹配应不区分大小写。"""
        evaluator = EvaluatorFactory.create(provider="CUSTOM")
        assert isinstance(evaluator, CustomEvaluator)

    def test_list_providers(self):
        """应列出所有已注册的提供者，包含内置的 custom。"""
        providers = EvaluatorFactory.list_providers()
        assert "custom" in providers

    def test_pass_kwargs(self):
        """应将 kwargs 传递给构造函数。"""
        evaluator = EvaluatorFactory.create(provider="custom", k=5)
        assert evaluator.k == 5


@pytest.mark.unit
class TestRegisterEvaluatorDecorator:
    """测试 @register_evaluator 装饰器。"""

    def teardown_method(self):
        """清理注册表。"""
        _EVALUATOR_REGISTRY.pop("test_provider", None)

    def test_register_decorator(self):
        """@register_evaluator 应将类添加到注册表。"""
        @register_evaluator("test_provider")
        class TestEvaluator(BaseEvaluator):
            def evaluate(self, cases, **kwargs):
                return EvalReport(results=[])

        assert "test_provider" in _EVALUATOR_REGISTRY
        assert _EVALUATOR_REGISTRY["test_provider"] is TestEvaluator

    def test_register_lowercase(self):
        """提供者名称应存储为小写。"""
        @register_evaluator("TEST_UPPER")
        class UpperEvaluator(BaseEvaluator):
            def evaluate(self, cases, **kwargs):
                return EvalReport(results=[])

        assert "test_upper" in _EVALUATOR_REGISTRY
        _EVALUATOR_REGISTRY.pop("test_upper", None)
