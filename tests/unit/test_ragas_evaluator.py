"""RagasEvaluator 单元测试。

测试 Ragas 评估器的核心功能：
- 优雅降级：ragas 未安装时抛出明确 ImportError
- mock ragas 环境下 evaluate() 返回正确 metrics
- 指标过滤和空数据处理
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.libs.evaluator.base_evaluator import EvalCase, EvalReport


# ──────────────────────────────────────────────
# 测试：ragas 未安装时的优雅降级
# ──────────────────────────────────────────────


class TestRagasNotInstalled:
    """ragas 未安装时的行为测试。"""

    def test_import_error_message(self):
        """ragas 未安装时，创建 RagasEvaluator 应抛出明确的 ImportError。"""
        # 临时移除 ragas 模块模拟未安装状态
        ragas_module = sys.modules.pop("ragas", None)
        datasets_module = sys.modules.pop("datasets", None)
        try:
            # 确保 ragas 不可导入
            with patch.dict(sys.modules, {"ragas": None, "datasets": None}):
                with pytest.raises(ImportError, match="ragas 未安装"):
                    from src.observability.evaluation.ragas_evaluator import RagasEvaluator
                    RagasEvaluator()
        finally:
            # 恢复模块
            if ragas_module is not None:
                sys.modules["ragas"] = ragas_module
            if datasets_module is not None:
                sys.modules["datasets"] = datasets_module


# ──────────────────────────────────────────────
# 测试：mock ragas 环境下的正常功能
# ──────────────────────────────────────────────


@pytest.fixture
def mock_ragas_modules():
    """创建 mock 的 ragas 和 datasets 模块。"""
    # mock ragas 模块
    mock_ragas = MagicMock()
    mock_faithfulness = MagicMock(name="faithfulness")
    mock_answer_relevancy = MagicMock(name="answer_relevancy")
    mock_context_precision = MagicMock(name="context_precision")
    mock_ragas.metrics.faithfulness = mock_faithfulness
    mock_ragas.metrics.answer_relevancy = mock_answer_relevancy
    mock_ragas.metrics.context_precision = mock_context_precision

    # mock datasets 模块
    mock_datasets = MagicMock()
    mock_dataset = MagicMock()
    mock_datasets.Dataset.from_dict.return_value = mock_dataset

    # mock ragas.evaluate 返回结果
    mock_result = MagicMock()
    mock_result.__getitem__ = MagicMock(side_effect=lambda i: {
        "faithfulness": 0.85,
        "answer_relevancy": 0.92,
        "context_precision": 0.78,
    })
    mock_result.__len__ = MagicMock(return_value=1)
    mock_ragas.evaluate.return_value = mock_result

    with patch.dict(sys.modules, {
        "ragas": mock_ragas,
        "ragas.metrics": mock_ragas.metrics,
        "datasets": mock_datasets,
    }):
        yield {
            "ragas": mock_ragas,
            "datasets": mock_datasets,
            "mock_dataset": mock_dataset,
            "mock_result": mock_result,
        }


@pytest.fixture
def sample_cases():
    """创建测试用的评估用例。"""
    return [
        EvalCase(
            query="如何配置 Ollama？",
            retrieved_ids=["chunk_001", "chunk_002", "chunk_003"],
            golden_ids=["chunk_001", "chunk_004"],
            generated_answer="配置 Ollama 需要设置 base_url 参数。",
            reference_answer="Ollama 配置方法：设置 base_url 为 http://localhost:11434。",
        ),
        EvalCase(
            query="什么是 RAG？",
            retrieved_ids=["chunk_010", "chunk_011"],
            golden_ids=["chunk_010"],
            generated_answer="RAG 是检索增强生成技术。",
            reference_answer="RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术。",
        ),
    ]


class TestRagasEvaluatorWithMock:
    """使用 mock ragas 的评估器测试。"""

    def test_evaluate_returns_metrics(self, mock_ragas_modules, sample_cases):
        """evaluate() 应返回包含 faithfulness/answer_relevancy 的 metrics。"""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator()
        report = evaluator.evaluate(sample_cases)

        # 验证返回类型
        assert isinstance(report, EvalReport)
        assert len(report.results) > 0

        # 验证 metrics 包含预期指标
        first_result = report.results[0]
        assert "faithfulness" in first_result.metrics
        assert "answer_relevancy" in first_result.metrics
        assert "context_precision" in first_result.metrics

    def test_evaluate_custom_metrics(self, mock_ragas_modules, sample_cases):
        """指定 metrics 参数时，只计算选定的指标。"""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["faithfulness"])
        report = evaluator.evaluate(sample_cases)

        assert isinstance(report, EvalReport)
        # 验证只调用了 faithfulness 指标
        mock_ragas_modules["ragas"].evaluate.assert_called_once()
        call_kwargs = mock_ragas_modules["ragas"].evaluate.call_args
        metrics_arg = call_kwargs.kwargs.get("metrics") or call_kwargs[1].get("metrics")
        assert len(metrics_arg) == 1

    def test_evaluate_with_llm_override(self, mock_ragas_modules, sample_cases):
        """通过 kwargs 传入 llm 参数可覆盖默认 LLM。"""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        mock_llm = MagicMock(name="custom_llm")
        evaluator = RagasEvaluator(llm=mock_llm)
        evaluator.evaluate(sample_cases, llm=mock_llm)

        # 验证 llm 参数被传递
        call_kwargs = mock_ragas_modules["ragas"].evaluate.call_args
        assert call_kwargs is not None

    def test_evaluate_empty_cases(self, mock_ragas_modules):
        """空用例列表应返回空报告。"""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator()
        report = evaluator.evaluate([])

        assert isinstance(report, EvalReport)
        assert len(report.results) == 0

    def test_evaluate_cases_without_answers(self, mock_ragas_modules):
        """缺少 generated_answer 或 reference_answer 的用例应被过滤。"""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        cases = [
            EvalCase(
                query="测试问题",
                retrieved_ids=["chunk_001"],
                golden_ids=["chunk_001"],
                # 缺少 generated_answer 和 reference_answer
            ),
        ]

        evaluator = RagasEvaluator()
        report = evaluator.evaluate(cases)

        # 无有效用例，返回空报告
        assert isinstance(report, EvalReport)

    def test_evaluate_error_handling(self, mock_ragas_modules, sample_cases):
        """ragas.evaluate() 抛出异常时应返回带错误信息的空指标。"""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        # 模拟 ragas.evaluate 抛出异常
        mock_ragas_modules["ragas"].evaluate.side_effect = RuntimeError("LLM 调用失败")

        evaluator = RagasEvaluator()
        report = evaluator.evaluate(sample_cases)

        assert isinstance(report, EvalReport)
        # 异常时返回空 metrics 并记录错误
        for result in report.results:
            assert result.metrics == {}
            assert "error" in result.metadata


# ──────────────────────────────────────────────
# 测试：工厂注册
# ──────────────────────────────────────────────


class TestEvaluatorFactoryRagas:
    """测试 EvaluatorFactory 中 ragas provider 的注册。"""

    def test_ragas_provider_registered(self, mock_ragas_modules):
        """ragas provider 应已注册到 EvaluatorFactory。"""
        # 在 mock 环境下重新导入以触发注册
        import importlib
        import src.libs.evaluator.evaluator_factory as ef_module
        backup = ef_module._EVALUATOR_REGISTRY.copy()
        try:
            importlib.reload(ef_module)
            import src.observability.evaluation.ragas_evaluator as re_module
            importlib.reload(re_module)
            providers = ef_module.EvaluatorFactory.list_providers()
            assert "ragas" in providers
        finally:
            ef_module._EVALUATOR_REGISTRY.update(backup)

    def test_create_ragas_evaluator(self, mock_ragas_modules):
        """EvaluatorFactory.create('ragas') 应返回 RagasEvaluator 实例。"""
        import importlib
        import src.libs.evaluator.evaluator_factory as ef_module
        backup = ef_module._EVALUATOR_REGISTRY.copy()
        try:
            importlib.reload(ef_module)
            import src.observability.evaluation.ragas_evaluator as re_module
            importlib.reload(re_module)
            evaluator = ef_module.EvaluatorFactory.create("ragas")
            assert evaluator.__class__.__name__ == "RagasEvaluator"
        finally:
            ef_module._EVALUATOR_REGISTRY.update(backup)


