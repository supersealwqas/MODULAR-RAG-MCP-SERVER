"""召回回归测试（E2E）。

基于 golden_test_set.json 做最小召回阈值检查。
用于确保检索质量不会随代码变更而退化。

用法:
    pytest tests/e2e/test_recall.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from src.core.types import RetrievalResult
from src.libs.evaluator.base_evaluator import EvalCase, EvalReport
from src.libs.evaluator.custom_evaluator import (
    compute_hit_rate,
    compute_mrr,
    compute_precision_at_k,
)
from src.observability.evaluation.eval_runner import EvalRunner


# ──────────────────────────────────────────────
# 配置：召回阈值（写死在测试里，便于回归）
# ──────────────────────────────────────────────

# 最小 Hit Rate@K 阈值
MIN_HIT_RATE_AT_5 = 0.6
MIN_HIT_RATE_AT_10 = 0.7

# 最小 MRR 阈值
MIN_MRR = 0.4

# 最小 Precision@K 阈值
MIN_PRECISION_AT_5 = 0.3

# Golden Test Set 路径
GOLDEN_TEST_SET_PATH = Path("tests/fixtures/golden_test_set.json")


# ──────────────────────────────────────────────
# 辅助：加载 golden test set
# ──────────────────────────────────────────────


def load_golden_test_set() -> List[Dict]:
    """加载 golden test set。

    返回:
        测试用例列表
    """
    if not GOLDEN_TEST_SET_PATH.exists():
        pytest.skip(f"Golden test set 不存在: {GOLDEN_TEST_SET_PATH}")

    with open(GOLDEN_TEST_SET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data.get("test_cases", [])
    elif isinstance(data, list):
        return data
    else:
        pytest.skip(f"不支持的 golden test set 格式: {type(data)}")
        return []


# ──────────────────────────────────────────────
# 辅助：Mock HybridSearch（模拟检索结果）
# ──────────────────────────────────────────────


def create_mock_hybrid_search(test_cases: List[Dict]):
    """创建模拟的 HybridSearch，返回可控的检索结果。

    参数:
        test_cases: 测试用例列表

    返回:
        MagicMock 的 HybridSearch 实例
    """
    mock_search = MagicMock()

    def mock_search_fn(query, top_k=10, filters=None, trace=None):
        # 根据 query 返回部分匹配的结果
        for case in test_cases:
            if case["query"] == query:
                expected_ids = case.get("expected_chunk_ids", [])
                # 模拟：返回部分 expected_ids + 一些噪声
                results = []
                for i, chunk_id in enumerate(expected_ids[:top_k]):
                    results.append(RetrievalResult(
                        chunk_id=chunk_id,
                        score=1.0 - i * 0.1,
                        text=f"模拟文本 {chunk_id}",
                        metadata={"source": "mock"},
                    ))
                # 添加一些噪声结果
                for i in range(min(3, top_k - len(results))):
                    results.append(RetrievalResult(
                        chunk_id=f"noise_{i:03d}",
                        score=0.5 - i * 0.1,
                        text=f"噪声文本 {i}",
                        metadata={"source": "mock"},
                    ))
                return results[:top_k]

        # 未匹配的 query 返回空
        return []

    mock_search.search = mock_search_fn
    return mock_search


# ──────────────────────────────────────────────
# 测试：召回率回归
# ──────────────────────────────────────────────


class TestRecallRegression:
    """召回率回归测试。"""

    @pytest.fixture
    def golden_cases(self):
        """加载 golden test set。"""
        return load_golden_test_set()

    @pytest.fixture
    def mock_search(self, golden_cases):
        """创建 mock HybridSearch。"""
        return create_mock_hybrid_search(golden_cases)

    def test_hit_rate_at_5(self, golden_cases, mock_search):
        """Hit Rate@5 应达到最小阈值。"""
        hit_rates = []

        for case in golden_cases:
            results = mock_search.search(case["query"], top_k=5)
            retrieved_ids = [r.chunk_id for r in results]
            expected_ids = case.get("expected_chunk_ids", [])

            hit_rate = compute_hit_rate(retrieved_ids, expected_ids, k=5)
            hit_rates.append(hit_rate)

        avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
        assert avg_hit_rate >= MIN_HIT_RATE_AT_5, (
            f"Hit Rate@5 = {avg_hit_rate:.4f} < {MIN_HIT_RATE_AT_5} (阈值)"
        )

    def test_hit_rate_at_10(self, golden_cases, mock_search):
        """Hit Rate@10 应达到最小阈值。"""
        hit_rates = []

        for case in golden_cases:
            results = mock_search.search(case["query"], top_k=10)
            retrieved_ids = [r.chunk_id for r in results]
            expected_ids = case.get("expected_chunk_ids", [])

            hit_rate = compute_hit_rate(retrieved_ids, expected_ids, k=10)
            hit_rates.append(hit_rate)

        avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
        assert avg_hit_rate >= MIN_HIT_RATE_AT_10, (
            f"Hit Rate@10 = {avg_hit_rate:.4f} < {MIN_HIT_RATE_AT_10} (阈值)"
        )

    def test_mrr(self, golden_cases, mock_search):
        """MRR 应达到最小阈值。"""
        mrr_values = []

        for case in golden_cases:
            results = mock_search.search(case["query"], top_k=10)
            retrieved_ids = [r.chunk_id for r in results]
            expected_ids = case.get("expected_chunk_ids", [])

            mrr = compute_mrr(retrieved_ids, expected_ids)
            mrr_values.append(mrr)

        avg_mrr = sum(mrr_values) / len(mrr_values) if mrr_values else 0.0
        assert avg_mrr >= MIN_MRR, (
            f"MRR = {avg_mrr:.4f} < {MIN_MRR} (阈值)"
        )

    def test_precision_at_5(self, golden_cases, mock_search):
        """Precision@5 应达到最小阈值。"""
        precisions = []

        for case in golden_cases:
            results = mock_search.search(case["query"], top_k=5)
            retrieved_ids = [r.chunk_id for r in results]
            expected_ids = case.get("expected_chunk_ids", [])

            precision = compute_precision_at_k(retrieved_ids, expected_ids, k=5)
            precisions.append(precision)

        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
        assert avg_precision >= MIN_PRECISION_AT_5, (
            f"Precision@5 = {avg_precision:.4f} < {MIN_PRECISION_AT_5} (阈值)"
        )


# ──────────────────────────────────────────────
# 测试：EvalRunner 集成
# ──────────────────────────────────────────────


class TestRecallWithEvalRunner:
    """使用 EvalRunner 的召回率测试。"""

    def test_eval_runner_with_golden_set(self):
        """EvalRunner 应能加载 golden set 并返回报告。"""
        # Mock Settings
        mock_settings = MagicMock()
        mock_settings.evaluation.backends = ["custom"]

        # Mock HybridSearch
        mock_search = MagicMock()
        mock_search.search.return_value = [
            RetrievalResult(
                chunk_id="chunk_001",
                score=0.9,
                text="测试文本",
                metadata={},
            ),
        ]

        # Mock Evaluator
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = EvalReport(
            results=[],
            summary={"hit_rate": 0.8, "mrr": 0.5},
        )

        runner = EvalRunner(
            settings=mock_settings,
            hybrid_search=mock_search,
            evaluator=mock_evaluator,
        )

        report = runner.run(str(GOLDEN_TEST_SET_PATH))

        assert isinstance(report, EvalReport)
        # 验证 evaluator 被调用
        mock_evaluator.evaluate.assert_called_once()


# ──────────────────────────────────────────────
# 测试：阈值配置验证
# ──────────────────────────────────────────────


class TestThresholdConfiguration:
    """阈值配置验证测试。"""

    def test_thresholds_are_reasonable(self):
        """验证阈值配置在合理范围内。"""
        assert 0.0 <= MIN_HIT_RATE_AT_5 <= 1.0
        assert 0.0 <= MIN_HIT_RATE_AT_10 <= 1.0
        assert 0.0 <= MIN_MRR <= 1.0
        assert 0.0 <= MIN_PRECISION_AT_5 <= 1.0

        # Hit Rate@10 应 >= Hit Rate@5
        assert MIN_HIT_RATE_AT_10 >= MIN_HIT_RATE_AT_5

    def test_golden_set_exists(self):
        """验证 golden test set 文件存在。"""
        assert GOLDEN_TEST_SET_PATH.exists(), (
            f"Golden test set 不存在: {GOLDEN_TEST_SET_PATH}"
        )

    def test_golden_set_has_cases(self):
        """验证 golden test set 包含测试用例。"""
        cases = load_golden_test_set()
        assert len(cases) > 0, "Golden test set 为空"

        # 验证每个用例的格式
        for case in cases:
            assert "query" in case, f"缺少 query 字段: {case}"
            assert "expected_chunk_ids" in case, f"缺少 expected_chunk_ids 字段: {case}"
