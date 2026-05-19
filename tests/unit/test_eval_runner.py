"""EvalRunner 单元测试。

测试评估运行器的核心功能：
- 加载黄金测试集
- 执行检索并构造 EvalCase
- 调用评估器计算指标
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.types import RetrievalResult
from src.libs.evaluator.base_evaluator import EvalCase, EvalReport, EvalResult
from src.observability.evaluation.eval_runner import EvalRunner


# ──────────────────────────────────────────────
# 辅助：Mock 组件
# ──────────────────────────────────────────────


@pytest.fixture
def mock_settings():
    """创建 mock Settings。"""
    settings = MagicMock()
    settings.evaluation.backends = ["custom"]
    return settings


@pytest.fixture
def mock_hybrid_search():
    """创建 mock HybridSearch。"""
    search = MagicMock()
    # 默认返回两个结果
    search.search.return_value = [
        RetrievalResult(
            chunk_id="chunk_001",
            score=0.9,
            text="测试文本1",
            metadata={"source": "test.pdf"},
        ),
        RetrievalResult(
            chunk_id="chunk_002",
            score=0.8,
            text="测试文本2",
            metadata={"source": "test.pdf"},
        ),
    ]
    return search


@pytest.fixture
def mock_evaluator():
    """创建 mock Evaluator。"""
    evaluator = MagicMock()
    evaluator.evaluate.return_value = EvalReport(
        results=[
            EvalResult(
                query="测试问题",
                metrics={"hit_rate": 1.0, "mrr": 0.5},
            ),
        ],
        summary={"hit_rate": 1.0, "mrr": 0.5},
    )
    return evaluator


@pytest.fixture
def golden_test_set_file():
    """创建临时黄金测试集文件。"""
    test_data = {
        "test_cases": [
            {
                "query": "如何配置 Ollama？",
                "expected_chunk_ids": ["chunk_001", "chunk_003"],
                "expected_sources": ["config_guide.pdf"],
            },
            {
                "query": "什么是 RAG？",
                "expected_chunk_ids": ["chunk_005"],
                "expected_sources": ["rag_overview.pdf"],
            },
        ]
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(test_data, f, ensure_ascii=False)
        temp_path = f.name

    yield temp_path

    # 清理
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def empty_test_set_file():
    """创建空测试集文件。"""
    test_data = {"test_cases": []}

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(test_data, f, ensure_ascii=False)
        temp_path = f.name

    yield temp_path

    Path(temp_path).unlink(missing_ok=True)


# ──────────────────────────────────────────────
# 测试：基本功能
# ──────────────────────────────────────────────


class TestEvalRunnerBasic:
    """基本功能测试。"""

    def test_init_with_components(self, mock_settings, mock_hybrid_search, mock_evaluator):
        """初始化时传入组件。"""
        runner = EvalRunner(
            settings=mock_settings,
            hybrid_search=mock_hybrid_search,
            evaluator=mock_evaluator,
        )

        assert runner._settings == mock_settings
        assert runner._hybrid_search == mock_hybrid_search
        assert runner._evaluator == mock_evaluator

    def test_run_returns_report(
        self, mock_settings, mock_hybrid_search, mock_evaluator, golden_test_set_file
    ):
        """run() 应返回 EvalReport。"""
        runner = EvalRunner(
            settings=mock_settings,
            hybrid_search=mock_hybrid_search,
            evaluator=mock_evaluator,
        )

        report = runner.run(golden_test_set_file)

        assert isinstance(report, EvalReport)
        assert report.total_cases > 0

    def test_run_calls_hybrid_search(
        self, mock_settings, mock_hybrid_search, mock_evaluator, golden_test_set_file
    ):
        """run() 应调用 HybridSearch.search()。"""
        runner = EvalRunner(
            settings=mock_settings,
            hybrid_search=mock_hybrid_search,
            evaluator=mock_evaluator,
        )

        runner.run(golden_test_set_file, top_k=5)

        # 应为每个 test_case 调用一次 search
        assert mock_hybrid_search.search.call_count == 2
        # 验证参数
        call_args = mock_hybrid_search.search.call_args_list[0]
        assert call_args[1]["top_k"] == 5

    def test_run_converts_to_eval_cases(
        self, mock_settings, mock_hybrid_search, mock_evaluator, golden_test_set_file
    ):
        """run() 应将检索结果转换为 EvalCase。"""
        runner = EvalRunner(
            settings=mock_settings,
            hybrid_search=mock_hybrid_search,
            evaluator=mock_evaluator,
        )

        runner.run(golden_test_set_file)

        # 验证 evaluate 被调用
        mock_evaluator.evaluate.assert_called_once()
        call_args = mock_evaluator.evaluate.call_args[0][0]  # 第一个参数是 EvalCase 列表

        assert len(call_args) == 2
        assert isinstance(call_args[0], EvalCase)
        assert call_args[0].query == "如何配置 Ollama？"
        assert call_args[0].golden_ids == ["chunk_001", "chunk_003"]
        assert call_args[0].retrieved_ids == ["chunk_001", "chunk_002"]


# ──────────────────────────────────────────────
# 测试：边界情况
# ──────────────────────────────────────────────


class TestEvalRunnerEdgeCases:
    """边界情况测试。"""

    def test_run_with_empty_test_set(
        self, mock_settings, mock_hybrid_search, mock_evaluator, empty_test_set_file
    ):
        """空测试集应返回空报告。"""
        runner = EvalRunner(
            settings=mock_settings,
            hybrid_search=mock_hybrid_search,
            evaluator=mock_evaluator,
        )

        report = runner.run(empty_test_set_file)

        assert isinstance(report, EvalReport)
        assert report.total_cases == 0

    def test_run_with_nonexistent_file(self, mock_settings, mock_hybrid_search, mock_evaluator):
        """不存在的文件应返回空报告。"""
        runner = EvalRunner(
            settings=mock_settings,
            hybrid_search=mock_hybrid_search,
            evaluator=mock_evaluator,
        )

        report = runner.run("nonexistent_file.json")

        assert isinstance(report, EvalReport)
        assert report.total_cases == 0

    def test_run_with_search_error(
        self, mock_settings, mock_hybrid_search, mock_evaluator, golden_test_set_file
    ):
        """检索失败时应记录空结果。"""
        mock_hybrid_search.search.side_effect = RuntimeError("检索失败")

        runner = EvalRunner(
            settings=mock_settings,
            hybrid_search=mock_hybrid_search,
            evaluator=mock_evaluator,
        )

        report = runner.run(golden_test_set_file)

        # 评估器仍应被调用
        mock_evaluator.evaluate.assert_called_once()
        call_args = mock_evaluator.evaluate.call_args[0][0]
        # 检索失败时 retrieved_ids 为空
        assert call_args[0].retrieved_ids == []

    def test_run_with_list_format(self, mock_settings, mock_hybrid_search, mock_evaluator):
        """支持直接列表格式的测试集。"""
        test_data = [
            {"query": "测试问题", "expected_chunk_ids": ["chunk_001"]},
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(test_data, f, ensure_ascii=False)
            temp_path = f.name

        try:
            runner = EvalRunner(
                settings=mock_settings,
                hybrid_search=mock_hybrid_search,
                evaluator=mock_evaluator,
            )

            report = runner.run(temp_path)

            assert isinstance(report, EvalReport)
        finally:
            Path(temp_path).unlink(missing_ok=True)


# ──────────────────────────────────────────────
# 测试：run_from_dict
# ──────────────────────────────────────────────


class TestEvalRunnerFromDict:
    """run_from_dict 方法测试。"""

    def test_run_from_dict(self, mock_settings, mock_hybrid_search, mock_evaluator):
        """run_from_dict 应直接从字典列表运行评估。"""
        test_cases = [
            {"query": "测试问题1", "expected_chunk_ids": ["chunk_001"]},
            {"query": "测试问题2", "expected_chunk_ids": ["chunk_002"]},
        ]

        runner = EvalRunner(
            settings=mock_settings,
            hybrid_search=mock_hybrid_search,
            evaluator=mock_evaluator,
        )

        report = runner.run_from_dict(test_cases, top_k=5)

        assert isinstance(report, EvalReport)
        assert mock_hybrid_search.search.call_count == 2

    def test_run_from_dict_with_collection(
        self, mock_settings, mock_hybrid_search, mock_evaluator
    ):
        """run_from_dict 应支持 collection 过滤。"""
        test_cases = [
            {"query": "测试问题", "expected_chunk_ids": ["chunk_001"]},
        ]

        runner = EvalRunner(
            settings=mock_settings,
            hybrid_search=mock_hybrid_search,
            evaluator=mock_evaluator,
        )

        runner.run_from_dict(test_cases, collection="test_collection")

        call_args = mock_hybrid_search.search.call_args
        assert call_args[1]["filters"]["collection"] == "test_collection"


# ──────────────────────────────────────────────
# 测试：加载测试集
# ──────────────────────────────────────────────


class TestEvalRunnerLoadTestSet:
    """加载测试集测试。"""

    def test_load_test_set_dict_format(self, mock_settings, golden_test_set_file):
        """加载 dict 格式测试集。"""
        runner = EvalRunner(settings=mock_settings)

        test_cases = runner._load_test_set(golden_test_set_file)

        assert len(test_cases) == 2
        assert test_cases[0]["query"] == "如何配置 Ollama？"

    def test_load_test_set_list_format(self, mock_settings):
        """加载 list 格式测试集。"""
        test_data = [{"query": "测试"}]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(test_data, f, ensure_ascii=False)
            temp_path = f.name

        try:
            runner = EvalRunner(settings=mock_settings)
            test_cases = runner._load_test_set(temp_path)

            assert len(test_cases) == 1
            assert test_cases[0]["query"] == "测试"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_test_set_invalid_json(self, mock_settings):
        """无效 JSON 应返回空列表。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write("invalid json content")
            temp_path = f.name

        try:
            runner = EvalRunner(settings=mock_settings)
            test_cases = runner._load_test_set(temp_path)

            assert test_cases == []
        finally:
            Path(temp_path).unlink(missing_ok=True)
