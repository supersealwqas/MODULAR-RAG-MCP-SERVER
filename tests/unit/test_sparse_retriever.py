"""D3 SparseRetriever 单元测试。

使用 Mock BM25Indexer 和 VectorStore 隔离测试，覆盖：
- 基本检索流程（keywords → bm25.query → get_by_ids → RetrievalResult）
- 空关键词处理
- top_k 传递
- Trace 记录
- 依赖注入
- 错误处理
验收标准：mock BM25Indexer 和 VectorStore 时能正确编排调用。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import pytest

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult
from src.core.query_engine.sparse_retriever import SparseRetriever


# ============================================================
# 测试辅助函数
# ============================================================


def _make_settings_stub(**kwargs) -> Settings:
    """创建测试用 Settings stub。"""
    return Settings(
        llm=MagicMock(),
        vision_llm=MagicMock(),
        ollama=MagicMock(),
        embedding=MagicMock(),
        splitter=MagicMock(),
        vector_store=MagicMock(),
        retrieval=MagicMock(top_k=kwargs.get("top_k", 10)),
        rerank=MagicMock(),
        evaluation=MagicMock(),
        observability=MagicMock(),
    )


def _make_mock_bm25(results: List[Tuple[str, float]] = None) -> MagicMock:
    """创建 Mock BM25Indexer。"""
    mock = MagicMock()
    mock.query.return_value = results or []
    return mock


def _make_mock_vector_store(records: List[Dict[str, Any]] = None) -> MagicMock:
    """创建 Mock VectorStore。"""
    mock = MagicMock()
    mock.get_by_ids.return_value = records or []
    return mock


def _make_bm25_results(count: int = 3) -> List[Tuple[str, float]]:
    """创建测试用 BM25 结果列表。"""
    return [(f"chunk_{i:04d}", 10.0 - i * 1.0) for i in range(count)]


def _make_vector_records(count: int = 3) -> List[Dict[str, Any]]:
    """创建测试用 VectorStore 记录列表。"""
    return [
        {
            "id": f"chunk_{i:04d}",
            "text": f"这是第 {i} 个检索结果的文本内容",
            "metadata": {"source_path": f"/test_{i}.pdf", "chunk_index": i},
        }
        for i in range(count)
    ]


# ============================================================
# 基本检索流程测试
# ============================================================


class TestRetrieveBasic:
    """基本检索流程测试。"""

    def test_retrieve_returns_retrieval_results(self):
        """retrieve 返回 RetrievalResult 列表。"""
        settings = _make_settings_stub()
        bm25 = _make_mock_bm25(_make_bm25_results(3))
        vs = _make_mock_vector_store(_make_vector_records(3))

        retriever = SparseRetriever(settings, bm25_indexer=bm25, vector_store=vs)
        results = retriever.retrieve(["test", "query"])

        assert len(results) == 3
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_retrieve_calls_bm25_query(self):
        """retrieve 正确调用 bm25_indexer.query。"""
        settings = _make_settings_stub()
        bm25 = _make_mock_bm25()
        vs = _make_mock_vector_store()

        retriever = SparseRetriever(settings, bm25_indexer=bm25, vector_store=vs)
        retriever.retrieve(["hello", "world"], top_k=5)

        bm25.query.assert_called_once_with(["hello", "world"], top_k=5)

    def test_retrieve_calls_get_by_ids(self):
        """retrieve 正确调用 vector_store.get_by_ids。"""
        settings = _make_settings_stub()
        bm25 = _make_mock_bm25(_make_bm25_results(3))
        vs = _make_mock_vector_store()

        retriever = SparseRetriever(settings, bm25_indexer=bm25, vector_store=vs)
        retriever.retrieve(["test"])

        vs.get_by_ids.assert_called_once()
        call_args = vs.get_by_ids.call_args[0][0]
        assert call_args == ["chunk_0000", "chunk_0001", "chunk_0002"]

    def test_retrieve_preserves_chunk_id(self):
        """retrieve 正确映射 chunk_id。"""
        settings = _make_settings_stub()
        bm25 = _make_mock_bm25(_make_bm25_results(2))
        vs = _make_mock_vector_store(_make_vector_records(2))

        retriever = SparseRetriever(settings, bm25_indexer=bm25, vector_store=vs)
        results = retriever.retrieve(["test"])

        assert results[0].chunk_id == "chunk_0000"
        assert results[1].chunk_id == "chunk_0001"

    def test_retrieve_preserves_score(self):
        """retrieve 正确映射 BM25 分数。"""
        settings = _make_settings_stub()
        bm25 = _make_mock_bm25(_make_bm25_results(2))
        vs = _make_mock_vector_store(_make_vector_records(2))

        retriever = SparseRetriever(settings, bm25_indexer=bm25, vector_store=vs)
        results = retriever.retrieve(["test"])

        assert results[0].score == pytest.approx(10.0)
        assert results[1].score == pytest.approx(9.0)

    def test_retrieve_preserves_text(self):
        """retrieve 正确映射 text。"""
        settings = _make_settings_stub()
        bm25 = _make_mock_bm25(_make_bm25_results(1))
        vs = _make_mock_vector_store(_make_vector_records(1))

        retriever = SparseRetriever(settings, bm25_indexer=bm25, vector_store=vs)
        results = retriever.retrieve(["test"])

        assert results[0].text == "这是第 0 个检索结果的文本内容"

    def test_retrieve_preserves_metadata(self):
        """retrieve 正确映射 metadata。"""
        settings = _make_settings_stub()
        bm25 = _make_mock_bm25(_make_bm25_results(1))
        vs = _make_mock_vector_store(_make_vector_records(1))

        retriever = SparseRetriever(settings, bm25_indexer=bm25, vector_store=vs)
        results = retriever.retrieve(["test"])

        assert results[0].metadata["source_path"] == "/test_0.pdf"


# ============================================================
# 空查询与边界测试
# ============================================================


class TestRetrieveEdgeCases:
    """空查询与边界场景测试。"""

    def test_empty_keywords_returns_empty(self):
        """空关键词列表返回空列表。"""
        settings = _make_settings_stub()
        bm25 = _make_mock_bm25()
        vs = _make_mock_vector_store()

        retriever = SparseRetriever(settings, bm25_indexer=bm25, vector_store=vs)
        assert retriever.retrieve([]) == []

    def test_no_bm25_results_returns_empty(self):
        """BM25 无结果时返回空列表。"""
        settings = _make_settings_stub()
        bm25 = _make_mock_bm25([])
        vs = _make_mock_vector_store()

        retriever = SparseRetriever(settings, bm25_indexer=bm25, vector_store=vs)
        results = retriever.retrieve(["test"])
        assert results == []

    def test_default_top_k_from_settings(self):
        """默认 top_k 从 settings.retrieval.top_k 读取。"""
        settings = _make_settings_stub(top_k=7)
        bm25 = _make_mock_bm25()
        vs = _make_mock_vector_store()

        retriever = SparseRetriever(settings, bm25_indexer=bm25, vector_store=vs)
        retriever.retrieve(["test"])

        bm25.query.assert_called_once_with(["test"], top_k=7)

    def test_explicit_top_k_overrides_settings(self):
        """显式 top_k 参数覆盖 settings 默认值。"""
        settings = _make_settings_stub(top_k=7)
        bm25 = _make_mock_bm25()
        vs = _make_mock_vector_store()

        retriever = SparseRetriever(settings, bm25_indexer=bm25, vector_store=vs)
        retriever.retrieve(["test"], top_k=3)

        bm25.query.assert_called_once_with(["test"], top_k=3)


# ============================================================
# Trace 记录测试
# ============================================================


class TestTraceRecording:
    """Trace 记录测试。"""

    def test_trace_records_sparse_retrieval(self):
        """传入 trace 时记录 sparse_retrieval 阶段。"""
        settings = _make_settings_stub()
        bm25 = _make_mock_bm25(_make_bm25_results(2))
        vs = _make_mock_vector_store(_make_vector_records(2))
        trace = TraceContext()

        retriever = SparseRetriever(settings, bm25_indexer=bm25, vector_store=vs)
        retriever.retrieve(["test", "query"], trace=trace)

        stages = [s for s in trace.stages if s["name"] == "sparse_retrieval"]
        assert len(stages) == 1
        assert stages[0]["result_count"] == 2
        assert stages[0]["keyword_count"] == 2
        assert "bm25_ms" in stages[0]
        assert "lookup_ms" in stages[0]
        assert "elapsed_ms" in stages[0]

    def test_no_trace_no_error(self):
        """不传 trace 时不报错。"""
        settings = _make_settings_stub()
        bm25 = _make_mock_bm25(_make_bm25_results(1))
        vs = _make_mock_vector_store(_make_vector_records(1))

        retriever = SparseRetriever(settings, bm25_indexer=bm25, vector_store=vs)
        results = retriever.retrieve(["test"])
        assert len(results) == 1


# ============================================================
# 依赖注入测试
# ============================================================


class TestDependencyInjection:
    """依赖注入测试。"""

    def test_injected_clients_used(self):
        """注入的 bm25_indexer 和 vector_store 被直接使用。"""
        settings = _make_settings_stub()
        bm25 = _make_mock_bm25(_make_bm25_results(1))
        vs = _make_mock_vector_store(_make_vector_records(1))

        retriever = SparseRetriever(settings, bm25_indexer=bm25, vector_store=vs)
        retriever.retrieve(["test"])

        # 验证注入的 mock 被调用
        bm25.query.assert_called_once()
        vs.get_by_ids.assert_called_once()

    def test_retrieval_result_serializable(self):
        """RetrievalResult 支持序列化。"""
        settings = _make_settings_stub()
        bm25 = _make_mock_bm25(_make_bm25_results(1))
        vs = _make_mock_vector_store(_make_vector_records(1))

        retriever = SparseRetriever(settings, bm25_indexer=bm25, vector_store=vs)
        results = retriever.retrieve(["test"])

        d = results[0].to_dict()
        assert d["chunk_id"] == "chunk_0000"
        assert d["score"] == pytest.approx(10.0)
        assert "text" in d
        assert "metadata" in d

        restored = RetrievalResult.from_dict(d)
        assert restored.chunk_id == results[0].chunk_id
        assert restored.score == results[0].score


# ============================================================
# 错误处理测试
# ============================================================


class TestErrorHandling:
    """错误处理测试。"""

    def test_bm25_error_propagates(self):
        """BM25 调用异常向上传播。"""
        settings = _make_settings_stub()
        bm25 = _make_mock_bm25()
        bm25.query.side_effect = RuntimeError("bm25 failed")
        vs = _make_mock_vector_store()

        retriever = SparseRetriever(settings, bm25_indexer=bm25, vector_store=vs)
        with pytest.raises(RuntimeError, match="bm25 failed"):
            retriever.retrieve(["test"])

    def test_vector_store_error_propagates(self):
        """VectorStore 调用异常向上传播。"""
        settings = _make_settings_stub()
        bm25 = _make_mock_bm25(_make_bm25_results(1))
        vs = _make_mock_vector_store()
        vs.get_by_ids.side_effect = RuntimeError("store failed")

        retriever = SparseRetriever(settings, bm25_indexer=bm25, vector_store=vs)
        with pytest.raises(RuntimeError, match="store failed"):
            retriever.retrieve(["test"])
