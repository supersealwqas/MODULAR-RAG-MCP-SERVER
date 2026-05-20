"""D2 DenseRetriever 单元测试。

使用 Mock EmbeddingClient 和 VectorStore 隔离测试，覆盖：
- 基本检索流程（embed → query → RetrievalResult）
- 空查询处理
- top_k 传递
- filters 传递
- Trace 记录
- 依赖注入
- 错误处理
验收标准：mock EmbeddingClient 和 VectorStore 时能正确编排调用。
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import pytest

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult
from src.core.query_engine.dense_retriever import DenseRetriever
from src.libs.vector_store.base_vector_store import QueryResult


# ============================================================
# 测试辅助函数
# ============================================================


def _make_settings_stub(**kwargs) -> Settings:
    """创建测试用 Settings stub。"""
    settings = Settings(
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
        pipeline=MagicMock(),
    )
    return settings


def _make_mock_embedding(vector_dim: int = 4) -> MagicMock:
    """创建 Mock EmbeddingClient。"""
    mock = MagicMock()
    mock.embed_single.return_value = [0.1, 0.2, 0.3, 0.4][:vector_dim]
    return mock


def _make_mock_vector_store(results: List[QueryResult] = None) -> MagicMock:
    """创建 Mock VectorStore。"""
    mock = MagicMock()
    mock.query.return_value = results or []
    return mock


def _make_query_results(count: int = 3) -> List[QueryResult]:
    """创建测试用 QueryResult 列表。"""
    results = []
    for i in range(count):
        results.append(QueryResult(
            id=f"chunk_{i:04d}",
            score=1.0 - i * 0.1,
            text=f"这是第 {i} 个检索结果的文本内容",
            metadata={"source_path": f"/test_{i}.pdf", "chunk_index": i},
        ))
    return results


# ============================================================
# 基本检索流程测试
# ============================================================


class TestRetrieveBasic:
    """基本检索流程测试。"""

    def test_retrieve_returns_retrieval_results(self):
        """retrieve 返回 RetrievalResult 列表。"""
        settings = _make_settings_stub()
        embedding = _make_mock_embedding()
        vs = _make_mock_vector_store(_make_query_results(3))

        retriever = DenseRetriever(settings, embedding_client=embedding, vector_store=vs)
        results = retriever.retrieve("test query")

        assert len(results) == 3
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_retrieve_calls_embed_single(self):
        """retrieve 正确调用 embedding_client.embed_single。"""
        settings = _make_settings_stub()
        embedding = _make_mock_embedding()
        vs = _make_mock_vector_store()

        retriever = DenseRetriever(settings, embedding_client=embedding, vector_store=vs)
        retriever.retrieve("hello world")

        embedding.embed_single.assert_called_once_with("hello world")

    def test_retrieve_calls_vector_store_query(self):
        """retrieve 正确调用 vector_store.query。"""
        settings = _make_settings_stub()
        embedding = _make_mock_embedding()
        vs = _make_mock_vector_store()

        retriever = DenseRetriever(settings, embedding_client=embedding, vector_store=vs)
        retriever.retrieve("test", top_k=5, filters={"collection": "docs"})

        vs.query.assert_called_once()
        call_kwargs = vs.query.call_args
        assert call_kwargs.kwargs["top_k"] == 5
        assert call_kwargs.kwargs["filters"] == {"collection": "docs"}

    def test_retrieve_preserves_chunk_id(self):
        """retrieve 正确映射 QueryResult.id → RetrievalResult.chunk_id。"""
        settings = _make_settings_stub()
        embedding = _make_mock_embedding()
        query_results = _make_query_results(2)
        vs = _make_mock_vector_store(query_results)

        retriever = DenseRetriever(settings, embedding_client=embedding, vector_store=vs)
        results = retriever.retrieve("test")

        assert results[0].chunk_id == "chunk_0000"
        assert results[1].chunk_id == "chunk_0001"

    def test_retrieve_preserves_score(self):
        """retrieve 正确映射 QueryResult.score → RetrievalResult.score。"""
        settings = _make_settings_stub()
        embedding = _make_mock_embedding()
        query_results = _make_query_results(2)
        vs = _make_mock_vector_store(query_results)

        retriever = DenseRetriever(settings, embedding_client=embedding, vector_store=vs)
        results = retriever.retrieve("test")

        assert results[0].score == pytest.approx(1.0)
        assert results[1].score == pytest.approx(0.9)

    def test_retrieve_preserves_text(self):
        """retrieve 正确映射 QueryResult.text → RetrievalResult.text。"""
        settings = _make_settings_stub()
        embedding = _make_mock_embedding()
        query_results = _make_query_results(1)
        vs = _make_mock_vector_store(query_results)

        retriever = DenseRetriever(settings, embedding_client=embedding, vector_store=vs)
        results = retriever.retrieve("test")

        assert results[0].text == "这是第 0 个检索结果的文本内容"

    def test_retrieve_preserves_metadata(self):
        """retrieve 正确映射 QueryResult.metadata → RetrievalResult.metadata。"""
        settings = _make_settings_stub()
        embedding = _make_mock_embedding()
        query_results = _make_query_results(1)
        vs = _make_mock_vector_store(query_results)

        retriever = DenseRetriever(settings, embedding_client=embedding, vector_store=vs)
        results = retriever.retrieve("test")

        assert results[0].metadata["source_path"] == "/test_0.pdf"


# ============================================================
# 空查询与边界测试
# ============================================================


class TestRetrieveEdgeCases:
    """空查询与边界场景测试。"""

    def test_empty_query_returns_empty(self):
        """空查询返回空列表。"""
        settings = _make_settings_stub()
        embedding = _make_mock_embedding()
        vs = _make_mock_vector_store()

        retriever = DenseRetriever(settings, embedding_client=embedding, vector_store=vs)
        assert retriever.retrieve("") == []
        assert retriever.retrieve("   ") == []

    def test_no_results_returns_empty(self):
        """无检索结果时返回空列表。"""
        settings = _make_settings_stub()
        embedding = _make_mock_embedding()
        vs = _make_mock_vector_store([])

        retriever = DenseRetriever(settings, embedding_client=embedding, vector_store=vs)
        results = retriever.retrieve("test")
        assert results == []

    def test_default_top_k_from_settings(self):
        """默认 top_k 从 settings.retrieval.top_k 读取。"""
        settings = _make_settings_stub(top_k=7)
        embedding = _make_mock_embedding()
        vs = _make_mock_vector_store()

        retriever = DenseRetriever(settings, embedding_client=embedding, vector_store=vs)
        retriever.retrieve("test")

        call_kwargs = vs.query.call_args.kwargs
        assert call_kwargs["top_k"] == 7

    def test_explicit_top_k_overrides_settings(self):
        """显式 top_k 参数覆盖 settings 默认值。"""
        settings = _make_settings_stub(top_k=7)
        embedding = _make_mock_embedding()
        vs = _make_mock_vector_store()

        retriever = DenseRetriever(settings, embedding_client=embedding, vector_store=vs)
        retriever.retrieve("test", top_k=3)

        call_kwargs = vs.query.call_args.kwargs
        assert call_kwargs["top_k"] == 3


# ============================================================
# Trace 记录测试
# ============================================================


class TestTraceRecording:
    """Trace 记录测试。"""

    def test_trace_records_dense_retrieval(self):
        """传入 trace 时记录 dense_retrieval 阶段。"""
        settings = _make_settings_stub()
        embedding = _make_mock_embedding()
        vs = _make_mock_vector_store(_make_query_results(2))
        trace = TraceContext()

        retriever = DenseRetriever(settings, embedding_client=embedding, vector_store=vs)
        retriever.retrieve("test query", trace=trace)

        stages = [s for s in trace.stages if s["name"] == "dense_retrieval"]
        assert len(stages) == 1
        assert stages[0]["result_count"] == 2
        assert "embed_ms" in stages[0]
        assert "query_ms" in stages[0]
        assert "elapsed_ms" in stages[0]

    def test_no_trace_no_error(self):
        """不传 trace 时不报错。"""
        settings = _make_settings_stub()
        embedding = _make_mock_embedding()
        vs = _make_mock_vector_store(_make_query_results(1))

        retriever = DenseRetriever(settings, embedding_client=embedding, vector_store=vs)
        results = retriever.retrieve("test")
        assert len(results) == 1


# ============================================================
# 依赖注入测试
# ============================================================


class TestDependencyInjection:
    """依赖注入测试。"""

    def test_injected_clients_used(self):
        """注入的 embedding_client 和 vector_store 被直接使用。"""
        settings = _make_settings_stub()
        embedding = _make_mock_embedding()
        vs = _make_mock_vector_store(_make_query_results(1))

        retriever = DenseRetriever(settings, embedding_client=embedding, vector_store=vs)
        retriever.retrieve("test")

        # 验证注入的 mock 被调用
        embedding.embed_single.assert_called_once()
        vs.query.assert_called_once()

    def test_retrieval_result_serializable(self):
        """RetrievalResult 支持序列化。"""
        settings = _make_settings_stub()
        embedding = _make_mock_embedding()
        vs = _make_mock_vector_store(_make_query_results(1))

        retriever = DenseRetriever(settings, embedding_client=embedding, vector_store=vs)
        results = retriever.retrieve("test")

        d = results[0].to_dict()
        assert d["chunk_id"] == "chunk_0000"
        assert d["score"] == pytest.approx(1.0)
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

    def test_embedding_error_propagates(self):
        """Embedding 调用异常向上传播。"""
        settings = _make_settings_stub()
        embedding = _make_mock_embedding()
        embedding.embed_single.side_effect = RuntimeError("embedding failed")
        vs = _make_mock_vector_store()

        retriever = DenseRetriever(settings, embedding_client=embedding, vector_store=vs)
        with pytest.raises(RuntimeError, match="embedding failed"):
            retriever.retrieve("test")

    def test_vector_store_error_propagates(self):
        """VectorStore 调用异常向上传播。"""
        settings = _make_settings_stub()
        embedding = _make_mock_embedding()
        vs = _make_mock_vector_store()
        vs.query.side_effect = RuntimeError("store failed")

        retriever = DenseRetriever(settings, embedding_client=embedding, vector_store=vs)
        with pytest.raises(RuntimeError, match="store failed"):
            retriever.retrieve("test")
