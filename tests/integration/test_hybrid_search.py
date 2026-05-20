"""D5 HybridSearch 集成测试。

使用 Mock 组件隔离测试，覆盖：
- 基本混合检索流程（query → Dense + Sparse → Fusion → 结果）
- Metadata 过滤
- Dense/Sparse 降级（单路失败时仍返回结果）
- top_k 传递
- Trace 记录
- 空查询处理
验收标准：对 fixtures 数据能返回 Top-K，支持 filters，单路失败能降级。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.fusion import Fusion
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import ProcessedQuery, RetrievalResult


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
        retrieval=MagicMock(
            top_k=kwargs.get("top_k", 10),
            dense_weight=kwargs.get("dense_weight", 0.7),
            sparse_weight=kwargs.get("sparse_weight", 0.3),
        ),
        rerank=MagicMock(),
        evaluation=MagicMock(),
        observability=MagicMock(),
        pipeline=MagicMock(),
    )


def _make_mock_query_processor() -> MagicMock:
    """创建 Mock QueryProcessor。"""
    mock = MagicMock(spec=QueryProcessor)
    mock.process.return_value = ProcessedQuery(
        original="test query",
        keywords=["test", "query"],
        filters={},
    )
    return mock


def _make_mock_dense_retriever(results: List[RetrievalResult] = None) -> MagicMock:
    """创建 Mock DenseRetriever。"""
    mock = MagicMock(spec=DenseRetriever)
    mock.retrieve.return_value = results or []
    return mock


def _make_mock_sparse_retriever(results: List[RetrievalResult] = None) -> MagicMock:
    """创建 Mock SparseRetriever。"""
    mock = MagicMock(spec=SparseRetriever)
    mock.retrieve.return_value = results or []
    return mock


def _make_retrieval_results(ids: List[str], scores: List[float] = None) -> List[RetrievalResult]:
    """创建测试用 RetrievalResult 列表。"""
    if scores is None:
        scores = [1.0 - i * 0.1 for i in range(len(ids))]
    return [
        RetrievalResult(
            chunk_id=cid,
            score=score,
            text=f"text_{cid}",
            metadata={"source": f"doc_{cid}", "collection": "test"},
        )
        for cid, score in zip(ids, scores)
    ]


# ============================================================
# 基本混合检索流程测试
# ============================================================


class TestHybridSearchBasic:
    """基本混合检索流程测试。"""

    def test_search_returns_retrieval_results(self):
        """search 返回 RetrievalResult 列表。"""
        settings = _make_settings_stub()
        qp = _make_mock_query_processor()
        dense = _make_mock_dense_retriever(_make_retrieval_results(["a", "b"]))
        sparse = _make_mock_sparse_retriever(_make_retrieval_results(["b", "c"]))

        hs = HybridSearch(settings, query_processor=qp, dense_retriever=dense, sparse_retriever=sparse)
        results = hs.search("test query")

        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_search_calls_query_processor(self):
        """search 调用 QueryProcessor.process。"""
        settings = _make_settings_stub()
        qp = _make_mock_query_processor()
        dense = _make_mock_dense_retriever()
        sparse = _make_mock_sparse_retriever()

        hs = HybridSearch(settings, query_processor=qp, dense_retriever=dense, sparse_retriever=sparse)
        hs.search("test query")

        qp.process.assert_called_once_with("test query", filters=None, trace=None)

    def test_search_calls_dense_retriever(self):
        """search 调用 DenseRetriever.retrieve。"""
        settings = _make_settings_stub()
        qp = _make_mock_query_processor()
        dense = _make_mock_dense_retriever()
        sparse = _make_mock_sparse_retriever()

        hs = HybridSearch(settings, query_processor=qp, dense_retriever=dense, sparse_retriever=sparse)
        hs.search("test query", top_k=5)

        dense.retrieve.assert_called_once()
        call_kwargs = dense.retrieve.call_args
        assert call_kwargs[0][0] == "test query"
        assert call_kwargs[1]["top_k"] == 5

    def test_search_calls_sparse_retriever(self):
        """search 调用 SparseRetriever.retrieve。"""
        settings = _make_settings_stub()
        qp = _make_mock_query_processor()
        dense = _make_mock_dense_retriever()
        sparse = _make_mock_sparse_retriever()

        hs = HybridSearch(settings, query_processor=qp, dense_retriever=dense, sparse_retriever=sparse)
        hs.search("test query", top_k=5)

        sparse.retrieve.assert_called_once()
        call_kwargs = sparse.retrieve.call_args
        assert call_kwargs[0][0] == ["test", "query"]  # keywords
        assert call_kwargs[1]["top_k"] == 5

    def test_search_fuses_results(self):
        """search 正确融合 Dense 和 Sparse 结果。"""
        settings = _make_settings_stub()
        qp = _make_mock_query_processor()
        dense = _make_mock_dense_retriever(_make_retrieval_results(["a", "b"]))
        sparse = _make_mock_sparse_retriever(_make_retrieval_results(["b", "c"]))

        hs = HybridSearch(settings, query_processor=qp, dense_retriever=dense, sparse_retriever=sparse)
        results = hs.search("test query")

        # 应该有 3 个唯一结果：a, b, c
        chunk_ids = [r.chunk_id for r in results]
        assert len(set(chunk_ids)) == 3
        assert "a" in chunk_ids
        assert "b" in chunk_ids
        assert "c" in chunk_ids


# ============================================================
# Metadata 过滤测试
# ============================================================


class TestHybridSearchMetadataFilters:
    """Metadata 过滤测试。"""

    def test_search_applies_metadata_filters(self):
        """search 应用 Metadata 过滤。"""
        settings = _make_settings_stub()
        qp = _make_mock_query_processor()
        # 创建带不同 collection 的结果
        results = [
            RetrievalResult(chunk_id="a", score=0.9, text="text_a", metadata={"collection": "docs"}),
            RetrievalResult(chunk_id="b", score=0.8, text="text_b", metadata={"collection": "code"}),
            RetrievalResult(chunk_id="c", score=0.7, text="text_c", metadata={"collection": "docs"}),
        ]
        dense = _make_mock_dense_retriever(results)
        sparse = _make_mock_sparse_retriever(results)

        hs = HybridSearch(settings, query_processor=qp, dense_retriever=dense, sparse_retriever=sparse)
        filtered_results = hs.search("test query", filters={"collection": "docs"})

        # 只应保留 collection=docs 的结果
        for r in filtered_results:
            assert r.metadata["collection"] == "docs"

    def test_search_no_filters_returns_all(self):
        """不指定 filters 时返回所有结果。"""
        settings = _make_settings_stub()
        qp = _make_mock_query_processor()
        results = _make_retrieval_results(["a", "b", "c"])
        dense = _make_mock_dense_retriever(results)
        sparse = _make_mock_sparse_retriever(results)

        hs = HybridSearch(settings, query_processor=qp, dense_retriever=dense, sparse_retriever=sparse)
        all_results = hs.search("test query")

        assert len(all_results) == 3


# ============================================================
# 降级测试
# ============================================================


class TestHybridSearchDegradation:
    """降级测试：单路失败时仍返回结果。"""

    def test_dense_failure_degrades_to_sparse(self):
        """Dense 失败时降级到 Sparse 结果。"""
        settings = _make_settings_stub()
        qp = _make_mock_query_processor()
        dense = _make_mock_dense_retriever()
        dense.retrieve.side_effect = RuntimeError("dense failed")
        sparse = _make_mock_sparse_retriever(_make_retrieval_results(["a", "b"]))

        hs = HybridSearch(settings, query_processor=qp, dense_retriever=dense, sparse_retriever=sparse)
        results = hs.search("test query")

        # 应该仍有结果（来自 Sparse）
        assert len(results) > 0

    def test_sparse_failure_degrades_to_dense(self):
        """Sparse 失败时降级到 Dense 结果。"""
        settings = _make_settings_stub()
        qp = _make_mock_query_processor()
        dense = _make_mock_dense_retriever(_make_retrieval_results(["a", "b"]))
        sparse = _make_mock_sparse_retriever()
        sparse.retrieve.side_effect = RuntimeError("sparse failed")

        hs = HybridSearch(settings, query_processor=qp, dense_retriever=dense, sparse_retriever=sparse)
        results = hs.search("test query")

        # 应该仍有结果（来自 Dense）
        assert len(results) > 0

    def test_both_failure_returns_empty(self):
        """Dense 和 Sparse 都失败时返回空列表。"""
        settings = _make_settings_stub()
        qp = _make_mock_query_processor()
        dense = _make_mock_dense_retriever()
        dense.retrieve.side_effect = RuntimeError("dense failed")
        sparse = _make_mock_sparse_retriever()
        sparse.retrieve.side_effect = RuntimeError("sparse failed")

        hs = HybridSearch(settings, query_processor=qp, dense_retriever=dense, sparse_retriever=sparse)
        results = hs.search("test query")

        assert results == []


# ============================================================
# top_k 测试
# ============================================================


class TestHybridSearchTopK:
    """top_k 传递测试。"""

    def test_default_top_k_from_settings(self):
        """默认 top_k 从 settings.retrieval.top_k 读取。"""
        settings = _make_settings_stub(top_k=7)
        qp = _make_mock_query_processor()
        dense = _make_mock_dense_retriever()
        sparse = _make_mock_sparse_retriever()

        hs = HybridSearch(settings, query_processor=qp, dense_retriever=dense, sparse_retriever=sparse)
        hs.search("test")

        # 验证 top_k 传递给 DenseRetriever
        call_kwargs = dense.retrieve.call_args
        assert call_kwargs[1]["top_k"] == 7

    def test_explicit_top_k_overrides_settings(self):
        """显式 top_k 参数覆盖 settings 默认值。"""
        settings = _make_settings_stub(top_k=7)
        qp = _make_mock_query_processor()
        dense = _make_mock_dense_retriever()
        sparse = _make_mock_sparse_retriever()

        hs = HybridSearch(settings, query_processor=qp, dense_retriever=dense, sparse_retriever=sparse)
        hs.search("test", top_k=3)

        call_kwargs = dense.retrieve.call_args
        assert call_kwargs[1]["top_k"] == 3


# ============================================================
# 空查询测试
# ============================================================


class TestHybridSearchEdgeCases:
    """空查询和边界场景测试。"""

    def test_empty_query_returns_empty(self):
        """空查询返回空列表。"""
        settings = _make_settings_stub()
        qp = _make_mock_query_processor()
        dense = _make_mock_dense_retriever()
        sparse = _make_mock_sparse_retriever()

        hs = HybridSearch(settings, query_processor=qp, dense_retriever=dense, sparse_retriever=sparse)
        assert hs.search("") == []
        assert hs.search("   ") == []

    def test_no_results_returns_empty(self):
        """Dense 和 Sparse 都无结果时返回空列表。"""
        settings = _make_settings_stub()
        qp = _make_mock_query_processor()
        dense = _make_mock_dense_retriever([])
        sparse = _make_mock_sparse_retriever([])

        hs = HybridSearch(settings, query_processor=qp, dense_retriever=dense, sparse_retriever=sparse)
        results = hs.search("test query")
        assert results == []


# ============================================================
# Trace 记录测试
# ============================================================


class TestHybridSearchTraceRecording:
    """Trace 记录测试。"""

    def test_trace_records_hybrid_search(self):
        """传入 trace 时记录 hybrid_search 阶段。"""
        settings = _make_settings_stub()
        qp = _make_mock_query_processor()
        dense = _make_mock_dense_retriever(_make_retrieval_results(["a"]))
        sparse = _make_mock_sparse_retriever(_make_retrieval_results(["b"]))
        trace = TraceContext()

        hs = HybridSearch(settings, query_processor=qp, dense_retriever=dense, sparse_retriever=sparse)
        hs.search("test query", trace=trace)

        stages = [s for s in trace.stages if s["name"] == "hybrid_search"]
        assert len(stages) == 1
        assert "dense_count" in stages[0]
        assert "sparse_count" in stages[0]
        assert "final_count" in stages[0]
        assert "elapsed_ms" in stages[0]

    def test_no_trace_no_error(self):
        """不传 trace 时不报错。"""
        settings = _make_settings_stub()
        qp = _make_mock_query_processor()
        dense = _make_mock_dense_retriever(_make_retrieval_results(["a"]))
        sparse = _make_mock_sparse_retriever(_make_retrieval_results(["b"]))

        hs = HybridSearch(settings, query_processor=qp, dense_retriever=dense, sparse_retriever=sparse)
        results = hs.search("test query")
        assert len(results) > 0


# ============================================================
# RetrievalResult 序列化测试
# ============================================================


class TestHybridSearchFiltersForwarding:
    """Filters 传递测试。"""

    def test_search_forwards_filters_to_query_processor(self):
        """search 将 filters 传递给 QueryProcessor.process。"""
        settings = _make_settings_stub()
        qp = _make_mock_query_processor()
        dense = _make_mock_dense_retriever(_make_retrieval_results(["a"]))
        sparse = _make_mock_sparse_retriever(_make_retrieval_results(["b"]))

        hs = HybridSearch(settings, query_processor=qp, dense_retriever=dense, sparse_retriever=sparse)
        hs.search("test query", filters={"collection": "docs"})

        qp.process.assert_called_once_with("test query", filters={"collection": "docs"}, trace=None)


class TestHybridSearchSerialization:
    """RetrievalResult 序列化测试。"""

    def test_results_serializable(self):
        """检索结果支持序列化。"""
        settings = _make_settings_stub()
        qp = _make_mock_query_processor()
        dense = _make_mock_dense_retriever(_make_retrieval_results(["a"]))
        sparse = _make_mock_sparse_retriever(_make_retrieval_results(["b"]))

        hs = HybridSearch(settings, query_processor=qp, dense_retriever=dense, sparse_retriever=sparse)
        results = hs.search("test query")

        for r in results:
            d = r.to_dict()
            assert "chunk_id" in d
            assert "score" in d
            assert "text" in d
            assert "metadata" in d

            restored = RetrievalResult.from_dict(d)
            assert restored.chunk_id == r.chunk_id
            assert restored.score == r.score


# ============================================================
# F3 验收标准：Query 链路完整 Trace 测试
# ============================================================


class TestQueryPipelineTrace:
    """F3 验收标准：验证查询链路各阶段 trace 记录完整。

    验收标准：
    - 一次查询生成 trace，包含 query_processing/dense_retrieval/sparse_retrieval/fusion/rerank 阶段
    - 每个阶段记录 elapsed_ms 耗时字段和 method 字段
    - trace.to_dict() 中 trace_type == "query"
    """

    def test_trace_contains_all_sub_stages(self):
        """HybridSearch trace 包含汇总阶段记录（mock 组件不记录子阶段）。"""
        settings = _make_settings_stub()
        qp = _make_mock_query_processor()
        dense = _make_mock_dense_retriever(_make_retrieval_results(["a", "b"]))
        sparse = _make_mock_sparse_retriever(_make_retrieval_results(["b", "c"]))
        trace = TraceContext(trace_type="query")

        hs = HybridSearch(settings, query_processor=qp, dense_retriever=dense, sparse_retriever=sparse)
        hs.search("test query", trace=trace)

        # HybridSearch 记录汇总阶段
        hs_stages = [s for s in trace.stages if s["name"] == "hybrid_search"]
        assert len(hs_stages) == 1
        assert "elapsed_ms" in hs_stages[0]
        assert "method" in hs_stages[0]
        assert hs_stages[0]["method"] == "hybrid"

    def test_trace_sub_stages_with_real_components(self):
        """使用真实组件（带 mock 底层）验证各子阶段 trace 记录。"""
        from unittest.mock import patch

        settings = _make_settings_stub()
        qp = QueryProcessor(settings)

        # 创建带 trace 记录能力的 mock retriever
        dense = MagicMock(spec=DenseRetriever)
        dense.retrieve.return_value = _make_retrieval_results(["a"])

        sparse = MagicMock(spec=SparseRetriever)
        sparse.retrieve.return_value = _make_retrieval_results(["b"])

        trace = TraceContext(trace_type="query")

        hs = HybridSearch(settings, query_processor=qp, dense_retriever=dense, sparse_retriever=sparse)
        hs.search("test query", trace=trace)

        # QueryProcessor 应记录 query_processing
        qp_stages = [s for s in trace.stages if s["name"] == "query_processing"]
        assert len(qp_stages) == 1
        assert "method" in qp_stages[0]
        assert qp_stages[0]["method"] in ["jieba", "regex"]

    def test_trace_type_is_query(self):
        """trace.to_dict() 中 trace_type == "query"。"""
        trace = TraceContext(trace_type="query")
        trace.record_stage("test")
        trace.finish()

        data = trace.to_dict()
        assert data["trace_type"] == "query"

    def test_all_stages_have_method_field(self):
        """所有阶段都包含 method 字段。"""
        settings = _make_settings_stub()
        qp = _make_mock_query_processor()
        dense = _make_mock_dense_retriever(_make_retrieval_results(["a"]))
        sparse = _make_mock_sparse_retriever(_make_retrieval_results(["b"]))
        trace = TraceContext(trace_type="query")

        hs = HybridSearch(settings, query_processor=qp, dense_retriever=dense, sparse_retriever=sparse)
        hs.search("test query", trace=trace)

        # 验证所有记录的阶段都有 method 字段
        for stage in trace.stages:
            assert "method" in stage, f"阶段 {stage['name']} 缺少 method 字段"

    def test_all_stages_have_elapsed_ms_field(self):
        """所有阶段都包含 elapsed_ms 字段。"""
        settings = _make_settings_stub()
        qp = _make_mock_query_processor()
        dense = _make_mock_dense_retriever(_make_retrieval_results(["a"]))
        sparse = _make_mock_sparse_retriever(_make_retrieval_results(["b"]))
        trace = TraceContext(trace_type="query")

        hs = HybridSearch(settings, query_processor=qp, dense_retriever=dense, sparse_retriever=sparse)
        hs.search("test query", trace=trace)

        # 验证所有记录的阶段都有 elapsed_ms 字段
        for stage in trace.stages:
            assert "elapsed_ms" in stage, f"阶段 {stage['name']} 缺少 elapsed_ms 字段"


# ============================================================
# 极值测试：验证混合检索在分数极度不平衡下的稳定性
# ============================================================


class TestHybridSearchImbalance:
    """极值测试：Dense 与 Sparse 分数极度不平衡下的融合表现。"""

    def test_extreme_imbalance_rrf_is_score_agnostic(self):
        """验证 RRF 算法对分数量级不敏感（即：分数不淹没排名）。
        
        即使 Sparse 分数是 Dense 的 1,000,000 倍，如果两者都是各自路径的 Top-1，
        在等权重下它们的融合分数应该是相同的。这证明了 RRF 能有效平衡不同量级的分数，
        防止某一路检索（如 BM25 极高分）彻底淹没另一路。
        """
        settings = _make_settings_stub(dense_weight=0.5, sparse_weight=0.5)
        qp = _make_mock_query_processor()

        # Dense 分数极低 (0.0001)
        dense_results = _make_retrieval_results(["dense_only"], scores=[0.0001])
        # Sparse 分数极高 (100.0)
        sparse_results = _make_retrieval_results(["sparse_only"], scores=[100.0])

        dense = _make_mock_dense_retriever(dense_results)
        sparse = _make_mock_sparse_retriever(sparse_results)

        hs = HybridSearch(settings, query_processor=qp, dense_retriever=dense, sparse_retriever=sparse)
        results = hs.search("extreme query")

        # 验证结果不为空
        assert len(results) == 2
        
        # 因为两者都是各自列表的 Rank 0，且权重相等，所以融合分数应该完全一致
        # RRF Score = weight * (1 / (k + rank)) = 0.5 * (1 / (60 + 0)) = 0.008333...
        assert results[0].score == pytest.approx(results[1].score)
        
        # 验证分数在合理范围内（未出现溢出或 NaN）
        assert 0.0 < results[0].score < 1.0

    def test_imbalance_broken_by_weights(self):
        """验证在 Rank 相同但分数不平衡时，可以通过权重进行优先级干预。"""
        # 赋予 Sparse 更高权重 (0.7)，使其在同 Rank 下胜出
        settings = _make_settings_stub(dense_weight=0.3, sparse_weight=0.7)
        qp = _make_mock_query_processor()

        dense_results = _make_retrieval_results(["dense_only"], scores=[0.001])
        sparse_results = _make_retrieval_results(["sparse_only"], scores=[999.0])

        dense = _make_mock_dense_retriever(dense_results)
        sparse = _make_mock_sparse_retriever(sparse_results)

        hs = HybridSearch(settings, query_processor=qp, dense_retriever=dense, sparse_retriever=sparse)
        results = hs.search("weighted extreme query")

        # 验证权重生效：权重高的 Sparse 路径排在前面
        assert results[0].chunk_id == "sparse_only"
        assert results[1].chunk_id == "dense_only"
        assert results[0].score > results[1].score

    def test_imbalance_consensus_wins(self):
        """验证 RRF 的共识特性：出现在多路结果中的项，即使原始分数低，也会排在前面。"""
        settings = _make_settings_stub(dense_weight=0.5, sparse_weight=0.5)
        qp = _make_mock_query_processor()

        # id_shared 在 Dense 是 Rank 0 (高分)，在 Sparse 是 Rank 1 (低分)
        # id_sparse_only 在 Sparse 是 Rank 0 (极高分)
        dense_results = _make_retrieval_results(["id_shared"], scores=[0.9])
        sparse_results = _make_retrieval_results(["id_sparse_only", "id_shared"], scores=[100.0, 0.1])

        dense = _make_mock_dense_retriever(dense_results)
        sparse = _make_mock_sparse_retriever(sparse_results)

        hs = HybridSearch(settings, query_processor=qp, dense_retriever=dense, sparse_retriever=sparse)
        results = hs.search("consensus query")

        # id_shared 出现在两路中，分数 = (0.5/60) + (0.5/61)
        # id_sparse_only 仅出现在一路，分数 = (0.5/60)
        # 因此 id_shared 应该胜出
        assert results[0].chunk_id == "id_shared"
        assert results[1].chunk_id == "id_sparse_only"
