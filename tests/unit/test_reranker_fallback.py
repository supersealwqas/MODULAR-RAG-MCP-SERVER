"""D6 Reranker Core 层编排 + Fallback 单元测试。

覆盖：
- 基本重排序流程（RetrievalResult → Candidate → backend → RankedCandidate → RetrievalResult）
- 后端异常回退到原始排序
- fallback 标记正确性
- top_k 参数传递
- Trace 记录
- 空输入处理
验收标准：模拟后端异常时不影响最终返回，且标记 fallback=true。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from src.core.query_engine.reranker import Reranker
from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult
from src.libs.reranker.base_reranker import (
    BaseReranker,
    Candidate,
    NoneReranker,
    RankedCandidate,
)


# ============================================================
# 测试辅助函数
# ============================================================


def _make_settings_stub(**kwargs) -> Settings:
    """创建测试用 Settings stub。"""
    rerank_mock = MagicMock()
    rerank_mock.enabled = kwargs.get("rerank_enabled", True)
    rerank_mock.provider = kwargs.get("rerank_provider", "none")
    rerank_mock.top_k = kwargs.get("rerank_top_k", 5)
    return Settings(
        llm=MagicMock(),
        vision_llm=MagicMock(),
        ollama=MagicMock(),
        embedding=MagicMock(),
        splitter=MagicMock(),
        vector_store=MagicMock(),
        retrieval=MagicMock(),
        rerank=rerank_mock,
        evaluation=MagicMock(),
        observability=MagicMock(),
    )


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


def _make_mock_backend(rerank_fn=None) -> MagicMock:
    """创建 Mock Reranker 后端。"""
    mock = MagicMock(spec=BaseReranker)
    if rerank_fn is not None:
        mock.rerank.side_effect = rerank_fn
    return mock


def _default_rerank_fn(
    query: str,
    candidates: List[Candidate],
    top_k: Optional[int] = None,
    **kwargs,
) -> List[RankedCandidate]:
    """默认的 rerank 实现：保持原始顺序，转换为 RankedCandidate。"""
    results = []
    for c in candidates:
        results.append(RankedCandidate(
            id=c.id,
            text=c.text,
            rerank_score=c.score,
            original_score=c.score,
            metadata=c.metadata,
        ))
    if top_k is not None:
        results = results[:top_k]
    return results


def _reverse_rerank_fn(
    query: str,
    candidates: List[Candidate],
    top_k: Optional[int] = None,
    **kwargs,
) -> List[RankedCandidate]:
    """反转顺序的 rerank 实现。"""
    results = []
    for c in reversed(candidates):
        results.append(RankedCandidate(
            id=c.id,
            text=c.text,
            rerank_score=c.score * 2,  # 修改分数以区分
            original_score=c.score,
            metadata=c.metadata,
        ))
    if top_k is not None:
        results = results[:top_k]
    return results


# ============================================================
# 基本重排序流程测试
# ============================================================


class TestRerankerBasic:
    """基本重排序流程测试。"""

    def test_rerank_returns_results(self):
        """rerank 返回包含 results、fallback、elapsed_ms 的字典。"""
        settings = _make_settings_stub()
        backend = _make_mock_backend(_default_rerank_fn)
        reranker = Reranker(settings, backend=backend)

        candidates = _make_retrieval_results(["a", "b", "c"])
        result = reranker.rerank("test query", candidates)

        assert "results" in result
        assert "fallback" in result
        assert "elapsed_ms" in result
        assert len(result["results"]) == 3

    def test_rerank_converts_correctly(self):
        """rerank 正确完成 RetrievalResult → Candidate → RankedCandidate → RetrievalResult 转换。"""
        settings = _make_settings_stub()
        backend = _make_mock_backend(_default_rerank_fn)
        reranker = Reranker(settings, backend=backend)

        candidates = _make_retrieval_results(["a", "b"])
        result = reranker.rerank("test query", candidates)

        results = result["results"]
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].chunk_id == "a"
        assert results[1].chunk_id == "b"

    def test_rerank_preserves_metadata(self):
        """rerank 保留原始 metadata。"""
        settings = _make_settings_stub()
        backend = _make_mock_backend(_default_rerank_fn)
        reranker = Reranker(settings, backend=backend)

        candidates = _make_retrieval_results(["a"])
        result = reranker.rerank("test query", candidates)

        assert result["results"][0].metadata["source"] == "doc_a"
        assert result["results"][0].metadata["collection"] == "test"

    def test_rerank_uses_rerank_score(self):
        """rerank 使用后端的 rerank_score 作为最终 score。"""
        settings = _make_settings_stub()
        backend = _make_mock_backend(_reverse_rerank_fn)
        reranker = Reranker(settings, backend=backend)

        candidates = _make_retrieval_results(["a", "b"], scores=[0.9, 0.8])
        result = reranker.rerank("test query", candidates)

        # reverse_rerank_fn 将分数乘以 2
        assert result["results"][0].score == pytest.approx(0.8 * 2)  # 原来排第二的 b
        assert result["results"][1].score == pytest.approx(0.9 * 2)  # 原来排第一的 a

    def test_rerank_fallback_false_on_success(self):
        """后端正常时 fallback=False。"""
        settings = _make_settings_stub()
        backend = _make_mock_backend(_default_rerank_fn)
        reranker = Reranker(settings, backend=backend)

        candidates = _make_retrieval_results(["a", "b"])
        result = reranker.rerank("test query", candidates)

        assert result["fallback"] is False


# ============================================================
# Fallback 测试
# ============================================================


class TestRerankerFallback:
    """后端异常回退测试。"""

    def test_backend_exception_falls_back(self):
        """后端抛出异常时回退到原始排序。"""
        settings = _make_settings_stub()
        backend = _make_mock_backend()
        backend.rerank.side_effect = RuntimeError("backend failed")
        reranker = Reranker(settings, backend=backend)

        candidates = _make_retrieval_results(["a", "b", "c"])
        result = reranker.rerank("test query", candidates)

        # 应返回原始顺序
        assert len(result["results"]) == 3
        assert result["results"][0].chunk_id == "a"
        assert result["results"][1].chunk_id == "b"
        assert result["results"][2].chunk_id == "c"

    def test_backend_exception_sets_fallback_true(self):
        """后端异常时标记 fallback=True。"""
        settings = _make_settings_stub()
        backend = _make_mock_backend()
        backend.rerank.side_effect = RuntimeError("backend failed")
        reranker = Reranker(settings, backend=backend)

        candidates = _make_retrieval_results(["a", "b"])
        result = reranker.rerank("test query", candidates)

        assert result["fallback"] is True

    def test_backend_exception_preserves_scores(self):
        """后端异常时保留原始分数。"""
        settings = _make_settings_stub()
        backend = _make_mock_backend()
        backend.rerank.side_effect = RuntimeError("backend failed")
        reranker = Reranker(settings, backend=backend)

        candidates = _make_retrieval_results(["a", "b"], scores=[0.9, 0.7])
        result = reranker.rerank("test query", candidates)

        assert result["results"][0].score == pytest.approx(0.9)
        assert result["results"][1].score == pytest.approx(0.7)

    def test_backend_timeout_falls_back(self):
        """后端超时时回退到原始排序。"""
        settings = _make_settings_stub()
        backend = _make_mock_backend()
        backend.rerank.side_effect = TimeoutError("rerank timeout")
        reranker = Reranker(settings, backend=backend)

        candidates = _make_retrieval_results(["a", "b"])
        result = reranker.rerank("test query", candidates)

        assert result["fallback"] is True
        assert len(result["results"]) == 2

    def test_fallback_respects_top_k(self):
        """回退时仍受 top_k 限制。"""
        settings = _make_settings_stub(rerank_top_k=2)
        backend = _make_mock_backend()
        backend.rerank.side_effect = RuntimeError("backend failed")
        reranker = Reranker(settings, backend=backend)

        candidates = _make_retrieval_results(["a", "b", "c", "d"])
        result = reranker.rerank("test query", candidates, top_k=2)

        assert len(result["results"]) == 2


# ============================================================
# top_k 测试
# ============================================================


class TestRerankerTopK:
    """top_k 参数测试。"""

    def test_default_top_k_from_settings(self):
        """默认 top_k 从 settings.rerank.top_k 读取。"""
        settings = _make_settings_stub(rerank_top_k=3)
        backend = _make_mock_backend(_default_rerank_fn)
        reranker = Reranker(settings, backend=backend)

        candidates = _make_retrieval_results(["a", "b", "c", "d", "e"])
        result = reranker.rerank("test query", candidates)

        assert len(result["results"]) == 3

    def test_explicit_top_k_overrides_settings(self):
        """显式 top_k 参数覆盖 settings 默认值。"""
        settings = _make_settings_stub(rerank_top_k=3)
        backend = _make_mock_backend(_default_rerank_fn)
        reranker = Reranker(settings, backend=backend)

        candidates = _make_retrieval_results(["a", "b", "c", "d", "e"])
        result = reranker.rerank("test query", candidates, top_k=2)

        assert len(result["results"]) == 2

    def test_top_k_passed_to_backend(self):
        """top_k 正确传递给后端。"""
        settings = _make_settings_stub(rerank_top_k=3)
        backend = _make_mock_backend(_default_rerank_fn)
        reranker = Reranker(settings, backend=backend)

        candidates = _make_retrieval_results(["a", "b", "c", "d", "e"])
        reranker.rerank("test query", candidates, top_k=2)

        call_kwargs = backend.rerank.call_args
        assert call_kwargs[1]["top_k"] == 2 or call_kwargs[0][2] == 2


# ============================================================
# 空输入测试
# ============================================================


class TestRerankerEdgeCases:
    """空输入和边界场景测试。"""

    def test_empty_candidates_returns_empty(self):
        """空候选列表返回空结果。"""
        settings = _make_settings_stub()
        backend = _make_mock_backend(_default_rerank_fn)
        reranker = Reranker(settings, backend=backend)

        result = reranker.rerank("test query", [])

        assert result["results"] == []
        assert result["fallback"] is False

    def test_empty_candidates_no_backend_call(self):
        """空候选列表不调用后端。"""
        settings = _make_settings_stub()
        backend = _make_mock_backend(_default_rerank_fn)
        reranker = Reranker(settings, backend=backend)

        reranker.rerank("test query", [])

        backend.rerank.assert_not_called()

    def test_single_candidate(self):
        """单个候选正常处理。"""
        settings = _make_settings_stub()
        backend = _make_mock_backend(_default_rerank_fn)
        reranker = Reranker(settings, backend=backend)

        candidates = _make_retrieval_results(["a"])
        result = reranker.rerank("test query", candidates)

        assert len(result["results"]) == 1
        assert result["results"][0].chunk_id == "a"


# ============================================================
# Trace 记录测试
# ============================================================


class TestRerankerTraceRecording:
    """Trace 记录测试。"""

    def test_trace_records_reranker_stage(self):
        """传入 trace 时记录 reranker 阶段。"""
        settings = _make_settings_stub()
        backend = _make_mock_backend(_default_rerank_fn)
        reranker = Reranker(settings, backend=backend)
        trace = TraceContext()

        candidates = _make_retrieval_results(["a", "b"])
        reranker.rerank("test query", candidates, trace=trace)

        stages = [s for s in trace.stages if s["name"] == "reranker"]
        assert len(stages) == 1
        assert "fallback" in stages[0]
        assert "input_count" in stages[0]
        assert "output_count" in stages[0]
        assert "elapsed_ms" in stages[0]

    def test_trace_records_fallback_on_error(self):
        """后端异常时 trace 记录 fallback=True。"""
        settings = _make_settings_stub()
        backend = _make_mock_backend()
        backend.rerank.side_effect = RuntimeError("backend failed")
        reranker = Reranker(settings, backend=backend)
        trace = TraceContext()

        candidates = _make_retrieval_results(["a", "b"])
        reranker.rerank("test query", candidates, trace=trace)

        stages = [s for s in trace.stages if s["name"] == "reranker"]
        assert stages[0]["fallback"] is True

    def test_no_trace_no_error(self):
        """不传 trace 时不报错。"""
        settings = _make_settings_stub()
        backend = _make_mock_backend(_default_rerank_fn)
        reranker = Reranker(settings, backend=backend)

        candidates = _make_retrieval_results(["a", "b"])
        result = reranker.rerank("test query", candidates)

        assert len(result["results"]) > 0


# ============================================================
# NoneReranker 集成测试
# ============================================================


class TestRerankerFallbackDefaultTopK:
    """回退时使用默认 top_k 测试。"""

    def test_fallback_uses_settings_top_k(self):
        """回退时使用 settings 默认 top_k（不显式传参）。"""
        settings = _make_settings_stub(rerank_top_k=2)
        backend = _make_mock_backend()
        backend.rerank.side_effect = RuntimeError("backend failed")
        reranker = Reranker(settings, backend=backend)

        candidates = _make_retrieval_results(["a", "b", "c", "d"])
        result = reranker.rerank("test query", candidates)

        assert result["fallback"] is True
        assert len(result["results"]) == 2


class TestRerankerWithNoneBackend:
    """使用 NoneReranker 后端的集成测试。"""

    def test_none_reranker_preserves_order(self):
        """NoneReranker 保持原始顺序。"""
        settings = _make_settings_stub()
        reranker = Reranker(settings, backend=NoneReranker())

        candidates = _make_retrieval_results(["a", "b", "c"])
        result = reranker.rerank("test query", candidates)

        assert result["fallback"] is False
        assert [r.chunk_id for r in result["results"]] == ["a", "b", "c"]

    def test_none_reranker_preserves_scores(self):
        """NoneReranker 保持原始分数。"""
        settings = _make_settings_stub()
        reranker = Reranker(settings, backend=NoneReranker())

        candidates = _make_retrieval_results(["a", "b"], scores=[0.9, 0.7])
        result = reranker.rerank("test query", candidates)

        assert result["results"][0].score == pytest.approx(0.9)
        assert result["results"][1].score == pytest.approx(0.7)
