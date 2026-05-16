"""D4 Fusion RRF 单元测试。

覆盖：
- 基本 RRF 融合（两个排名列表）
- 多个排名列表融合
- top_k 截断
- k 参数可配置
- 空输入处理
- 确定性输出（deterministic）
- Trace 记录
验收标准：对构造的排名输入输出 deterministic；k 参数可配置。
"""

from __future__ import annotations

from typing import List

import pytest

from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult
from src.core.query_engine.fusion import Fusion


# ============================================================
# 测试辅助函数
# ============================================================


def _make_results(ids: List[str], scores: List[float] = None) -> List[RetrievalResult]:
    """创建测试用 RetrievalResult 列表。"""
    if scores is None:
        scores = [1.0 - i * 0.1 for i in range(len(ids))]
    return [
        RetrievalResult(
            chunk_id=cid,
            score=score,
            text=f"text_{cid}",
            metadata={"source": f"doc_{cid}"},
        )
        for cid, score in zip(ids, scores)
    ]


# ============================================================
# 基本 RRF 融合测试
# ============================================================


class TestFuseBasic:
    """基本 RRF 融合测试。"""

    def test_two_rankings_fusion(self):
        """两个排名列表能正确融合。"""
        fusion = Fusion(k=60)

        dense_results = _make_results(["a", "b", "c"])
        sparse_results = _make_results(["b", "a", "d"])

        results = fusion.fuse([dense_results, sparse_results])

        # a: 1/(60+0) + 1/(60+1) = 0.01667 + 0.01639 = 0.03306
        # b: 1/(60+1) + 1/(60+0) = 0.01639 + 0.01667 = 0.03306
        # c: 1/(60+2) = 0.01613
        # d: 1/(60+2) = 0.01613
        assert len(results) == 4
        # a 和 b 分数相同，排在前面
        assert results[0].chunk_id in ["a", "b"]
        assert results[1].chunk_id in ["a", "b"]
        assert results[2].chunk_id in ["c", "d"]
        assert results[3].chunk_id in ["c", "d"]

    def test_single_ranking(self):
        """单个排名列表直接返回。"""
        fusion = Fusion(k=60)
        results_list = _make_results(["a", "b", "c"])

        results = fusion.fuse([results_list])

        assert len(results) == 3
        # 分数应为 RRF 分数
        assert results[0].score == pytest.approx(1.0 / 60)
        assert results[1].score == pytest.approx(1.0 / 61)

    def test_preserves_text_and_metadata(self):
        """融合后保留原始 text 和 metadata。"""
        fusion = Fusion(k=60)
        results_list = _make_results(["a"])

        results = fusion.fuse([results_list])

        assert results[0].text == "text_a"
        assert results[0].metadata["source"] == "doc_a"


# ============================================================
# 多排名列表融合测试
# ============================================================


class TestFuseMultipleRankings:
    """多个排名列表融合测试。"""

    def test_three_rankings(self):
        """三个排名列表能正确融合。"""
        fusion = Fusion(k=60)

        r1 = _make_results(["a", "b", "c"])
        r2 = _make_results(["b", "c", "a"])
        r3 = _make_results(["c", "a", "b"])

        results = fusion.fuse([r1, r2, r3])

        # 每个 chunk 出现 3 次，排名分别是 0,1,2
        # 所有 chunk 的 RRF 分数相同
        assert len(results) == 3
        scores = [r.score for r in results]
        # 所有分数应该非常接近
        assert max(scores) - min(scores) < 0.001

    def test_different_lengths(self):
        """不同长度的排名列表能正确融合。"""
        fusion = Fusion(k=60)

        r1 = _make_results(["a", "b", "c", "d"])
        r2 = _make_results(["a", "b"])  # 较短

        results = fusion.fuse([r1, r2])

        # a: 1/60 + 1/60 = 2/60
        # b: 1/61 + 1/61 = 2/61
        # c: 1/62
        # d: 1/63
        assert len(results) == 4
        assert results[0].chunk_id == "a"
        assert results[1].chunk_id == "b"


# ============================================================
# top_k 截断测试
# ============================================================


class TestFuseTopK:
    """top_k 截断测试。"""

    def test_top_k_limits_results(self):
        """top_k 参数限制返回数量。"""
        fusion = Fusion(k=60)
        results_list = _make_results(["a", "b", "c", "d", "e"])

        results = fusion.fuse([results_list], top_k=3)

        assert len(results) == 3

    def test_top_k_larger_than_results(self):
        """top_k 大于结果数时返回全部。"""
        fusion = Fusion(k=60)
        results_list = _make_results(["a", "b"])

        results = fusion.fuse([results_list], top_k=5)

        assert len(results) == 2

    def test_no_top_k_returns_all(self):
        """不指定 top_k 时返回全部结果。"""
        fusion = Fusion(k=60)
        results_list = _make_results(["a", "b", "c"])

        results = fusion.fuse([results_list])

        assert len(results) == 3


# ============================================================
# k 参数可配置测试
# ============================================================


class TestFuseKParameter:
    """k 参数可配置测试。"""

    def test_custom_k(self):
        """自定义 k 参数影响分数。"""
        fusion_small_k = Fusion(k=10)
        fusion_large_k = Fusion(k=100)

        results_list = _make_results(["a", "b"])

        results_small = fusion_small_k.fuse([results_list])
        results_large = fusion_large_k.fuse([results_list])

        # k 越小，排名第一和第二的分数差距越大
        gap_small = results_small[0].score - results_small[1].score
        gap_large = results_large[0].score - results_large[1].score
        assert gap_small > gap_large

    def test_default_k_is_60(self):
        """默认 k 值为 60。"""
        fusion = Fusion()
        assert fusion.k == 60


# ============================================================
# 空输入处理测试
# ============================================================


class TestFuseEdgeCases:
    """空输入处理测试。"""

    def test_empty_rankings(self):
        """空排名列表返回空结果。"""
        fusion = Fusion()
        results = fusion.fuse([])
        assert results == []

    def test_all_empty_rankings(self):
        """所有排名列表都为空时返回空结果。"""
        fusion = Fusion()
        results = fusion.fuse([[], [], []])
        assert results == []

    def test_mixed_empty_rankings(self):
        """混合空和非空排名列表时忽略空列表。"""
        fusion = Fusion(k=60)
        r1 = _make_results(["a", "b"])
        r2 = []

        results = fusion.fuse([r1, r2])

        assert len(results) == 2
        assert results[0].chunk_id == "a"


# ============================================================
# 确定性输出测试
# ============================================================


class TestFuseDeterministic:
    """确定性输出测试。"""

    def test_same_input_same_output(self):
        """相同输入产生相同输出。"""
        fusion = Fusion(k=60)

        r1 = _make_results(["a", "b", "c"])
        r2 = _make_results(["b", "a", "d"])

        results1 = fusion.fuse([r1, r2])
        results2 = fusion.fuse([r1, r2])

        assert len(results1) == len(results2)
        for r1_item, r2_item in zip(results1, results2):
            assert r1_item.chunk_id == r2_item.chunk_id
            assert r1_item.score == pytest.approx(r2_item.score)

    def test_rrf_scores_symmetric(self):
        """RRF 分数对相同排名是对称的。"""
        fusion = Fusion(k=60)

        r1 = _make_results(["a", "b"])
        r2 = _make_results(["b", "a"])

        results = fusion.fuse([r1, r2])

        # a: 1/60 + 1/61, b: 1/61 + 1/60 → 分数相同
        assert len(results) == 2
        assert results[0].score == pytest.approx(results[1].score)


# ============================================================
# Trace 记录测试
# ============================================================


class TestFuseTraceRecording:
    """Trace 记录测试。"""

    def test_trace_records_fusion(self):
        """传入 trace 时记录 fusion 阶段。"""
        fusion = Fusion(k=60)
        r1 = _make_results(["a", "b"])
        trace = TraceContext()

        fusion.fuse([r1], trace=trace)

        stages = [s for s in trace.stages if s["name"] == "fusion"]
        assert len(stages) == 1
        assert stages[0]["method"] == "rrf"
        assert stages[0]["k"] == 60
        assert stages[0]["input_rankings"] == 1
        assert stages[0]["output_count"] == 2

    def test_no_trace_no_error(self):
        """不传 trace 时不报错。"""
        fusion = Fusion(k=60)
        r1 = _make_results(["a", "b"])

        results = fusion.fuse([r1])
        assert len(results) == 2


# ============================================================
# RetrievalResult 序列化测试
# ============================================================


class TestFuseSerialization:
    """RetrievalResult 序列化测试。"""

    def test_results_serializable(self):
        """融合后的结果支持序列化。"""
        fusion = Fusion(k=60)
        r1 = _make_results(["a", "b"])

        results = fusion.fuse([r1])

        for r in results:
            d = r.to_dict()
            assert "chunk_id" in d
            assert "score" in d
            assert "text" in d
            assert "metadata" in d

            restored = RetrievalResult.from_dict(d)
            assert restored.chunk_id == r.chunk_id
            assert restored.score == pytest.approx(r.score)
