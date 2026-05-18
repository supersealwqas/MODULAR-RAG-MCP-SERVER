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


class TestFuseOverlappingIds:
    """完全重叠 ID 的融合测试。"""

    def test_two_rankings_same_ids_different_order(self):
        """两个排名包含相同 ID 但顺序不同时，RRF 分数正确。"""
        fusion = Fusion(k=60)

        r1 = _make_results(["a", "b", "c"])
        r2 = _make_results(["c", "b", "a"])

        results = fusion.fuse([r1, r2])

        # a: rank0+rank2 → 1/(60+0) + 1/(60+2) = 1/60 + 1/62
        # b: rank1+rank1 → 1/(60+1) + 1/(60+1) = 2/61
        # c: rank2+rank0 → 1/(60+2) + 1/(60+0) = 1/62 + 1/60
        # a 和 c 的分数应相同（对称）
        assert len(results) == 3
        scores = {r.chunk_id: r.score for r in results}
        assert scores["a"] == pytest.approx(scores["c"])
        # RRF 凸性：极端排名的组合分数略高于中间排名
        assert scores["a"] > scores["b"]

    def test_top_k_zero_returns_empty(self):
        """top_k=0 返回空列表。"""
        fusion = Fusion(k=60)
        r1 = _make_results(["a", "b", "c"])

        results = fusion.fuse([r1], top_k=0)
        assert results == []


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


# ============================================================
# 权重测试
# ============================================================


class TestFuseWeights:
    """加权 RRF 融合测试。"""

    def test_init_weights(self):
        """初始化时应能传入权重列表。"""
        fusion = Fusion(k=60, weights=[0.7, 0.3])
        assert fusion.weights == [0.7, 0.3]

    def test_init_no_weights(self):
        """不传权重时 weights 应为 None。"""
        fusion = Fusion(k=60)
        assert fusion.weights is None

    def test_weights_affect_scores(self):
        """权重应影响 RRF 分数。"""
        fusion_equal = Fusion(k=60)  # 等权
        fusion_weighted = Fusion(k=60, weights=[0.8, 0.2])

        r1 = _make_results(["a", "b"])
        r2 = _make_results(["b", "a"])

        results_equal = fusion_equal.fuse([r1, r2])
        results_weighted = fusion_weighted.fuse([r1, r2])

        # a: 等权时 1/60 + 1/61, 加权时 0.8/60 + 0.2/61
        # 加权后 a 的分数应更低（因为第二个列表权重降低，而 a 在第二列表排名靠后）
        score_equal = next(r.score for r in results_equal if r.chunk_id == "a")
        score_weighted = next(r.score for r in results_weighted if r.chunk_id == "a")
        assert score_weighted != score_equal

    def test_weights_order_matters(self):
        """权重顺序应与排名列表顺序对应。"""
        # 第一个列表权重高
        fusion_1 = Fusion(k=60, weights=[0.9, 0.1])
        # 第二个列表权重高
        fusion_2 = Fusion(k=60, weights=[0.1, 0.9])

        r1 = _make_results(["a", "b"])  # a 排第一
        r2 = _make_results(["b", "a"])  # b 排第一

        results_1 = fusion_1.fuse([r1, r2])
        results_2 = fusion_2.fuse([r1, r2])

        # fusion_1: 第一个列表权重高，a 在第一个列表排第一，所以 a 应该排前面
        assert results_1[0].chunk_id == "a"
        # fusion_2: 第二个列表权重高，b 在第二个列表排第一，所以 b 应该排前面
        assert results_2[0].chunk_id == "b"

    def test_dynamic_weights_override(self):
        """调用 fuse 时传入的权重应覆盖初始化权重。"""
        fusion = Fusion(k=60, weights=[0.9, 0.1])

        r1 = _make_results(["a", "b"])
        r2 = _make_results(["b", "a"])

        # 初始化权重: 第一个列表 0.9
        results_init = fusion.fuse([r1, r2])

        # 动态权重: 反转权重
        results_dynamic = fusion.fuse([r1, r2], weights=[0.1, 0.9])

        # 结果顺序应该不同
        assert results_init[0].chunk_id != results_dynamic[0].chunk_id

    def test_single_ranking_with_weight(self):
        """单个排名列表时权重应生效。"""
        fusion = Fusion(k=60, weights=[0.5])
        r1 = _make_results(["a", "b"])

        results = fusion.fuse([r1])

        # 单列表时，权重 0.5 应乘以 RRF 分数
        assert results[0].score == pytest.approx(0.5 * 1.0 / 60)
        assert results[1].score == pytest.approx(0.5 * 1.0 / 61)

    def test_mixed_empty_with_weights(self):
        """混合空列表时权重应正确映射。"""
        fusion = Fusion(k=60, weights=[0.7, 0.3])

        r1 = _make_results(["a", "b"])
        r2 = []  # 空列表

        results = fusion.fuse([r1, r2])

        # r2 为空，只有 r1 参与融合，权重应为 0.7
        assert len(results) == 2
        assert results[0].score == pytest.approx(0.7 * 1.0 / 60)

    def test_three_rankings_with_weights(self):
        """三个排名列表的加权融合。"""
        fusion = Fusion(k=60, weights=[0.5, 0.3, 0.2])

        r1 = _make_results(["a", "b", "c"])
        r2 = _make_results(["b", "c", "a"])
        r3 = _make_results(["c", "a", "b"])

        results = fusion.fuse([r1, r2, r3])

        # 计算预期分数
        # a: 0.5/60 + 0.3/62 + 0.2/62
        # b: 0.5/61 + 0.3/60 + 0.2/62
        # c: 0.5/62 + 0.3/61 + 0.2/60
        assert len(results) == 3
        # 分数应各不相同（因为权重不对称）
        scores = [r.score for r in results]
        assert len(set(scores)) == 3

    def test_weights_with_trace(self):
        """权重信息应记录在 Trace 中。"""
        fusion = Fusion(k=60, weights=[0.7, 0.3])
        r1 = _make_results(["a", "b"])
        r2 = _make_results(["b", "a"])
        trace = TraceContext()

        fusion.fuse([r1, r2], trace=trace)

        stages = [s for s in trace.stages if s["name"] == "fusion"]
        assert len(stages) == 1
        assert stages[0]["weights"] == [0.7, 0.3]
