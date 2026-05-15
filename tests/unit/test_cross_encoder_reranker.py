"""测试 CrossEncoderReranker 实现。

使用 mock scorer 隔离测试，不加载真实 Cross-Encoder 模型。
"""

import pytest
from typing import List, Tuple
from unittest.mock import MagicMock

from src.libs.reranker.base_reranker import (
    BaseReranker,
    Candidate,
    RankedCandidate,
)
from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from src.libs.reranker.reranker_factory import RerankerFactory


def make_scorer(scores: List[float]):
    """构造预设分数的 mock scorer。

    参数:
        scores: 预设的分数列表

    返回:
        scorer 函数
    """

    def scorer(pairs: List[Tuple[str, str]]) -> List[float]:
        """返回预设分数。"""
        return scores

    return scorer


@pytest.mark.unit
class TestCrossEncoderRerankerBasic:
    """测试 CrossEncoderReranker 基本功能。"""

    def test_is_subclass_of_base(self):
        """CrossEncoderReranker 应继承 BaseReranker。"""
        reranker = CrossEncoderReranker(scorer=make_scorer([0.5]))
        assert isinstance(reranker, BaseReranker)

    def test_factory_creates_cross_encoder(self):
        """backend=cross_encoder 时 RerankerFactory 应创建 CrossEncoderReranker。"""
        reranker = RerankerFactory.create(
            provider="cross_encoder",
            scorer=make_scorer([0.5]),
        )
        assert isinstance(reranker, CrossEncoderReranker)

    def test_factory_case_insensitive(self):
        """工厂应不区分大小写。"""
        reranker = RerankerFactory.create(
            provider="Cross_Encoder",
            scorer=make_scorer([0.5]),
        )
        assert isinstance(reranker, CrossEncoderReranker)

    def test_cross_encoder_in_list_providers(self):
        """cross_encoder 应在可用提供者列表中。"""
        providers = RerankerFactory.list_providers()
        assert "cross_encoder" in providers

    def test_default_model(self):
        """默认模型名称应正确设置。"""
        reranker = CrossEncoderReranker()
        assert reranker._model_name == "models/bge-reranker-large"

    def test_custom_model(self):
        """自定义模型名称应正确传递。"""
        reranker = CrossEncoderReranker(model="my-model")
        assert reranker._model_name == "my-model"


@pytest.mark.unit
class TestCrossEncoderRerankerRerank:
    """测试 CrossEncoderReranker 重排序逻辑。"""

    def _make_candidates(self, ids: List[str]) -> List[Candidate]:
        """构造测试候选文档。"""
        return [
            Candidate(id=cid, text=f"文档 {cid} 的内容", score=0.5)
            for cid in ids
        ]

    def test_rerank_basic(self):
        """基本重排序：scorer 返回有效分数。"""
        scorer = make_scorer([0.3, 0.9, 0.6])
        reranker = CrossEncoderReranker(scorer=scorer)
        candidates = self._make_candidates(["doc_1", "doc_2", "doc_3"])

        results = reranker.rerank("测试查询", candidates)

        assert len(results) == 3
        assert results[0].id == "doc_2"  # 最高分 0.9
        assert results[1].id == "doc_3"  # 0.6
        assert results[2].id == "doc_1"  # 0.3
        assert results[0].rerank_score == 0.9
        assert results[1].rerank_score == 0.6
        assert results[2].rerank_score == 0.3

    def test_rerank_preserves_original_score(self):
        """重排序应保留原始分数。"""
        scorer = make_scorer([0.8, 0.2])
        reranker = CrossEncoderReranker(scorer=scorer)
        candidates = [
            Candidate(id="doc_1", text="内容1", score=0.7),
            Candidate(id="doc_2", text="内容2", score=0.5),
        ]

        results = reranker.rerank("查询", candidates)

        id_to_result = {r.id: r for r in results}
        assert id_to_result["doc_1"].original_score == 0.7
        assert id_to_result["doc_2"].original_score == 0.5

    def test_rerank_with_top_k(self):
        """top_k 应限制返回数量。"""
        scorer = make_scorer([0.1, 0.9, 0.5])
        reranker = CrossEncoderReranker(scorer=scorer)
        candidates = self._make_candidates(["doc_1", "doc_2", "doc_3"])

        results = reranker.rerank("查询", candidates, top_k=2)

        assert len(results) == 2
        assert results[0].id == "doc_2"
        assert results[1].id == "doc_3"

    def test_rerank_empty_candidates(self):
        """空候选列表应返回空结果。"""
        reranker = CrossEncoderReranker(scorer=make_scorer([]))
        results = reranker.rerank("查询", [])
        assert results == []

    def test_rerank_single_candidate(self):
        """单个候选应正常返回。"""
        scorer = make_scorer([0.75])
        reranker = CrossEncoderReranker(scorer=scorer)
        candidates = self._make_candidates(["doc_1"])

        results = reranker.rerank("查询", candidates)

        assert len(results) == 1
        assert results[0].id == "doc_1"
        assert results[0].rerank_score == 0.75

    def test_rerank_records_elapsed_ms(self):
        """结果 metadata 应包含耗时信息。"""
        scorer = make_scorer([0.5, 0.8])
        reranker = CrossEncoderReranker(scorer=scorer)
        candidates = self._make_candidates(["doc_1", "doc_2"])

        results = reranker.rerank("查询", candidates)

        for r in results:
            assert "rerank_elapsed_ms" in r.metadata
            assert r.metadata["rerank_elapsed_ms"] >= 0


@pytest.mark.unit
class TestCrossEncoderRerankerErrorHandling:
    """测试 CrossEncoderReranker 错误处理。"""

    def test_scorer_failure_raises_runtime_error(self):
        """scorer 调用失败时应抛出 RuntimeError。"""

        def bad_scorer(pairs):
            """模拟打分失败的 scorer。"""
            raise ValueError("打分失败")

        reranker = CrossEncoderReranker(scorer=bad_scorer)
        candidates = [Candidate(id="doc_1", text="内容", score=0.5)]

        with pytest.raises(RuntimeError, match="Cross-Encoder 打分失败"):
            reranker.rerank("查询", candidates)

    def test_timeout_raises_runtime_error(self):
        """超时时应抛出 RuntimeError。"""
        import time

        def slow_scorer(pairs):
            """模拟超时的 scorer。"""
            time.sleep(0.1)
            return [0.5] * len(pairs)

        reranker = CrossEncoderReranker(scorer=slow_scorer, timeout=0.01)
        candidates = [Candidate(id="doc_1", text="内容", score=0.5)]

        with pytest.raises(RuntimeError, match="超时"):
            reranker.rerank("查询", candidates)

    def test_scorer_via_kwargs(self):
        """应支持通过 kwargs 临时覆盖 scorer。"""
        default_scorer = make_scorer([0.1])
        override_scorer = make_scorer([0.9])

        reranker = CrossEncoderReranker(scorer=default_scorer)
        candidates = self._make_candidates(["doc_1"])

        results = reranker.rerank("查询", candidates, scorer=override_scorer)
        assert results[0].rerank_score == 0.9

    def _make_candidates(self, ids):
        """构造测试候选文档。"""
        return [Candidate(id=cid, text=f"内容{cid}", score=0.5) for cid in ids]


@pytest.mark.unit
class TestCrossEncoderRerankerIntegration:
    """测试 CrossEncoderReranker 与其他 Reranker 的一致性。"""

    def test_rerank_order_descending(self):
        """结果应按 rerank_score 降序排列。"""
        scores = [0.2, 0.8, 0.5, 0.1, 0.9]
        scorer = make_scorer(scores)
        reranker = CrossEncoderReranker(scorer=scorer)
        candidates = [
            Candidate(id=f"doc_{i}", text=f"内容{i}", score=0.5)
            for i in range(5)
        ]

        results = reranker.rerank("查询", candidates)

        rerank_scores = [r.rerank_score for r in results]
        assert rerank_scores == sorted(rerank_scores, reverse=True)

    def test_rerank_all_candidates_returned(self):
        """不指定 top_k 时应返回所有候选。"""
        scorer = make_scorer([0.1, 0.2, 0.3])
        reranker = CrossEncoderReranker(scorer=scorer)
        candidates = [
            Candidate(id=f"doc_{i}", text=f"内容{i}", score=0.5)
            for i in range(3)
        ]

        results = reranker.rerank("查询", candidates)

        assert len(results) == 3
        result_ids = {r.id for r in results}
        assert result_ids == {"doc_0", "doc_1", "doc_2"}
