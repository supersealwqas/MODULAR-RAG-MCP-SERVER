"""测试 Reranker 抽象接口、NoneReranker 和工厂。"""

import pytest
from typing import List, Optional

from src.libs.reranker.base_reranker import (
    BaseReranker,
    NoneReranker,
    Candidate,
    RankedCandidate,
)
from src.libs.reranker.reranker_factory import (
    RerankerFactory,
    register_reranker,
    _RERANKER_REGISTRY,
)


# --- Fake Reranker 用于测试 ---

class FakeReranker(BaseReranker):
    """测试用的假 Reranker 实现，按文本长度重排序。"""

    def rerank(
        self,
        query: str,
        candidates: List[Candidate],
        top_k: Optional[int] = None,
        **kwargs,
    ) -> List[RankedCandidate]:
        """按文本长度降序重排序（越长越相关）。"""
        sorted_candidates = sorted(candidates, key=lambda c: len(c.text), reverse=True)
        results = []
        for candidate in sorted_candidates:
            results.append(RankedCandidate(
                id=candidate.id,
                text=candidate.text,
                rerank_score=float(len(candidate.text)),
                original_score=candidate.score,
                metadata=candidate.metadata,
            ))
        if top_k is not None:
            results = results[:top_k]
        return results


# --- 数据类测试 ---

@pytest.mark.unit
class TestCandidate:
    """测试 Candidate 数据类。"""

    def test_create_candidate(self):
        """应能创建包含所有字段的候选文档。"""
        candidate = Candidate(
            id="doc1",
            text="Hello world",
            score=0.9,
            metadata={"source": "test.pdf"},
        )
        assert candidate.id == "doc1"
        assert candidate.text == "Hello world"
        assert candidate.score == 0.9
        assert candidate.metadata["source"] == "test.pdf"

    def test_default_values(self):
        """应有合理的默认值。"""
        candidate = Candidate(id="1", text="test")
        assert candidate.score == 0.0
        assert candidate.metadata == {}


@pytest.mark.unit
class TestRankedCandidate:
    """测试 RankedCandidate 数据类。"""

    def test_create_ranked_candidate(self):
        """应能创建包含所有字段的重排序结果。"""
        ranked = RankedCandidate(
            id="doc1",
            text="Hello world",
            rerank_score=0.95,
            original_score=0.8,
            metadata={"source": "test.pdf"},
        )
        assert ranked.id == "doc1"
        assert ranked.rerank_score == 0.95
        assert ranked.original_score == 0.8


# --- NoneReranker 测试 ---

@pytest.mark.unit
class TestNoneReranker:
    """测试 NoneReranker 默认回退实现。"""

    def test_none_reranker_preserves_order(self):
        """NoneReranker 应保持原始顺序。"""
        reranker = NoneReranker()
        candidates = [
            Candidate(id="1", text="first", score=0.9),
            Candidate(id="2", text="second", score=0.8),
            Candidate(id="3", text="third", score=0.7),
        ]
        results = reranker.rerank(query="test", candidates=candidates)
        assert len(results) == 3
        assert results[0].id == "1"
        assert results[1].id == "2"
        assert results[2].id == "3"

    def test_none_reranker_preserves_scores(self):
        """NoneReranker 应保持原始分数。"""
        reranker = NoneReranker()
        candidates = [
            Candidate(id="1", text="hello", score=0.9),
            Candidate(id="2", text="world", score=0.8),
        ]
        results = reranker.rerank(query="test", candidates=candidates)
        assert results[0].rerank_score == 0.9
        assert results[0].original_score == 0.9
        assert results[1].rerank_score == 0.8

    def test_none_reranker_top_k(self):
        """NoneReranker 应支持 top_k 限制。"""
        reranker = NoneReranker()
        candidates = [
            Candidate(id=str(i), text=f"text{i}", score=float(i) / 10)
            for i in range(5)
        ]
        results = reranker.rerank(query="test", candidates=candidates, top_k=2)
        assert len(results) == 2

    def test_none_reranker_empty_candidates(self):
        """空候选列表应返回空结果。"""
        reranker = NoneReranker()
        results = reranker.rerank(query="test", candidates=[])
        assert results == []


# --- BaseReranker 测试 ---

@pytest.mark.unit
class TestBaseReranker:
    """测试 BaseReranker 抽象类。"""

    def test_cannot_instantiate_abstract(self):
        """BaseReranker 是抽象类，不能直接实例化。"""
        with pytest.raises(TypeError):
            BaseReranker()

    def test_fake_reranker_rerank(self):
        """FakeReranker 应按文本长度重排序。"""
        reranker = FakeReranker()
        candidates = [
            Candidate(id="1", text="hi", score=0.9),
            Candidate(id="2", text="hello world", score=0.8),
            Candidate(id="3", text="hey", score=0.7),
        ]
        results = reranker.rerank(query="test", candidates=candidates)
        assert len(results) == 3
        assert results[0].id == "2"  # "hello world" 最长
        assert results[0].rerank_score == 11.0

    def test_fake_reranker_top_k(self):
        """FakeReranker 应支持 top_k 限制。"""
        reranker = FakeReranker()
        candidates = [
            Candidate(id="1", text="short", score=0.9),
            Candidate(id="2", text="a bit longer", score=0.8),
            Candidate(id="3", text="the longest text here", score=0.7),
        ]
        results = reranker.rerank(query="test", candidates=candidates, top_k=1)
        assert len(results) == 1
        assert results[0].id == "3"


# --- RerankerFactory 测试 ---

@pytest.mark.unit
class TestRerankerFactory:
    """测试 RerankerFactory 的路由逻辑。"""

    def setup_method(self):
        """每个测试前注册 fake 提供者。"""
        _RERANKER_REGISTRY["fake"] = FakeReranker

    def teardown_method(self):
        """每个测试后清理注册表（保留 none）。"""
        _RERANKER_REGISTRY.pop("fake", None)
        _RERANKER_REGISTRY.pop("test_provider", None)

    def test_create_none_provider(self):
        """应能创建内置的 NoneReranker。"""
        reranker = RerankerFactory.create(provider="none")
        assert isinstance(reranker, NoneReranker)

    def test_create_registered_provider(self):
        """应为已注册的提供者创建实例。"""
        reranker = RerankerFactory.create(provider="fake")
        assert isinstance(reranker, FakeReranker)

    def test_unknown_provider_raises(self):
        """未注册的提供者应抛出 ValueError。"""
        with pytest.raises(ValueError, match="未知的重排序提供者.*unknown"):
            RerankerFactory.create(provider="unknown")

    def test_case_insensitive_provider(self):
        """提供者名称匹配应不区分大小写。"""
        reranker = RerankerFactory.create(provider="NONE")
        assert isinstance(reranker, NoneReranker)

    def test_list_providers(self):
        """应列出所有已注册的提供者，包含内置的 none。"""
        providers = RerankerFactory.list_providers()
        assert "none" in providers
        assert "fake" in providers

    def test_default_provider_is_none(self):
        """默认提供者应为 none。"""
        reranker = RerankerFactory.create()
        assert isinstance(reranker, NoneReranker)


@pytest.mark.unit
class TestRegisterRerankerDecorator:
    """测试 @register_reranker 装饰器。"""

    def teardown_method(self):
        """清理注册表。"""
        _RERANKER_REGISTRY.pop("test_provider", None)

    def test_register_decorator(self):
        """@register_reranker 应将类添加到注册表。"""
        @register_reranker("test_provider")
        class TestReranker(BaseReranker):
            def rerank(self, query, candidates, top_k=None, **kwargs):
                return []

        assert "test_provider" in _RERANKER_REGISTRY
        assert _RERANKER_REGISTRY["test_provider"] is TestReranker

    def test_register_lowercase(self):
        """提供者名称应存储为小写。"""
        @register_reranker("TEST_UPPER")
        class UpperReranker(BaseReranker):
            def rerank(self, query, candidates, top_k=None, **kwargs):
                return []

        assert "test_upper" in _RERANKER_REGISTRY
        _RERANKER_REGISTRY.pop("test_upper", None)
