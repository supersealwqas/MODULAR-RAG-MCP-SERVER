"""测试 Embedding 抽象接口与工厂。"""

import pytest
from typing import List

from src.core.settings import EmbeddingConfig
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory, register_embedding, _EMBEDDING_REGISTRY


# --- Fake Embedding 用于测试 ---

class FakeEmbedding(BaseEmbedding):
    """测试用的假 Embedding 实现，返回固定向量。"""

    def __init__(self, model: str, dimensions: int = 1024, api_key: str = "", **kwargs):
        super().__init__(model, dimensions, api_key, **kwargs)
        self.kwargs = kwargs

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """返回与输入文本数量相同的固定向量。"""
        # 使用文本长度生成伪向量，确保相同文本返回相同结果
        return [[float(i % 10) / 10] * self.dimensions for i, _ in enumerate(texts)]


# --- 测试用例 ---

@pytest.mark.unit
class TestBaseEmbedding:
    """测试 BaseEmbedding 抽象类。"""

    def test_cannot_instantiate_abstract(self):
        """BaseEmbedding 是抽象类，不能直接实例化。"""
        with pytest.raises(TypeError):
            BaseEmbedding(model="test")

    def test_fake_embedding_embed(self):
        """FakeEmbedding 应正确实现 embed 方法。"""
        emb = FakeEmbedding(model="fake-model", dimensions=128)
        results = emb.embed(["hello", "world"])
        assert len(results) == 2
        assert len(results[0]) == 128
        assert len(results[1]) == 128

    def test_embed_single(self):
        """embed_single 应返回单个向量。"""
        emb = FakeEmbedding(model="fake", dimensions=64)
        vector = emb.embed_single("test text")
        assert len(vector) == 64
        assert isinstance(vector[0], float)

    def test_stable_embeddings(self):
        """相同文本应返回相同向量。"""
        emb = FakeEmbedding(model="fake", dimensions=32)
        v1 = emb.embed_single("hello")
        v2 = emb.embed_single("hello")
        assert v1 == v2


@pytest.mark.unit
class TestEmbeddingFactory:
    """测试 EmbeddingFactory 的路由逻辑。"""

    def setup_method(self):
        """每个测试前注册 fake 提供者。"""
        _EMBEDDING_REGISTRY["fake"] = FakeEmbedding

    def teardown_method(self):
        """每个测试后清理注册表。"""
        _EMBEDDING_REGISTRY.pop("fake", None)

    def test_create_registered_provider(self):
        """应为已注册的提供者创建实例。"""
        config = EmbeddingConfig(provider="fake", model="test-model", dimensions=256)
        emb = EmbeddingFactory.create(config)
        assert isinstance(emb, FakeEmbedding)
        assert emb.model == "test-model"
        assert emb.dimensions == 256

    def test_create_passes_kwargs(self):
        """应将 api_key 传递给构造函数。"""
        config = EmbeddingConfig(provider="fake", model="test", api_key="sk-test")
        emb = EmbeddingFactory.create(config)
        assert emb.api_key == "sk-test"

    def test_unknown_provider_raises(self):
        """未注册的提供者应抛出 ValueError。"""
        config = EmbeddingConfig(provider="unknown", model="test")
        with pytest.raises(ValueError, match="未知的 Embedding 提供者.*unknown"):
            EmbeddingFactory.create(config)

    def test_case_insensitive_provider(self):
        """提供者名称匹配应不区分大小写。"""
        config = EmbeddingConfig(provider="FAKE", model="test")
        emb = EmbeddingFactory.create(config)
        assert isinstance(emb, FakeEmbedding)

    def test_list_providers(self):
        """应列出所有已注册的提供者。"""
        providers = EmbeddingFactory.list_providers()
        assert "fake" in providers


@pytest.mark.unit
class TestRegisterEmbeddingDecorator:
    """测试 @register_embedding 装饰器。"""

    def teardown_method(self):
        """清理注册表。"""
        _EMBEDDING_REGISTRY.pop("test_provider", None)

    def test_register_decorator(self):
        """@register_embedding 应将类添加到注册表。"""
        @register_embedding("test_provider")
        class TestEmbedding(BaseEmbedding):
            def embed(self, texts, **kwargs):
                return [[0.0] * self.dimensions for _ in texts]

        assert "test_provider" in _EMBEDDING_REGISTRY
        assert _EMBEDDING_REGISTRY["test_provider"] is TestEmbedding

    def test_register_lowercase(self):
        """提供者名称应存储为小写。"""
        @register_embedding("TEST_UPPER")
        class UpperEmbedding(BaseEmbedding):
            def embed(self, texts, **kwargs):
                return [[0.0] * self.dimensions for _ in texts]

        assert "test_upper" in _EMBEDDING_REGISTRY
        _EMBEDDING_REGISTRY.pop("test_upper", None)
