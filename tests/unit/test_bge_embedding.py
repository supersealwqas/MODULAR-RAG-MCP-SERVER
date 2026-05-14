"""BGE Embedding 单元测试。

测试 BGEEmbedding 类的功能，包括：
- 工厂创建
- 模型加载
- 批量编码
- 维度验证
- 错误处理

注意：此测试会真实加载 models/bge-m3 模型，需要确保模型文件存在。
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pytest
from unittest.mock import patch, MagicMock

from src.core.settings import EmbeddingConfig
from src.libs.embedding.bge_embedding import BGEEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory


class TestBGEEmbedding:
    """BGE Embedding 测试类。"""

    def test_factory_create(self):
        """测试通过工厂创建 BGE Embedding 实例。"""
        config = EmbeddingConfig(
            provider="bge",
            model="bge-m3",
            dimensions=1024,
            model_path="models/bge-m3",
        )
        embedding = EmbeddingFactory.create(config)
        assert isinstance(embedding, BGEEmbedding)
        assert embedding.model == "bge-m3"
        assert embedding.dimensions == 1024
        assert embedding.model_path == "models/bge-m3"

    def test_default_values(self):
        """测试默认参数值。"""
        embedding = BGEEmbedding()
        assert embedding.model == "bge-m3"
        assert embedding.dimensions == 1024
        assert embedding.model_path == ""
        assert embedding._embedding_model is None  # 延迟加载

    def test_custom_model_path(self):
        """测试自定义模型路径。"""
        embedding = BGEEmbedding(
            model="bge-m3",
            dimensions=1024,
            model_path="models/bge-m3",
        )
        assert embedding.model_path == "models/bge-m3"

    def test_embed_real_load(self):
        """测试真实加载模型并编码文本。"""
        embedding = BGEEmbedding(
            model="bge-m3",
            dimensions=1024,
            model_path="models/bge-m3",
        )

        # 测试单文本编码
        texts = ["这是一个测试文本"]
        vectors = embedding.embed(texts)

        assert len(vectors) == 1
        assert len(vectors[0]) == 1024  # bge-m3 维度
        assert all(isinstance(v, float) for v in vectors[0])

    def test_embed_batch(self):
        """测试批量编码。"""
        embedding = BGEEmbedding(
            model="bge-m3",
            dimensions=1024,
            model_path="models/bge-m3",
        )

        texts = ["第一个文本", "第二个文本", "第三个文本"]
        vectors = embedding.embed(texts)

        assert len(vectors) == 3
        assert all(len(v) == 1024 for v in vectors)

    def test_embed_single(self):
        """测试单文本编码便捷方法。"""
        embedding = BGEEmbedding(
            model="bge-m3",
            dimensions=1024,
            model_path="models/bge-m3",
        )

        vector = embedding.embed_single("测试单文本编码")
        assert len(vector) == 1024

    def test_embed_empty_input(self):
        """测试空输入抛出异常。"""
        embedding = BGEEmbedding(
            model="bge-m3",
            dimensions=1024,
            model_path="models/bge-m3",
        )

        with pytest.raises(ValueError, match="不能为空"):
            embedding.embed([])

    def test_embed_stable_vectors(self):
        """测试相同文本生成稳定向量。"""
        embedding = BGEEmbedding(
            model="bge-m3",
            dimensions=1024,
            model_path="models/bge-m3",
        )

        text = "稳定性测试文本"
        vector1 = embedding.embed_single(text)
        vector2 = embedding.embed_single(text)

        # 相同文本应生成完全相同的向量
        assert vector1 == vector2

    def test_model_not_found(self):
        """测试模型路径不存在时抛出异常。"""
        embedding = BGEEmbedding(
            model="bge-m3",
            dimensions=1024,
            model_path="models/not_exist_model",
        )

        with pytest.raises(FileNotFoundError, match="路径不存在"):
            embedding.embed(["test"])

    def test_list_providers(self):
        """测试列出已注册的提供者。"""
        providers = EmbeddingFactory.list_providers()
        assert "bge" in providers
