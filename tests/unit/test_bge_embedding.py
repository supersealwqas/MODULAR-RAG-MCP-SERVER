"""BGE Embedding 单元测试。

测试 BGEEmbedding 类的功能，包括：
- 工厂创建
- 模型加载
- 批量编码
- 维度验证
- 错误处理

注意：真实加载模型的测试标记为 @pytest.mark.integration，
      因为 FlagEmbedding 与 mcp.server 的 C 扩展存在 DLL 加载顺序冲突。
"""

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

    def test_embed_mock_single(self):
        """测试单文本编码（mock 模型）。"""
        import numpy as np

        embedding = BGEEmbedding(
            model="bge-m3",
            dimensions=1024,
            model_path="models/bge-m3",
        )
        # mock 模型避免真实加载
        mock_model = MagicMock()
        mock_vec = np.random.randn(1, 1024).astype(np.float32)
        mock_model.encode.return_value = {"dense_vecs": mock_vec}
        embedding._embedding_model = mock_model

        texts = ["这是一个测试文本"]
        vectors = embedding.embed(texts)

        assert len(vectors) == 1
        assert len(vectors[0]) == 1024
        mock_model.encode.assert_called_once()

    def test_embed_mock_batch(self):
        """测试批量编码（mock 模型）。"""
        import numpy as np

        embedding = BGEEmbedding(
            model="bge-m3",
            dimensions=1024,
            model_path="models/bge-m3",
        )
        mock_model = MagicMock()
        mock_vecs = np.random.randn(3, 1024).astype(np.float32)
        mock_model.encode.return_value = {"dense_vecs": mock_vecs}
        embedding._embedding_model = mock_model

        texts = ["第一个文本", "第二个文本", "第三个文本"]
        vectors = embedding.embed(texts)

        assert len(vectors) == 3
        assert all(len(v) == 1024 for v in vectors)

    def test_embed_mock_single_method(self):
        """测试 embed_single 便捷方法（mock 模型）。"""
        import numpy as np

        embedding = BGEEmbedding(
            model="bge-m3",
            dimensions=1024,
            model_path="models/bge-m3",
        )
        mock_model = MagicMock()
        mock_vec = np.random.randn(1, 1024).astype(np.float32)
        mock_model.encode.return_value = {"dense_vecs": mock_vec}
        embedding._embedding_model = mock_model

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
        """测试相同文本生成稳定向量（mock 模型）。"""
        import numpy as np

        embedding = BGEEmbedding(
            model="bge-m3",
            dimensions=1024,
            model_path="models/bge-m3",
        )
        mock_model = MagicMock()
        fixed_vec = np.ones((1, 1024), dtype=np.float32) * 0.5
        mock_model.encode.return_value = {"dense_vecs": fixed_vec}
        embedding._embedding_model = mock_model

        vector1 = embedding.embed_single("稳定性测试文本")
        vector2 = embedding.embed_single("稳定性测试文本")
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


@pytest.mark.integration
class TestBGEEmbeddingRealLoad:
    """BGE Embedding 真实模型加载测试。

    这些测试会真实加载 models/bge-m3 模型。
    标记为 integration，需单独运行：
        uv run pytest tests/unit/test_bge_embedding.py -m integration -v
    """

    def test_embed_real_load(self):
        """测试真实加载模型并编码文本。"""
        embedding = BGEEmbedding(
            model="bge-m3",
            dimensions=1024,
            model_path="models/bge-m3",
        )
        texts = ["这是一个测试文本"]
        vectors = embedding.embed(texts)
        assert len(vectors) == 1
        assert len(vectors[0]) == 1024
        assert all(isinstance(v, float) for v in vectors[0])

    def test_embed_real_batch(self):
        """测试真实批量编码。"""
        embedding = BGEEmbedding(
            model="bge-m3",
            dimensions=1024,
            model_path="models/bge-m3",
        )
        texts = ["第一个文本", "第二个文本", "第三个文本"]
        vectors = embedding.embed(texts)
        assert len(vectors) == 3
        assert all(len(v) == 1024 for v in vectors)

    def test_embed_real_single(self):
        """测试真实 embed_single。"""
        embedding = BGEEmbedding(
            model="bge-m3",
            dimensions=1024,
            model_path="models/bge-m3",
        )
        vector = embedding.embed_single("测试单文本编码")
        assert len(vector) == 1024

    def test_embed_real_stable(self):
        """测试真实向量稳定性。"""
        embedding = BGEEmbedding(
            model="bge-m3",
            dimensions=1024,
            model_path="models/bge-m3",
        )
        text = "稳定性测试文本"
        vector1 = embedding.embed_single(text)
        vector2 = embedding.embed_single(text)
        assert vector1 == vector2
