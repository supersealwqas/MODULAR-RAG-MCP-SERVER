"""C8 DenseEncoder 单元测试。

使用 Mock Embedding 隔离测试，覆盖向量编码、批量处理、降级、异常处理等。
验收标准：向量数量与 chunks 数量一致，维度一致。
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import pytest

from src.core.settings import EmbeddingConfig, Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk, ChunkRecord
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.libs.embedding.base_embedding import BaseEmbedding


# ============================================================
# 测试辅助函数
# ============================================================


def _make_chunk(text: str, chunk_id: str = "test_0000_abcd1234") -> Chunk:
    """创建测试用 Chunk。"""
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "/test.pdf", "chunk_index": 0},
        source_ref="doc_001",
    )


def _make_settings_stub() -> Settings:
    """创建测试用 Settings stub。"""
    return Settings(
        llm=MagicMock(),
        vision_llm=MagicMock(),
        ollama=MagicMock(),
        embedding=EmbeddingConfig(
            provider="test",
            model="test-model",
            dimensions=4,
        ),
        splitter=MagicMock(),
        vector_store=MagicMock(),
        retrieval=MagicMock(),
        rerank=MagicMock(),
        evaluation=MagicMock(),
        observability=MagicMock(),
        pipeline=MagicMock(),
    )


def _make_mock_embedding(dimensions: int = 4) -> MagicMock:
    """创建 Mock Embedding 实例。

    参数:
        dimensions: 输出向量维度
    """
    embedding = MagicMock(spec=BaseEmbedding)
    embedding.dimensions = dimensions

    def mock_embed(texts: List[str], **kwargs) -> List[List[float]]:
        """根据文本长度生成确定性向量。"""
        return [[float(i + 1)] * dimensions for i in range(len(texts))]

    embedding.embed.side_effect = mock_embed
    return embedding


# ============================================================
# 测试：基本编码功能
# ============================================================

class TestEncodeBasic:
    """基本编码功能测试。"""

    def test_encode_returns_chunk_records(self):
        """encode 返回 ChunkRecord 列表。"""
        mock_embedding = _make_mock_embedding()
        encoder = DenseEncoder(settings=_make_settings_stub(), embedding=mock_embedding)
        chunks = [_make_chunk("文本一", "c1"), _make_chunk("文本二", "c2")]
        records = encoder.encode(chunks)

        assert len(records) == 2
        assert all(isinstance(r, ChunkRecord) for r in records)

    def test_encode_vector_count_matches_chunks(self):
        """输出向量数量与 chunks 数量一致。"""
        mock_embedding = _make_mock_embedding(dimensions=8)
        encoder = DenseEncoder(settings=_make_settings_stub(), embedding=mock_embedding)
        chunks = [_make_chunk(f"text_{i}", f"c{i}") for i in range(5)]
        records = encoder.encode(chunks)

        assert len(records) == 5
        assert all(r.dense_vector is not None for r in records)

    def test_encode_vector_dimension_consistent(self):
        """输出向量维度与 embedding 维度一致。"""
        dim = 16
        mock_embedding = _make_mock_embedding(dimensions=dim)
        encoder = DenseEncoder(settings=_make_settings_stub(), embedding=mock_embedding)
        chunks = [_make_chunk("text", "c1")]
        records = encoder.encode(chunks)

        assert len(records[0].dense_vector) == dim

    def test_encode_preserves_chunk_id(self):
        """encode 保持 chunk ID 不变。"""
        mock_embedding = _make_mock_embedding()
        encoder = DenseEncoder(settings=_make_settings_stub(), embedding=mock_embedding)
        chunks = [_make_chunk("text_1", "id_001"), _make_chunk("text_2", "id_002")]
        records = encoder.encode(chunks)

        assert records[0].id == "id_001"
        assert records[1].id == "id_002"

    def test_encode_preserves_chunk_text(self):
        """encode 保持 chunk 文本不变。"""
        mock_embedding = _make_mock_embedding()
        encoder = DenseEncoder(settings=_make_settings_stub(), embedding=mock_embedding)
        chunks = [_make_chunk("原始文本", "c1")]
        records = encoder.encode(chunks)

        assert records[0].text == "原始文本"

    def test_encode_preserves_metadata(self):
        """encode 保持原有 metadata 字段。"""
        mock_embedding = _make_mock_embedding()
        encoder = DenseEncoder(settings=_make_settings_stub(), embedding=mock_embedding)
        chunk = _make_chunk("text", "c1")
        chunk.metadata["custom_field"] = 42
        records = encoder.encode([chunk])

        assert records[0].metadata["custom_field"] == 42
        assert records[0].metadata["source_path"] == "/test.pdf"


# ============================================================
# 测试：批量处理
# ============================================================

class TestBatchProcessing:
    """批量处理测试。"""

    def test_batch_size_controls_calls(self):
        """batch_size 控制每批处理的 chunk 数量。"""
        mock_embedding = _make_mock_embedding(dimensions=4)
        encoder = DenseEncoder(
            settings=_make_settings_stub(),
            embedding=mock_embedding,
            batch_size=2,
        )
        chunks = [_make_chunk(f"text_{i}", f"c{i}") for i in range(5)]
        encoder.encode(chunks)

        # 5 chunks / batch_size=2 → 3 批调用
        assert mock_embedding.embed.call_count == 3

    def test_large_batch_single_call(self):
        """batch_size 大于 chunk 数量时只调用一次。"""
        mock_embedding = _make_mock_embedding(dimensions=4)
        encoder = DenseEncoder(
            settings=_make_settings_stub(),
            embedding=mock_embedding,
            batch_size=100,
        )
        chunks = [_make_chunk(f"text_{i}", f"c{i}") for i in range(3)]
        encoder.encode(chunks)

        assert mock_embedding.embed.call_count == 1

    def test_default_batch_size(self):
        """默认 batch_size 为 64。"""
        encoder = DenseEncoder(settings=_make_settings_stub())
        assert encoder.batch_size == 64


# ============================================================
# 测试：空输入
# ============================================================

class TestEmptyInput:
    """空输入测试。"""

    def test_empty_chunks_returns_empty(self):
        """空列表输入返回空列表。"""
        mock_embedding = _make_mock_embedding()
        encoder = DenseEncoder(settings=_make_settings_stub(), embedding=mock_embedding)
        records = encoder.encode([])

        assert records == []
        mock_embedding.embed.assert_not_called()


# ============================================================
# 测试：降级行为
# ============================================================

class TestFallback:
    """Embedding 不可用时的降级测试。"""

    def test_embedding_unavailable_returns_records_without_vectors(self):
        """Embedding 不可用时返回不含向量的 ChunkRecord。"""
        # 创建会抛异常的 settings
        bad_settings = _make_settings_stub()
        bad_settings.embedding = EmbeddingConfig(
            provider="nonexistent",
            model="nonexistent-model",
        )

        encoder = DenseEncoder(settings=bad_settings, embedding=None)
        chunks = [_make_chunk("text", "c1")]
        records = encoder.encode(chunks)

        assert len(records) == 1
        assert records[0].id == "c1"
        assert records[0].dense_vector is None

    def test_embedding_creation_failure_returns_records(self):
        """EmbeddingFactory 抛异常时返回不含向量的 ChunkRecord。"""
        with patch("src.ingestion.embedding.dense_encoder.EmbeddingFactory") as mock_factory:
            mock_factory.create.side_effect = ValueError("未知 provider")
            encoder = DenseEncoder(settings=_make_settings_stub())
            chunks = [_make_chunk("text", "c1")]
            records = encoder.encode(chunks)

        assert len(records) == 1
        assert records[0].dense_vector is None


# ============================================================
# 测试：异常处理
# ============================================================

class TestErrorHandling:
    """异常处理测试。"""

    def test_single_batch_failure_fills_with_zero_vectors(self):
        """单个批次失败时用零向量填充。"""
        mock_embedding = _make_mock_embedding(dimensions=4)
        call_count = [0]

        def failing_embed(texts):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("API error")
            return [[1.0] * 4 for _ in texts]

        mock_embedding.embed.side_effect = failing_embed
        encoder = DenseEncoder(
            settings=_make_settings_stub(),
            embedding=mock_embedding,
            batch_size=2,
        )
        chunks = [_make_chunk(f"text_{i}", f"c{i}") for i in range(6)]
        records = encoder.encode(chunks)

        # 6 chunks / batch_size=2 → 3 批，第 2 批失败
        assert len(records) == 6
        # 第 1 批（c0, c1）应有正常向量
        assert records[0].dense_vector == [1.0, 1.0, 1.0, 1.0]
        # 第 2 批（c2, c3）应为零向量
        assert records[2].dense_vector == [0.0, 0.0, 0.0, 0.0]
        # 第 3 批（c4, c5）应有正常向量
        assert records[4].dense_vector is not None

    def test_metadata_not_shared_between_records(self):
        """不同 ChunkRecord 的 metadata 应独立，不共享引用。"""
        mock_embedding = _make_mock_embedding()
        encoder = DenseEncoder(settings=_make_settings_stub(), embedding=mock_embedding)
        chunks = [_make_chunk("text_1", "c1"), _make_chunk("text_2", "c2")]
        records = encoder.encode(chunks)

        records[0].metadata["extra"] = "value"
        assert "extra" not in records[1].metadata


# ============================================================
# 测试：Trace 记录
# ============================================================

class TestTraceRecording:
    """Trace 阶段记录测试。"""

    def test_trace_records_dense_encode_stage(self):
        """encode 成功时 trace 记录 dense_encode 阶段。"""
        mock_embedding = _make_mock_embedding(dimensions=8)
        encoder = DenseEncoder(settings=_make_settings_stub(), embedding=mock_embedding)
        trace = TraceContext()
        chunks = [_make_chunk("text", "c1")]
        encoder.encode(chunks, trace=trace)

        stage_names = [s["name"] for s in trace.stages]
        assert "dense_encode" in stage_names

    def test_trace_records_chunk_count(self):
        """trace 记录 chunk 数量。"""
        mock_embedding = _make_mock_embedding()
        encoder = DenseEncoder(settings=_make_settings_stub(), embedding=mock_embedding)
        trace = TraceContext()
        chunks = [_make_chunk(f"text_{i}", f"c{i}") for i in range(3)]
        encoder.encode(chunks, trace=trace)

        dense_stage = [s for s in trace.stages if s["name"] == "dense_encode"][0]
        assert dense_stage["chunk_count"] == 3

    def test_trace_records_vector_dim(self):
        """trace 记录向量维度。"""
        dim = 32
        mock_embedding = _make_mock_embedding(dimensions=dim)
        encoder = DenseEncoder(settings=_make_settings_stub(), embedding=mock_embedding)
        trace = TraceContext()
        encoder.encode([_make_chunk("text", "c1")], trace=trace)

        dense_stage = [s for s in trace.stages if s["name"] == "dense_encode"][0]
        assert dense_stage["vector_dim"] == dim

    def test_trace_records_elapsed_ms(self):
        """trace 记录耗时。"""
        mock_embedding = _make_mock_embedding()
        encoder = DenseEncoder(settings=_make_settings_stub(), embedding=mock_embedding)
        trace = TraceContext()
        encoder.encode([_make_chunk("text", "c1")], trace=trace)

        dense_stage = [s for s in trace.stages if s["name"] == "dense_encode"][0]
        assert "elapsed_ms" in dense_stage
        assert dense_stage["elapsed_ms"] >= 0

    def test_no_trace_no_error(self):
        """不传 trace 时不报错。"""
        mock_embedding = _make_mock_embedding()
        encoder = DenseEncoder(settings=_make_settings_stub(), embedding=mock_embedding)
        records = encoder.encode([_make_chunk("text", "c1")])
        assert len(records) == 1


# ============================================================
# 测试：Embedding 延迟创建
# ============================================================

class TestLazyCreation:
    """Embedding 延迟创建测试。"""

    def test_embedding_created_on_first_encode(self):
        """Embedding 在首次 encode 时延迟创建。"""
        with patch("src.ingestion.embedding.dense_encoder.EmbeddingFactory") as mock_factory:
            mock_embedding = _make_mock_embedding()
            mock_factory.create.return_value = mock_embedding

            encoder = DenseEncoder(settings=_make_settings_stub())
            encoder.encode([_make_chunk("text", "c1")])

            mock_factory.create.assert_called_once()

    def test_embedding_reused_across_calls(self):
        """多次 encode 调用复用同一个 Embedding 实例。"""
        with patch("src.ingestion.embedding.dense_encoder.EmbeddingFactory") as mock_factory:
            mock_embedding = _make_mock_embedding()
            mock_factory.create.return_value = mock_embedding

            encoder = DenseEncoder(settings=_make_settings_stub())
            encoder.encode([_make_chunk("text1", "c1")])
            encoder.encode([_make_chunk("text2", "c2")])

            # 只创建一次
            mock_factory.create.assert_called_once()
