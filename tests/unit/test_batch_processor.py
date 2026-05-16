"""C10 BatchProcessor 单元测试。

使用 Mock Encoder 隔离测试，覆盖批处理编排、分批逻辑、顺序保持、
合并逻辑、Trace 记录等。
验收标准：batch_size=2 时对 5 chunks 分成 3 批，且顺序稳定。
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, call, patch

import pytest

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk, ChunkRecord
from src.ingestion.embedding.batch_processor import BatchProcessor
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder


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
        embedding=MagicMock(),
        splitter=MagicMock(),
        vector_store=MagicMock(),
        retrieval=MagicMock(),
        rerank=MagicMock(),
        evaluation=MagicMock(),
        observability=MagicMock(),
    )


def _make_mock_dense_encoder(dimensions: int = 4) -> MagicMock:
    """创建 Mock DenseEncoder。"""
    encoder = MagicMock(spec=DenseEncoder)

    def mock_encode(chunks, trace=None):
        records = []
        for i, chunk in enumerate(chunks):
            records.append(ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=chunk.metadata.copy(),
                dense_vector=[float(i + 1)] * dimensions,
            ))
        return records

    encoder.encode.side_effect = mock_encode
    return encoder


def _make_mock_sparse_encoder() -> MagicMock:
    """创建 Mock SparseEncoder。"""
    encoder = MagicMock(spec=SparseEncoder)

    def mock_encode(chunks, trace=None):
        records = []
        for chunk in chunks:
            records.append(ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=chunk.metadata.copy(),
                sparse_vector={"term": 1.0, chunk.id: 2.0},
            ))
        return records

    encoder.encode.side_effect = mock_encode
    return encoder


# ============================================================
# 测试：基本处理功能
# ============================================================

class TestProcessBasic:
    """基本处理功能测试。"""

    def test_process_returns_chunk_records(self):
        """process 返回 ChunkRecord 列表。"""
        dense = _make_mock_dense_encoder()
        sparse = _make_mock_sparse_encoder()
        processor = BatchProcessor(
            settings=_make_settings_stub(),
            dense_encoder=dense,
            sparse_encoder=sparse,
        )
        chunks = [_make_chunk("text_1", "c1"), _make_chunk("text_2", "c2")]
        records = processor.process(chunks)

        assert len(records) == 2
        assert all(isinstance(r, ChunkRecord) for r in records)

    def test_process_fills_both_vectors(self):
        """process 输出同时包含 dense_vector 和 sparse_vector。"""
        dense = _make_mock_dense_encoder()
        sparse = _make_mock_sparse_encoder()
        processor = BatchProcessor(
            settings=_make_settings_stub(),
            dense_encoder=dense,
            sparse_encoder=sparse,
        )
        chunks = [_make_chunk("text", "c1")]
        records = processor.process(chunks)

        assert records[0].dense_vector is not None
        assert records[0].sparse_vector is not None

    def test_process_preserves_chunk_count(self):
        """输出记录数量与输入 chunk 数量一致。"""
        dense = _make_mock_dense_encoder()
        sparse = _make_mock_sparse_encoder()
        processor = BatchProcessor(
            settings=_make_settings_stub(),
            dense_encoder=dense,
            sparse_encoder=sparse,
        )
        chunks = [_make_chunk(f"text_{i}", f"c{i}") for i in range(5)]
        records = processor.process(chunks)

        assert len(records) == 5

    def test_process_preserves_ids(self):
        """process 保持 chunk ID 不变。"""
        dense = _make_mock_dense_encoder()
        sparse = _make_mock_sparse_encoder()
        processor = BatchProcessor(
            settings=_make_settings_stub(),
            dense_encoder=dense,
            sparse_encoder=sparse,
        )
        chunks = [_make_chunk("t1", "id_001"), _make_chunk("t2", "id_002")]
        records = processor.process(chunks)

        assert records[0].id == "id_001"
        assert records[1].id == "id_002"


# ============================================================
# 测试：分批逻辑
# ============================================================

class TestBatchSplitting:
    """分批逻辑测试。"""

    def test_batch_size_2_with_5_chunks_makes_3_batches(self):
        """batch_size=2 时对 5 chunks 分成 3 批。"""
        dense = _make_mock_dense_encoder()
        sparse = _make_mock_sparse_encoder()
        processor = BatchProcessor(
            settings=_make_settings_stub(),
            dense_encoder=dense,
            sparse_encoder=sparse,
            batch_size=2,
        )
        chunks = [_make_chunk(f"text_{i}", f"c{i}") for i in range(5)]
        processor.process(chunks)

        # dense 和 sparse 各被调用 3 次
        assert dense.encode.call_count == 3
        assert sparse.encode.call_count == 3

    def test_batch_size_10_with_3_chunks_makes_1_batch(self):
        """batch_size 大于 chunk 数时只分 1 批。"""
        dense = _make_mock_dense_encoder()
        sparse = _make_mock_sparse_encoder()
        processor = BatchProcessor(
            settings=_make_settings_stub(),
            dense_encoder=dense,
            sparse_encoder=sparse,
            batch_size=10,
        )
        chunks = [_make_chunk(f"text_{i}", f"c{i}") for i in range(3)]
        processor.process(chunks)

        assert dense.encode.call_count == 1
        assert sparse.encode.call_count == 1

    def test_default_batch_size(self):
        """默认 batch_size 为 64。"""
        processor = BatchProcessor(settings=_make_settings_stub())
        assert processor.batch_size == 64

    def test_split_into_batches(self):
        """split_into_batches 正确分批。"""
        processor = BatchProcessor(
            settings=_make_settings_stub(),
            batch_size=2,
        )
        chunks = [_make_chunk(f"text_{i}", f"c{i}") for i in range(5)]
        batches = processor.split_into_batches(chunks)

        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1

    def test_split_into_batches_preserves_order(self):
        """split_into_batches 保持顺序。"""
        processor = BatchProcessor(
            settings=_make_settings_stub(),
            batch_size=2,
        )
        chunks = [_make_chunk(f"text_{i}", f"c{i}") for i in range(5)]
        batches = processor.split_into_batches(chunks)

        # 所有 batch 拼接后顺序与原始一致
        all_chunks = [c for batch in batches for c in batch]
        assert [c.id for c in all_chunks] == [c.id for c in chunks]


# ============================================================
# 测试：顺序稳定性
# ============================================================

class TestOrderStability:
    """顺序稳定性测试。"""

    def test_output_order_matches_input_order(self):
        """输出顺序与输入顺序一致。"""
        dense = _make_mock_dense_encoder()
        sparse = _make_mock_sparse_encoder()
        processor = BatchProcessor(
            settings=_make_settings_stub(),
            dense_encoder=dense,
            sparse_encoder=sparse,
            batch_size=2,
        )
        chunk_ids = [f"c{i}" for i in range(7)]
        chunks = [_make_chunk(f"text_{i}", cid) for i, cid in enumerate(chunk_ids)]
        records = processor.process(chunks)

        assert [r.id for r in records] == chunk_ids

    def test_repeated_process_stable(self):
        """多次 process 产生相同顺序。"""
        dense = _make_mock_dense_encoder()
        sparse = _make_mock_sparse_encoder()
        processor = BatchProcessor(
            settings=_make_settings_stub(),
            dense_encoder=dense,
            sparse_encoder=sparse,
            batch_size=2,
        )
        chunks = [_make_chunk(f"text_{i}", f"c{i}") for i in range(5)]

        records1 = processor.process(chunks)
        records2 = processor.process(chunks)

        assert [r.id for r in records1] == [r.id for r in records2]


# ============================================================
# 测试：空输入
# ============================================================

class TestEmptyInput:
    """空输入测试。"""

    def test_empty_chunks_returns_empty(self):
        """空列表输入返回空列表。"""
        dense = _make_mock_dense_encoder()
        sparse = _make_mock_sparse_encoder()
        processor = BatchProcessor(
            settings=_make_settings_stub(),
            dense_encoder=dense,
            sparse_encoder=sparse,
        )
        records = processor.process([])

        assert records == []
        dense.encode.assert_not_called()
        sparse.encode.assert_not_called()


# ============================================================
# 测试：合并逻辑
# ============================================================

class TestMergeLogic:
    """合并逻辑测试。"""

    def test_merge_combines_vectors(self):
        """合并后同时包含 dense_vector 和 sparse_vector。"""
        dense = _make_mock_dense_encoder(dimensions=8)
        sparse = _make_mock_sparse_encoder()
        processor = BatchProcessor(
            settings=_make_settings_stub(),
            dense_encoder=dense,
            sparse_encoder=sparse,
        )
        chunks = [_make_chunk("text", "c1")]
        records = processor.process(chunks)

        assert len(records[0].dense_vector) == 8
        assert isinstance(records[0].sparse_vector, dict)

    def test_merge_preserves_metadata(self):
        """合并后 metadata 保持完整。"""
        dense = _make_mock_dense_encoder()
        sparse = _make_mock_sparse_encoder()
        processor = BatchProcessor(
            settings=_make_settings_stub(),
            dense_encoder=dense,
            sparse_encoder=sparse,
        )
        chunk = _make_chunk("text", "c1")
        chunk.metadata["custom"] = "value"
        records = processor.process([chunk])

        assert records[0].metadata["custom"] == "value"
        assert records[0].metadata["source_path"] == "/test.pdf"


# ============================================================
# 测试：Trace 记录
# ============================================================

class TestTraceRecording:
    """Trace 阶段记录测试。"""

    def test_trace_records_batch_stages(self):
        """每批次记录一个 trace stage。"""
        dense = _make_mock_dense_encoder()
        sparse = _make_mock_sparse_encoder()
        processor = BatchProcessor(
            settings=_make_settings_stub(),
            dense_encoder=dense,
            sparse_encoder=sparse,
            batch_size=2,
        )
        trace = TraceContext()
        chunks = [_make_chunk(f"text_{i}", f"c{i}") for i in range(5)]
        processor.process(chunks, trace=trace)

        batch_stages = [s for s in trace.stages if s["name"].startswith("batch_") and s["name"] != "batch_process"]
        assert len(batch_stages) == 3

    def test_trace_records_batch_process_stage(self):
        """总体阶段记录在 trace 中。"""
        dense = _make_mock_dense_encoder()
        sparse = _make_mock_sparse_encoder()
        processor = BatchProcessor(
            settings=_make_settings_stub(),
            dense_encoder=dense,
            sparse_encoder=sparse,
        )
        trace = TraceContext()
        processor.process([_make_chunk("text", "c1")], trace=trace)

        stage_names = [s["name"] for s in trace.stages]
        assert "batch_process" in stage_names

    def test_trace_records_chunk_count(self):
        """trace 记录总 chunk 数量。"""
        dense = _make_mock_dense_encoder()
        sparse = _make_mock_sparse_encoder()
        processor = BatchProcessor(
            settings=_make_settings_stub(),
            dense_encoder=dense,
            sparse_encoder=sparse,
        )
        trace = TraceContext()
        chunks = [_make_chunk(f"text_{i}", f"c{i}") for i in range(3)]
        processor.process(chunks, trace=trace)

        process_stage = [s for s in trace.stages if s["name"] == "batch_process"][0]
        assert process_stage["chunk_count"] == 3

    def test_trace_records_batch_count(self):
        """trace 记录批次数。"""
        dense = _make_mock_dense_encoder()
        sparse = _make_mock_sparse_encoder()
        processor = BatchProcessor(
            settings=_make_settings_stub(),
            dense_encoder=dense,
            sparse_encoder=sparse,
            batch_size=2,
        )
        trace = TraceContext()
        chunks = [_make_chunk(f"text_{i}", f"c{i}") for i in range(5)]
        processor.process(chunks, trace=trace)

        process_stage = [s for s in trace.stages if s["name"] == "batch_process"][0]
        assert process_stage["batch_count"] == 3

    def test_no_trace_no_error(self):
        """不传 trace 时不报错。"""
        dense = _make_mock_dense_encoder()
        sparse = _make_mock_sparse_encoder()
        processor = BatchProcessor(
            settings=_make_settings_stub(),
            dense_encoder=dense,
            sparse_encoder=sparse,
        )
        records = processor.process([_make_chunk("text", "c1")])
        assert len(records) == 1


# ============================================================
# 测试：延迟创建
# ============================================================

class TestLazyCreation:
    """Encoder 延迟创建测试。"""

    def test_encoders_created_on_first_process(self):
        """Encoder 在首次 process 时延迟创建。"""
        with (
            patch("src.ingestion.embedding.batch_processor.DenseEncoder") as MockDense,
            patch("src.ingestion.embedding.batch_processor.SparseEncoder") as MockSparse,
        ):
            MockDense.return_value = _make_mock_dense_encoder()
            MockSparse.return_value = _make_mock_sparse_encoder()

            processor = BatchProcessor(settings=_make_settings_stub())
            processor.process([_make_chunk("text", "c1")])

            MockDense.assert_called_once()
            MockSparse.assert_called_once()

    def test_encoders_reused_across_calls(self):
        """多次 process 复用同一个 Encoder 实例。"""
        dense = _make_mock_dense_encoder()
        sparse = _make_mock_sparse_encoder()
        processor = BatchProcessor(
            settings=_make_settings_stub(),
            dense_encoder=dense,
            sparse_encoder=sparse,
        )
        processor.process([_make_chunk("t1", "c1")])
        processor.process([_make_chunk("t2", "c2")])

        # encode 各被调用 2 次（每次 process 调用 1 次）
        assert dense.encode.call_count == 2
        assert sparse.encode.call_count == 2
