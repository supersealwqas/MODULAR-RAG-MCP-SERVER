"""VectorUpserter 单元测试。

测试幂等性、确定性 ID 生成、批量写入、Trace 记录等功能。
使用 Mock VectorStore 隔离外部依赖。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from src.core.settings import Settings, VectorStoreConfig
from src.core.trace.trace_context import TraceContext
from src.core.types import ChunkRecord
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord


# ============================================================
# 辅助工具
# ============================================================


class MockVectorStore(BaseVectorStore):
    """内存 Mock VectorStore，用于测试幂等性。"""

    def __init__(self) -> None:
        super().__init__(collection_name="test")
        self._storage: Dict[str, VectorRecord] = {}
        self.upsert_call_count = 0
        self.last_upserted: List[VectorRecord] = []

    def upsert(self, records: List[VectorRecord], **kwargs) -> int:
        self.upsert_call_count += 1
        self.last_upserted = records
        for record in records:
            self._storage[record.id] = record
        return len(records)

    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> list:
        return []

    def delete(self, ids: List[str], **kwargs) -> int:
        deleted = 0
        for id_ in ids:
            if id_ in self._storage:
                del self._storage[id_]
                deleted += 1
        return deleted

    def get_by_ids(self, ids: List[str], **kwargs) -> list:
        results = []
        for id_ in ids:
            record = self._storage.get(id_)
            if record:
                results.append({
                    "id": record.id,
                    "text": record.text,
                    "metadata": record.metadata,
                })
        return results

    def count(self, **kwargs) -> int:
        return len(self._storage)


def _make_settings() -> Settings:
    """创建测试用 Settings 对象。"""
    return Settings(
        llm=MagicMock(),
        vision_llm=MagicMock(),
        ollama=MagicMock(),
        embedding=MagicMock(),
        splitter=MagicMock(),
        vector_store=VectorStoreConfig(
            provider="mock",
            persist_directory="/tmp/test_chroma",
        ),
        retrieval=MagicMock(),
        rerank=MagicMock(),
        evaluation=MagicMock(),
        observability=MagicMock(),
    )


def _make_record(
    record_id: str = "doc_0001_abcd1234",
    text: str = "RAG 检索增强生成",
    dense_vector: Optional[List[float]] = ...,  # ... 表示使用默认值
    metadata: Optional[Dict[str, Any]] = None,
) -> ChunkRecord:
    """创建测试用 ChunkRecord。"""
    if dense_vector is ...:
        dense_vector = [0.1, 0.2, 0.3]
    return ChunkRecord(
        id=record_id,
        text=text,
        metadata=metadata or {"source_path": "/test.pdf", "chunk_index": 0},
        dense_vector=dense_vector,
    )


def _make_records(n: int = 5) -> List[ChunkRecord]:
    """创建 n 条测试用 ChunkRecord。"""
    return [
        _make_record(
            record_id=f"doc_{i:04d}_abcd1234",
            text=f"测试文档片段 {i}",
            dense_vector=[0.1 * i, 0.2 * i, 0.3 * i],
            metadata={"source_path": "/test.pdf", "chunk_index": i},
        )
        for i in range(n)
    ]


# ============================================================
# 测试用例
# ============================================================


class TestVectorUpserterUpsert:
    """upsert 基础功能测试。"""

    def test_upsert_returns_count(self):
        """upsert 应返回成功写入的记录数。"""
        store = MockVectorStore()
        settings = _make_settings()
        upserter = VectorUpserter(settings=settings, vector_store=store)

        records = _make_records(3)
        result = upserter.upsert(records)

        assert result == 3

    def test_upsert_writes_to_store(self):
        """upsert 应将记录写入 VectorStore。"""
        store = MockVectorStore()
        settings = _make_settings()
        upserter = VectorUpserter(settings=settings, vector_store=store)

        records = _make_records(3)
        upserter.upsert(records)

        assert store.count() == 3
        assert store.upsert_call_count == 1

    def test_upsert_empty_list(self):
        """空列表应返回 0，不调用 store。"""
        store = MockVectorStore()
        settings = _make_settings()
        upserter = VectorUpserter(settings=settings, vector_store=store)

        result = upserter.upsert([])

        assert result == 0
        assert store.upsert_call_count == 0

    def test_upsert_no_dense_vector_skipped(self):
        """没有 dense_vector 的记录应被跳过。"""
        store = MockVectorStore()
        settings = _make_settings()
        upserter = VectorUpserter(settings=settings, vector_store=store)

        records = [
            _make_record(record_id="r1", dense_vector=None),
            _make_record(record_id="r2", dense_vector=[0.1, 0.2]),
        ]
        result = upserter.upsert(records)

        assert result == 1
        assert store.count() == 1

    def test_upsert_all_no_vector_returns_zero(self):
        """所有记录都没有向量时应返回 0。"""
        store = MockVectorStore()
        settings = _make_settings()
        upserter = VectorUpserter(settings=settings, vector_store=store)

        records = [
            _make_record(record_id="r1", dense_vector=None),
            _make_record(record_id="r2", dense_vector=None),
        ]
        result = upserter.upsert(records)

        assert result == 0
        assert store.upsert_call_count == 0


class TestVectorUpserterIdempotency:
    """幂等性测试：同一内容重复写入不产生重复记录。"""

    def test_same_content_same_id(self):
        """相同内容两次 upsert 产生相同 ID，不增加记录数。"""
        store = MockVectorStore()
        settings = _make_settings()
        upserter = VectorUpserter(settings=settings, vector_store=store)

        records = _make_records(3)

        # 第一次写入
        result1 = upserter.upsert(records)
        assert result1 == 3
        assert store.count() == 3

        # 第二次写入相同内容
        result2 = upserter.upsert(records)
        assert result2 == 3
        assert store.count() == 3  # 记录数不变

        # 只调用了 2 次 upsert，但存储中仍为 3 条
        assert store.upsert_call_count == 2

    def test_content_change_different_id(self):
        """内容变更时 ID 应不同（通过 ChunkRecord.id 体现）。"""
        store = MockVectorStore()
        settings = _make_settings()
        upserter = VectorUpserter(settings=settings, vector_store=store)

        # 第一次写入
        record1 = _make_record(record_id="doc_0001_abcd1234", text="原始内容")
        upserter.upsert([record1])
        assert store.count() == 1

        # 内容变更后写入（不同的 id）
        record2 = _make_record(record_id="doc_0001_efgh5678", text="修改后内容")
        upserter.upsert([record2])
        assert store.count() == 2  # 新增一条

    def test_overwrite_same_id(self):
        """相同 ID 重复写入应覆盖旧记录（幂等）。"""
        store = MockVectorStore()
        settings = _make_settings()
        upserter = VectorUpserter(settings=settings, vector_store=store)

        # 第一次写入
        record1 = _make_record(
            record_id="doc_0001_abcd1234",
            text="原始内容",
            dense_vector=[1.0, 0.0, 0.0],
        )
        upserter.upsert([record1])

        # 同 ID 但不同内容/向量（模拟更新）
        record2 = _make_record(
            record_id="doc_0001_abcd1234",
            text="更新后内容",
            dense_vector=[0.0, 1.0, 0.0],
        )
        upserter.upsert([record2])

        # 仍为 1 条记录，但内容已更新
        assert store.count() == 1
        stored = store._storage["doc_0001_abcd1234"]
        assert stored.text == "更新后内容"
        assert stored.vector == [0.0, 1.0, 0.0]


class TestVectorUpserterBatchOrder:
    """批量写入与顺序测试。"""

    def test_batch_upsert_preserves_order(self):
        """批量 upsert 应保持记录顺序。"""
        store = MockVectorStore()
        settings = _make_settings()
        upserter = VectorUpserter(settings=settings, vector_store=store)

        records = _make_records(5)
        upserter.upsert(records)

        # 验证 last_upserted 的顺序与输入一致
        for i, vr in enumerate(store.last_upserted):
            assert vr.id == f"doc_{i:04d}_abcd1234"

    def test_batch_upsert_all_records(self):
        """大批量 upsert 应一次性写入所有记录。"""
        store = MockVectorStore()
        settings = _make_settings()
        upserter = VectorUpserter(settings=settings, vector_store=store)

        records = _make_records(100)
        result = upserter.upsert(records)

        assert result == 100
        assert store.count() == 100
        assert store.upsert_call_count == 1  # 一次调用


class TestVectorUpserterRecordConversion:
    """ChunkRecord → VectorRecord 转换测试。"""

    def test_conversion_preserves_id(self):
        """转换后 ID 应保持一致。"""
        store = MockVectorStore()
        settings = _make_settings()
        upserter = VectorUpserter(settings=settings, vector_store=store)

        record = _make_record(record_id="test_id_1234")
        upserter.upsert([record])

        assert "test_id_1234" in store._storage

    def test_conversion_preserves_text(self):
        """转换后文本应保持一致。"""
        store = MockVectorStore()
        settings = _make_settings()
        upserter = VectorUpserter(settings=settings, vector_store=store)

        record = _make_record(text="RAG 检索增强生成技术")
        upserter.upsert([record])

        stored = store.last_upserted[0]
        assert stored.text == "RAG 检索增强生成技术"

    def test_conversion_preserves_metadata(self):
        """转换后元数据应保持一致（且为副本）。"""
        store = MockVectorStore()
        settings = _make_settings()
        upserter = VectorUpserter(settings=settings, vector_store=store)

        metadata = {"source_path": "/test.pdf", "chunk_index": 5}
        record = _make_record(metadata=metadata)
        upserter.upsert([record])

        stored = store.last_upserted[0]
        assert stored.metadata == metadata
        # 确认是副本（修改不影响原始）
        stored.metadata["extra"] = "added"
        assert "extra" not in record.metadata

    def test_conversion_preserves_vector(self):
        """转换后向量应保持一致。"""
        store = MockVectorStore()
        settings = _make_settings()
        upserter = VectorUpserter(settings=settings, vector_store=store)

        vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        record = _make_record(dense_vector=vector)
        upserter.upsert([record])

        stored = store.last_upserted[0]
        assert stored.vector == vector


class TestVectorUpserterDelete:
    """delete 功能测试。"""

    def test_delete_removes_records(self):
        """delete 应从 VectorStore 中移除记录。"""
        store = MockVectorStore()
        settings = _make_settings()
        upserter = VectorUpserter(settings=settings, vector_store=store)

        records = _make_records(3)
        upserter.upsert(records)
        assert store.count() == 3

        deleted = upserter.delete(["doc_0000_abcd1234", "doc_0001_abcd1234"])
        assert deleted == 2
        assert store.count() == 1

    def test_delete_nonexistent_returns_zero(self):
        """删除不存在的记录应返回 0。"""
        store = MockVectorStore()
        settings = _make_settings()
        upserter = VectorUpserter(settings=settings, vector_store=store)

        deleted = upserter.delete(["nonexistent_id"])
        assert deleted == 0

    def test_delete_empty_list(self):
        """空列表应返回 0。"""
        store = MockVectorStore()
        settings = _make_settings()
        upserter = VectorUpserter(settings=settings, vector_store=store)

        deleted = upserter.delete([])
        assert deleted == 0


class TestVectorUpserterTrace:
    """Trace 记录测试。"""

    def test_upsert_records_trace(self):
        """upsert 应记录 Trace 阶段数据。"""
        store = MockVectorStore()
        settings = _make_settings()
        upserter = VectorUpserter(settings=settings, vector_store=store)

        trace = TraceContext(trace_type="ingestion")
        records = _make_records(3)
        upserter.upsert(records, trace=trace)

        assert len(trace.stages) == 1
        stage = trace.stages[0]
        assert stage["name"] == "vector_upsert"
        assert stage["total_records"] == 3
        assert stage["valid_records"] == 3
        assert stage["skipped_records"] == 0
        assert stage["upserted_count"] == 3

    def test_delete_records_trace(self):
        """delete 应记录 Trace 阶段数据。"""
        store = MockVectorStore()
        settings = _make_settings()
        upserter = VectorUpserter(settings=settings, vector_store=store)

        trace = TraceContext(trace_type="ingestion")
        upserter.delete(["id1", "id2"], trace=trace)

        assert len(trace.stages) == 1
        stage = trace.stages[0]
        assert stage["name"] == "vector_delete"
        assert stage["requested_count"] == 2


class TestVectorUpserterStableId:
    """确定性 ID 生成测试。"""

    def test_same_input_same_id(self):
        """相同输入应产生相同 ID。"""
        id1 = VectorUpserter.generate_stable_id("/test.pdf", 0, "abcd1234")
        id2 = VectorUpserter.generate_stable_id("/test.pdf", 0, "abcd1234")
        assert id1 == id2

    def test_different_path_different_id(self):
        """不同 source_path 应产生不同 ID。"""
        id1 = VectorUpserter.generate_stable_id("/test1.pdf", 0, "abcd1234")
        id2 = VectorUpserter.generate_stable_id("/test2.pdf", 0, "abcd1234")
        assert id1 != id2

    def test_different_index_different_id(self):
        """不同 chunk_index 应产生不同 ID。"""
        id1 = VectorUpserter.generate_stable_id("/test.pdf", 0, "abcd1234")
        id2 = VectorUpserter.generate_stable_id("/test.pdf", 1, "abcd1234")
        assert id1 != id2

    def test_different_content_different_id(self):
        """不同内容哈希应产生不同 ID。"""
        id1 = VectorUpserter.generate_stable_id("/test.pdf", 0, "abcd1234")
        id2 = VectorUpserter.generate_stable_id("/test.pdf", 0, "efgh5678")
        assert id1 != id2

    def test_id_length(self):
        """生成的 ID 应为 16 字符。"""
        stable_id = VectorUpserter.generate_stable_id("/test.pdf", 0, "abcd1234")
        assert len(stable_id) == 16

    def test_id_is_hex(self):
        """生成的 ID 应为十六进制字符串。"""
        stable_id = VectorUpserter.generate_stable_id("/test.pdf", 0, "abcd1234")
        int(stable_id, 16)  # 不应抛出异常


class TestVectorUpserterContentHash:
    """内容哈希测试。"""

    def test_same_text_same_hash(self):
        """相同文本应产生相同哈希。"""
        h1 = VectorUpserter._compute_content_hash("RAG 检索增强生成")
        h2 = VectorUpserter._compute_content_hash("RAG 检索增强生成")
        assert h1 == h2

    def test_different_text_different_hash(self):
        """不同文本应产生不同哈希。"""
        h1 = VectorUpserter._compute_content_hash("RAG 检索增强生成")
        h2 = VectorUpserter._compute_content_hash("向量数据库存储")
        assert h1 != h2

    def test_hash_is_sha256(self):
        """哈希应为 64 字符的 SHA256。"""
        h = VectorUpserter._compute_content_hash("test")
        assert len(h) == 64
