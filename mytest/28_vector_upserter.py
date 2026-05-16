"""C12 VectorUpserter 手动测试脚本。

测试向量写入器的幂等性、批量写入、Trace 记录等功能。
使用 Mock VectorStore 进行隔离测试。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.settings import Settings, VectorStoreConfig
from src.core.trace.trace_context import TraceContext
from src.core.types import ChunkRecord
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord
from unittest.mock import MagicMock
from typing import Any, Dict, List, Optional


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def safe_print(text: str):
    """安全打印，忽略无法编码的字符。"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Windows GBK 环境下降级输出
        print(text.encode("gbk", errors="replace").decode("gbk", errors="replace"))


class MemoryVectorStore(BaseVectorStore):
    """内存向量存储，用于手动测试。"""

    def __init__(self) -> None:
        super().__init__(collection_name="manual_test")
        self._storage: Dict[str, VectorRecord] = {}

    def upsert(self, records: List[VectorRecord], **kwargs) -> int:
        for record in records:
            self._storage[record.id] = record
        return len(records)

    def query(self, vector, top_k=10, filters=None, **kwargs) -> list:
        return []

    def delete(self, ids: List[str], **kwargs) -> int:
        deleted = 0
        for id_ in ids:
            if id_ in self._storage:
                del self._storage[id_]
                deleted += 1
        return deleted

    def count(self, **kwargs) -> int:
        return len(self._storage)


def _make_settings() -> Settings:
    """创建测试配置。"""
    return Settings(
        llm=MagicMock(),
        vision_llm=MagicMock(),
        ollama=MagicMock(),
        embedding=MagicMock(),
        splitter=MagicMock(),
        vector_store=VectorStoreConfig(provider="mock", persist_directory="/tmp/test"),
        retrieval=MagicMock(),
        rerank=MagicMock(),
        evaluation=MagicMock(),
        observability=MagicMock(),
    )


def _make_records(n: int = 5) -> list:
    """创建 n 条测试记录。"""
    return [
        ChunkRecord(
            id=f"doc_{i:04d}_abcd1234",
            text=f"RAG 检索增强生成技术是当前 AI 领域的热门方向，文档片段 {i}",
            metadata={"source_path": "/test.pdf", "chunk_index": i},
            dense_vector=[0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i],
        )
        for i in range(n)
    ]


def test_basic_upsert():
    """测试基础 upsert 功能。"""
    section("基础 Upsert")

    store = MemoryVectorStore()
    settings = _make_settings()
    upserter = VectorUpserter(settings=settings, vector_store=store)

    records = _make_records(5)
    trace = TraceContext(trace_type="ingestion")
    result = upserter.upsert(records, trace=trace)

    safe_print(f"  写入记录数: {result}")
    safe_print(f"  存储中记录数: {store.count()}")
    safe_print(f"  Trace 阶段: {[s['name'] for s in trace.stages]}")
    safe_print(f"  Trace 详情: {trace.stages[0]}")
    return store, upserter


def test_idempotency():
    """测试幂等性：重复写入不产生重复记录。"""
    section("幂等性测试")

    store = MemoryVectorStore()
    settings = _make_settings()
    upserter = VectorUpserter(settings=settings, vector_store=store)

    records = _make_records(3)

    # 第一次写入
    result1 = upserter.upsert(records)
    safe_print(f"  第一次写入: {result1} 条, 存储: {store.count()} 条")

    # 第二次写入相同内容
    result2 = upserter.upsert(records)
    safe_print(f"  第二次写入: {result2} 条, 存储: {store.count()} 条")

    # 验证幂等
    assert store.count() == 3, f"幂等性失败: 期望 3 条, 实际 {store.count()} 条"
    safe_print("  幂等性验证通过 OK")


def test_content_change():
    """测试内容变更时 ID 变更。"""
    section("内容变更 ID 测试")

    store = MemoryVectorStore()
    settings = _make_settings()
    upserter = VectorUpserter(settings=settings, vector_store=store)

    # 写入原始内容
    record1 = ChunkRecord(
        id="doc_0001_abcd1234",
        text="原始内容",
        metadata={"source_path": "/test.pdf", "chunk_index": 0},
        dense_vector=[1.0, 0.0, 0.0],
    )
    upserter.upsert([record1])
    safe_print(f"  写入原始内容: {store.count()} 条")

    # 写入不同 ID 的记录（模拟内容变更）
    record2 = ChunkRecord(
        id="doc_0001_efgh5678",
        text="修改后内容",
        metadata={"source_path": "/test.pdf", "chunk_index": 0},
        dense_vector=[0.0, 1.0, 0.0],
    )
    upserter.upsert([record2])
    safe_print(f"  写入变更内容: {store.count()} 条")

    # 覆盖同 ID
    record3 = ChunkRecord(
        id="doc_0001_abcd1234",
        text="覆盖原始内容",
        metadata={"source_path": "/test.pdf", "chunk_index": 0},
        dense_vector=[0.0, 0.0, 1.0],
    )
    upserter.upsert([record3])
    safe_print(f"  覆盖同 ID: {store.count()} 条")

    stored = store._storage["doc_0001_abcd1234"]
    safe_print(f"  覆盖后文本: {stored.text}")
    safe_print(f"  覆盖后向量: {stored.vector}")


def test_stable_id_generation():
    """测试确定性 ID 生成。"""
    section("确定性 ID 生成")

    id1 = VectorUpserter.generate_stable_id("/test.pdf", 0, "abcd1234efgh5678")
    id2 = VectorUpserter.generate_stable_id("/test.pdf", 0, "abcd1234efgh5678")
    id3 = VectorUpserter.generate_stable_id("/test.pdf", 1, "abcd1234efgh5678")

    safe_print(f"  相同输入 ID1: {id1}")
    safe_print(f"  相同输入 ID2: {id2}")
    safe_print(f"  不同 index ID: {id3}")
    safe_print(f"  ID1 == ID2: {id1 == id2}")
    safe_print(f"  ID1 != ID3: {id1 != id3}")
    safe_print(f"  ID 长度: {len(id1)}")


def test_delete():
    """测试删除功能。"""
    section("删除功能")

    store = MemoryVectorStore()
    settings = _make_settings()
    upserter = VectorUpserter(settings=settings, vector_store=store)

    records = _make_records(5)
    upserter.upsert(records)
    safe_print(f"  写入后: {store.count()} 条")

    # 删除部分
    trace = TraceContext(trace_type="ingestion")
    deleted = upserter.delete(
        ["doc_0000_abcd1234", "doc_0001_abcd1234", "nonexistent"],
        trace=trace,
    )
    safe_print(f"  删除: {deleted} 条, 剩余: {store.count()} 条")
    safe_print(f"  Trace: {trace.stages[0]}")


def test_no_vector_skipped():
    """测试无向量记录被跳过。"""
    section("无向量记录跳过")

    store = MemoryVectorStore()
    settings = _make_settings()
    upserter = VectorUpserter(settings=settings, vector_store=store)

    records = [
        ChunkRecord(id="r1", text="无向量", metadata={}, dense_vector=None),
        ChunkRecord(id="r2", text="有向量", metadata={}, dense_vector=[0.1, 0.2]),
    ]
    result = upserter.upsert(records)
    safe_print(f"  输入: {len(records)} 条")
    safe_print(f"  写入: {result} 条")
    safe_print(f"  存储: {store.count()} 条")
    assert result == 1
    assert store.count() == 1
    safe_print("  跳过逻辑正确 OK")


def main():
    test_basic_upsert()
    test_idempotency()
    test_content_change()
    test_stable_id_generation()
    test_delete()
    test_no_vector_skipped()

    print(f"\n{'='*60}")
    print("  全部手动测试完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
