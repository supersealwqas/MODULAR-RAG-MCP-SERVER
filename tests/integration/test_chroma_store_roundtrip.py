"""ChromaStore 集成测试：完整的 upsert→query roundtrip 验证。

使用临时目录进行持久化测试，测试结束后自动清理。
"""

import pytest
import shutil
import tempfile
from pathlib import Path

from src.libs.vector_store.base_vector_store import VectorRecord, QueryResult
from src.libs.vector_store.chroma_store import ChromaStore
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.core.settings import VectorStoreConfig


@pytest.fixture
def tmp_chroma_dir():
    """创建临时 ChromaDB 目录，测试后自动清理。"""
    tmp_dir = tempfile.mkdtemp(prefix="chroma_test_")
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def store(tmp_chroma_dir):
    """创建 ChromaStore 实例，使用临时目录。"""
    return ChromaStore(
        collection_name="test_collection",
        persist_directory=tmp_chroma_dir,
    )


@pytest.fixture
def sample_records():
    """生成测试用向量记录。"""
    return [
        VectorRecord(
            id="doc_001",
            vector=[1.0, 0.0, 0.0],
            text="这是第一个文档",
            metadata={"source": "doc1.pdf", "page": 1},
        ),
        VectorRecord(
            id="doc_002",
            vector=[0.0, 1.0, 0.0],
            text="这是第二个文档",
            metadata={"source": "doc2.pdf", "page": 1},
        ),
        VectorRecord(
            id="doc_003",
            vector=[0.0, 0.0, 1.0],
            text="这是第三个文档",
            metadata={"source": "doc1.pdf", "page": 2},
        ),
    ]


@pytest.mark.integration
class TestChromaStoreUpsertQuery:
    """测试 ChromaStore 的 upsert 和 query 基本功能。"""

    def test_upsert_returns_count(self, store, sample_records):
        """upsert 应返回成功插入的记录数。"""
        count = store.upsert(sample_records)
        assert count == 3

    def test_count_after_upsert(self, store, sample_records):
        """upsert 后 count 应返回正确的记录数。"""
        store.upsert(sample_records)
        assert store.count() == 3

    def test_query_returns_results(self, store, sample_records):
        """query 应返回按相似度排序的结果。"""
        store.upsert(sample_records)
        # 查询向量与 doc_001 最相似
        results = store.query(vector=[1.0, 0.0, 0.0], top_k=3)
        assert len(results) == 3
        assert isinstance(results[0], QueryResult)

    def test_query_top_k(self, store, sample_records):
        """top_k 参数应限制返回数量。"""
        store.upsert(sample_records)
        results = store.query(vector=[1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1

    def test_query_similarity_order(self, store, sample_records):
        """查询结果应按相似度降序排列。"""
        store.upsert(sample_records)
        results = store.query(vector=[1.0, 0.0, 0.0], top_k=3)
        # 第一个结果应与查询向量最相似（doc_001）
        assert results[0].id == "doc_001"
        # 分数应递减
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_query_result_contains_text(self, store, sample_records):
        """查询结果应包含原始文本。"""
        store.upsert(sample_records)
        results = store.query(vector=[1.0, 0.0, 0.0], top_k=1)
        assert results[0].text == "这是第一个文档"

    def test_query_result_contains_metadata(self, store, sample_records):
        """查询结果应包含元数据。"""
        store.upsert(sample_records)
        results = store.query(vector=[1.0, 0.0, 0.0], top_k=1)
        assert results[0].metadata["source"] == "doc1.pdf"
        assert results[0].metadata["page"] == 1


@pytest.mark.integration
class TestChromaStoreMetadataFilter:
    """测试 ChromaStore 的元数据过滤功能。"""

    def test_filter_by_single_field(self, store, sample_records):
        """应支持按单个字段过滤。"""
        store.upsert(sample_records)
        results = store.query(
            vector=[1.0, 0.0, 0.0],
            top_k=10,
            filters={"source": "doc1.pdf"},
        )
        # 只有 doc_001 和 doc_003 的 source 是 doc1.pdf
        assert len(results) == 2
        for r in results:
            assert r.metadata["source"] == "doc1.pdf"

    def test_filter_by_multiple_fields(self, store, sample_records):
        """应支持按多个字段过滤。"""
        store.upsert(sample_records)
        results = store.query(
            vector=[1.0, 0.0, 0.0],
            top_k=10,
            filters={"source": "doc1.pdf", "page": 1},
        )
        # 只有 doc_001 同时满足两个条件
        assert len(results) == 1
        assert results[0].id == "doc_001"


@pytest.mark.integration
class TestChromaStoreDelete:
    """测试 ChromaStore 的删除功能。"""

    def test_delete_removes_records(self, store, sample_records):
        """delete 应删除指定 ID 的记录。"""
        store.upsert(sample_records)
        assert store.count() == 3

        deleted = store.delete(["doc_001", "doc_002"])
        assert deleted == 2
        assert store.count() == 1

    def test_delete_nonexistent_id(self, store, sample_records):
        """删除不存在的 ID 不应报错，且不应计入删除数。"""
        store.upsert(sample_records)
        deleted = store.delete(["nonexistent_id"])
        assert deleted == 0  # 不存在的 ID 不应被计入删除数
        assert store.count() == 3

    def test_delete_empty_list(self, store):
        """空列表删除应返回 0。"""
        assert store.delete([]) == 0


@pytest.mark.integration
class TestChromaStoreIdempotency:
    """测试 ChromaStore 的幂等性。"""

    def test_upsert_same_id_twice(self, store):
        """相同 ID 重复 upsert 应更新而非重复插入。"""
        record = VectorRecord(
            id="doc_same",
            vector=[1.0, 0.0, 0.0],
            text="原始文本",
            metadata={"version": 1},
        )
        store.upsert([record])
        assert store.count() == 1

        # 更新同一条记录
        record_updated = VectorRecord(
            id="doc_same",
            vector=[1.0, 0.0, 0.0],
            text="更新后的文本",
            metadata={"version": 2},
        )
        store.upsert([record_updated])
        assert store.count() == 1

        # 查询应返回更新后的内容
        results = store.query(vector=[1.0, 0.0, 0.0], top_k=1)
        assert results[0].text == "更新后的文本"
        assert results[0].metadata["version"] == 2

    def test_upsert_preserves_other_records(self, store, sample_records):
        """upsert 新记录不应影响已有记录。"""
        store.upsert(sample_records)
        assert store.count() == 3

        # 添加新记录
        new_record = VectorRecord(
            id="doc_new",
            vector=[0.5, 0.5, 0.0],
            text="新文档",
            metadata={"source": "new.pdf"},
        )
        store.upsert([new_record])
        assert store.count() == 4


@pytest.mark.integration
class TestChromaStoreGetByIds:
    """测试 ChromaStore 的 get_by_ids 功能。"""

    def test_get_by_ids_returns_records(self, store, sample_records):
        """get_by_ids 应返回对应记录。"""
        store.upsert(sample_records)
        records = store.get_by_ids(["doc_001", "doc_003"])
        assert len(records) == 2
        ids = {r["id"] for r in records}
        assert ids == {"doc_001", "doc_003"}

    def test_get_by_ids_contains_text_and_metadata(self, store, sample_records):
        """get_by_ids 返回的记录应包含 text 和 metadata。"""
        store.upsert(sample_records)
        records = store.get_by_ids(["doc_001"])
        assert records[0]["text"] == "这是第一个文档"
        assert records[0]["metadata"]["source"] == "doc1.pdf"

    def test_get_by_ids_empty_list(self, store):
        """空列表应返回空结果。"""
        assert store.get_by_ids([]) == []


@pytest.mark.integration
class TestChromaStorePersistence:
    """测试 ChromaStore 的持久化功能。"""

    def test_data_persists_after_reopen(self, tmp_chroma_dir, sample_records):
        """关闭后重新打开应保留数据。"""
        # 第一次写入
        store1 = ChromaStore(
            collection_name="persist_test",
            persist_directory=tmp_chroma_dir,
        )
        store1.upsert(sample_records)
        assert store1.count() == 3

        # 重新打开
        store2 = ChromaStore(
            collection_name="persist_test",
            persist_directory=tmp_chroma_dir,
        )
        assert store2.count() == 3

        # 查询结果应一致
        results = store2.query(vector=[1.0, 0.0, 0.0], top_k=1)
        assert results[0].id == "doc_001"

    def test_different_collections_isolated(self, tmp_chroma_dir):
        """不同 collection 应相互隔离。"""
        store_a = ChromaStore(
            collection_name="collection_a",
            persist_directory=tmp_chroma_dir,
        )
        record_a = VectorRecord(id="a1", vector=[1.0, 0.0], text="A文档", metadata={})
        store_a.upsert([record_a])
        assert store_a.count() == 1
        
        # 显式删除和垃圾回收以释放 SQLite 文件锁
        del store_a
        import gc
        gc.collect()

        store_b = ChromaStore(
            collection_name="collection_b",
            persist_directory=tmp_chroma_dir,
        )
        record_b = VectorRecord(id="b1", vector=[1.0, 0.0], text="B文档", metadata={})
        store_b.upsert([record_b])
        assert store_b.count() == 1

        del store_b
        gc.collect()

        # 重新加载 A 验证隔离性
        store_a = ChromaStore(
            collection_name="collection_a",
            persist_directory=tmp_chroma_dir,
        )
        assert store_a.count() == 1
        results_a = store_a.query(vector=[1.0, 0.0], top_k=10)
        assert len(results_a) == 1
        assert results_a[0].text == "A文档"


@pytest.mark.integration
class TestChromaStoreFactory:
    """测试 ChromaStore 通过工厂创建。"""

    def test_factory_creates_chroma(self, tmp_chroma_dir):
        """VectorStoreFactory 应能创建 ChromaStore。"""
        config = VectorStoreConfig(
            provider="chroma",
            persist_directory=tmp_chroma_dir,
        )
        store = VectorStoreFactory.create(config)
        assert isinstance(store, ChromaStore)

    def test_factory_chroma_in_providers(self):
        """chroma 应在可用提供者列表中。"""
        providers = VectorStoreFactory.list_providers()
        assert "chroma" in providers


@pytest.mark.integration
class TestChromaStoreEdgeCases:
    """测试 ChromaStore 边界情况。"""

    def test_upsert_empty_list(self, store):
        """空列表 upsert 应返回 0。"""
        assert store.upsert([]) == 0

    def test_query_empty_collection(self, store):
        """空集合查询应返回空列表。"""
        results = store.query(vector=[1.0, 0.0, 0.0], top_k=10)
        assert results == []

    def test_count_empty_collection(self, store):
        """空集合 count 应返回 0。"""
        assert store.count() == 0

    def test_upsert_large_batch(self, store):
        """大批量 upsert 应正常工作。"""
        records = [
            VectorRecord(
                id=f"batch_{i:04d}",
                vector=[float(i % 3 == 0), float(i % 3 == 1), float(i % 3 == 2)],
                text=f"批量文档 {i}",
                metadata={"index": i},
            )
            for i in range(100)
        ]
        count = store.upsert(records)
        assert count == 100
        assert store.count() == 100

    def test_high_dimensional_vectors(self, store):
        """高维向量应正常存储和查询。"""
        dim = 1024
        records = [
            VectorRecord(
                id=f"high_dim_{i}",
                vector=[float(j == i) for j in range(dim)],
                text=f"高维文档 {i}",
                metadata={},
            )
            for i in range(5)
        ]
        store.upsert(records)

        query_vector = [float(j == 0) for j in range(dim)]
        results = store.query(vector=query_vector, top_k=1)
        assert results[0].id == "high_dim_0"
