"""测试 VectorStore 抽象接口、数据类和工厂的契约。"""

import pytest
from typing import Any, Dict, List, Optional

from src.core.settings import VectorStoreConfig
from src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorRecord,
    QueryResult,
)
from src.libs.vector_store.vector_store_factory import (
    VectorStoreFactory,
    register_vector_store,
    _VECTOR_STORE_REGISTRY,
)


# --- Fake VectorStore 用于测试 ---

class FakeVectorStore(BaseVectorStore):
    """测试用的假 VectorStore 实现，使用内存列表存储。"""

    def __init__(self, collection_name: str = "default", persist_directory: str = "", **kwargs):
        super().__init__(collection_name, **kwargs)
        self.persist_directory = persist_directory
        self._records: Dict[str, VectorRecord] = {}

    def upsert(self, records: List[VectorRecord], **kwargs) -> int:
        """将记录存入内存字典。"""
        for record in records:
            self._records[record.id] = record
        return len(records)

    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[QueryResult]:
        """返回所有记录（忽略向量相似度计算，仅用于测试）。"""
        results = []
        for record in self._records.values():
            # 应用过滤条件
            if filters:
                match = all(record.metadata.get(k) == v for k, v in filters.items())
                if not match:
                    continue
            results.append(QueryResult(
                id=record.id,
                score=1.0,  # 固定分数
                text=record.text,
                metadata=record.metadata,
            ))
        return results[:top_k]

    def delete(self, ids: List[str], **kwargs) -> int:
        """删除指定 ID 的记录。"""
        deleted = 0
        for id in ids:
            if id in self._records:
                del self._records[id]
                deleted += 1
        return deleted

    def get_by_ids(self, ids: List[str], **kwargs) -> List[Dict[str, Any]]:
        """根据 ID 批量获取记录。"""
        results = []
        for id_ in ids:
            record = self._records.get(id_)
            if record:
                results.append({
                    "id": record.id,
                    "text": record.text,
                    "metadata": record.metadata,
                })
        return results

    def get_by_metadata(
        self,
        filters: Dict[str, Any],
        include_documents: bool = False,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """根据元数据过滤条件查询记录。"""
        if not filters:
            raise ValueError("过滤条件不能为空")
            
        results = []
        for record in self._records.values():
            match = all(record.metadata.get(k) == v for k, v in filters.items())
            if match:
                res = {
                    "id": record.id,
                    "metadata": record.metadata,
                }
                if include_documents:
                    res["text"] = record.text
                results.append(res)
        return results

    def count(self, **kwargs) -> int:
        """返回记录总数。"""
        return len(self._records)


# --- 数据类契约测试 ---

@pytest.mark.unit
class TestVectorRecord:
    """测试 VectorRecord 数据类。"""

    def test_create_record(self):
        """应能创建包含所有字段的记录。"""
        record = VectorRecord(
            id="doc1",
            vector=[0.1, 0.2, 0.3],
            text="Hello world",
            metadata={"source": "test.pdf", "page": 1},
        )
        assert record.id == "doc1"
        assert len(record.vector) == 3
        assert record.text == "Hello world"
        assert record.metadata["source"] == "test.pdf"

    def test_default_metadata(self):
        """元数据应有空字典默认值。"""
        record = VectorRecord(id="1", vector=[0.0], text="test")
        assert record.metadata == {}

    def test_record_fields_types(self):
        """字段类型应符合契约。"""
        record = VectorRecord(id="1", vector=[1.0], text="test")
        assert isinstance(record.id, str)
        assert isinstance(record.vector, list)
        assert isinstance(record.text, str)
        assert isinstance(record.metadata, dict)


@pytest.mark.unit
class TestQueryResult:
    """测试 QueryResult 数据类。"""

    def test_create_result(self):
        """应能创建包含所有字段的结果。"""
        result = QueryResult(
            id="doc1",
            score=0.95,
            text="Hello world",
            metadata={"source": "test.pdf"},
        )
        assert result.id == "doc1"
        assert result.score == 0.95
        assert result.text == "Hello world"

    def test_default_metadata(self):
        """元数据应有空字典默认值。"""
        result = QueryResult(id="1", score=0.5, text="test")
        assert result.metadata == {}


# --- BaseVectorStore 契约测试 ---

@pytest.mark.unit
class TestBaseVectorStore:
    """测试 BaseVectorStore 抽象类。"""

    def test_cannot_instantiate_abstract(self):
        """BaseVectorStore 是抽象类，不能直接实例化。"""
        with pytest.raises(TypeError):
            BaseVectorStore()

    def test_fake_store_upsert(self):
        """FakeVectorStore 应正确实现 upsert。"""
        store = FakeVectorStore()
        records = [
            VectorRecord(id="1", vector=[0.1], text="hello"),
            VectorRecord(id="2", vector=[0.2], text="world"),
        ]
        count = store.upsert(records)
        assert count == 2
        assert store.count() == 2

    def test_fake_store_query(self):
        """FakeVectorStore 应正确实现 query。"""
        store = FakeVectorStore()
        store.upsert([
            VectorRecord(id="1", vector=[0.1], text="hello", metadata={"type": "greeting"}),
            VectorRecord(id="2", vector=[0.2], text="world", metadata={"type": "noun"}),
        ])
        results = store.query(vector=[0.1], top_k=10)
        assert len(results) == 2
        assert isinstance(results[0], QueryResult)

    def test_query_with_filters(self):
        """查询应支持元数据过滤。"""
        store = FakeVectorStore()
        store.upsert([
            VectorRecord(id="1", vector=[0.1], text="hello", metadata={"type": "greeting"}),
            VectorRecord(id="2", vector=[0.2], text="world", metadata={"type": "noun"}),
        ])
        results = store.query(vector=[0.1], filters={"type": "greeting"})
        assert len(results) == 1
        assert results[0].text == "hello"

    def test_query_top_k(self):
        """查询应限制返回数量。"""
        store = FakeVectorStore()
        store.upsert([
            VectorRecord(id=str(i), vector=[float(i)], text=f"text{i}")
            for i in range(10)
        ])
        results = store.query(vector=[0.0], top_k=3)
        assert len(results) == 3

    def test_delete(self):
        """delete 应正确删除记录。"""
        store = FakeVectorStore()
        store.upsert([
            VectorRecord(id="1", vector=[0.1], text="hello"),
            VectorRecord(id="2", vector=[0.2], text="world"),
        ])
        deleted = store.delete(["1"])
        assert deleted == 1
        assert store.count() == 1

    def test_delete_by_metadata(self):
        """delete_by_metadata 应正确按条件删除记录。"""
        store = FakeVectorStore()
        store.upsert([
            VectorRecord(id="1", vector=[0.1], text="doc1", metadata={"source": "A", "type": "pdf"}),
            VectorRecord(id="2", vector=[0.2], text="doc2", metadata={"source": "B", "type": "pdf"}),
            VectorRecord(id="3", vector=[0.3], text="doc3", metadata={"source": "A", "type": "txt"}),
        ])
        
        # 删除 source=A 的记录
        deleted = store.delete_by_metadata({"source": "A"})
        assert deleted == 2
        assert store.count() == 1
        assert store.get_by_ids(["2"])  # 剩下 2
        assert not store.get_by_ids(["1", "3"])
        
    def test_delete_by_metadata_no_match(self):
        """delete_by_metadata 没有匹配时返回 0。"""
        store = FakeVectorStore()
        store.upsert([VectorRecord(id="1", vector=[0.1], text="doc1", metadata={"source": "A"})])
        
        deleted = store.delete_by_metadata({"source": "B"})
        assert deleted == 0
        assert store.count() == 1
        
    def test_delete_by_metadata_empty_filters(self):
        """delete_by_metadata 过滤条件为空时抛出 ValueError。"""
        store = FakeVectorStore()
        with pytest.raises(ValueError, match="过滤条件不能为空"):
            store.delete_by_metadata({})


# --- VectorStoreFactory 测试 ---

@pytest.mark.unit
class TestVectorStoreFactory:
    """测试 VectorStoreFactory 的路由逻辑。"""

    def setup_method(self):
        """每个测试前注册 fake 提供者。"""
        _VECTOR_STORE_REGISTRY["fake"] = FakeVectorStore

    def teardown_method(self):
        """每个测试后清理注册表。"""
        _VECTOR_STORE_REGISTRY.pop("fake", None)

    def test_create_registered_provider(self):
        """应为已注册的提供者创建实例。"""
        config = VectorStoreConfig(provider="fake", persist_directory="/tmp/db")
        store = VectorStoreFactory.create(config)
        assert isinstance(store, FakeVectorStore)
        assert store.persist_directory == "/tmp/db"

    def test_unknown_provider_raises(self):
        """未注册的提供者应抛出 ValueError。"""
        config = VectorStoreConfig(provider="unknown")
        with pytest.raises(ValueError, match="未知的向量存储提供者.*unknown"):
            VectorStoreFactory.create(config)

    def test_case_insensitive_provider(self):
        """提供者名称匹配应不区分大小写。"""
        config = VectorStoreConfig(provider="FAKE")
        store = VectorStoreFactory.create(config)
        assert isinstance(store, FakeVectorStore)

    def test_list_providers(self):
        """应列出所有已注册的提供者。"""
        providers = VectorStoreFactory.list_providers()
        assert "fake" in providers


@pytest.mark.unit
class TestRegisterVectorStoreDecorator:
    """测试 @register_vector_store 装饰器。"""

    def teardown_method(self):
        """清理注册表。"""
        _VECTOR_STORE_REGISTRY.pop("test_provider", None)

    def test_register_decorator(self):
        """@register_vector_store 应将类添加到注册表。"""
        @register_vector_store("test_provider")
        class TestStore(BaseVectorStore):
            def upsert(self, records, **kwargs):
                return len(records)
            def query(self, vector, top_k=10, filters=None, **kwargs):
                return []

        assert "test_provider" in _VECTOR_STORE_REGISTRY
        assert _VECTOR_STORE_REGISTRY["test_provider"] is TestStore

    def test_register_lowercase(self):
        """提供者名称应存储为小写。"""
        @register_vector_store("TEST_UPPER")
        class UpperStore(BaseVectorStore):
            def upsert(self, records, **kwargs):
                return len(records)
            def query(self, vector, top_k=10, filters=None, **kwargs):
                return []

        assert "test_upper" in _VECTOR_STORE_REGISTRY
        _VECTOR_STORE_REGISTRY.pop("test_upper", None)
