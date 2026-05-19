"""DocumentManager 单元测试。

测试文档生命周期管理器的四大核心操作：
- list_documents: 文档列表查询
- get_document_detail: 文档详情获取
- delete_document: 跨存储协调删除
- get_collection_stats: 集合统计汇总

全部使用 mock 隔离外部依赖，无需真实数据库。
"""

import pytest
from unittest.mock import MagicMock, patch

from src.ingestion.document_manager import (
    CollectionStats,
    DeleteResult,
    DocumentDetail,
    DocumentInfo,
    DocumentManager,
)


@pytest.fixture
def mock_chroma():
    """Mock ChromaDB 向量存储。"""
    return MagicMock()


@pytest.fixture
def mock_bm25():
    """Mock BM25 索引器。"""
    indexer = MagicMock()
    indexer.remove_document = MagicMock(return_value=True)
    indexer.save = MagicMock()
    return indexer


@pytest.fixture
def mock_image_storage():
    """Mock 图片存储。"""
    storage = MagicMock()
    storage.list_by_doc_hash = MagicMock(return_value=[])
    storage.delete_by_doc_hash = MagicMock(return_value=0)
    return storage


@pytest.fixture
def mock_integrity():
    """Mock 文件完整性检查器。"""
    checker = MagicMock()
    checker.list_processed = MagicMock(return_value=[])
    checker.remove_record = MagicMock(return_value=True)
    return checker


@pytest.fixture
def manager(mock_chroma, mock_bm25, mock_image_storage, mock_integrity):
    """创建 DocumentManager 实例（注入 mock 依赖）。"""
    return DocumentManager(
        chroma_store=mock_chroma,
        bm25_indexer=mock_bm25,
        image_storage=mock_image_storage,
        file_integrity=mock_integrity,
    )


# ============================================================
# list_documents 测试
# ============================================================


class TestListDocuments:
    """list_documents 方法测试。"""

    def test_empty_store(self, manager, mock_integrity):
        """空存储返回空列表。"""
        mock_integrity.list_processed.return_value = []
        result = manager.list_documents()
        assert result == []

    def test_single_document(self, manager, mock_integrity, mock_chroma):
        """单文档返回正确信息。"""
        mock_integrity.list_processed.return_value = [
            {
                "file_hash": "abc123",
                "file_path": "/path/to/doc.pdf",
                "file_size": 1024,
                "processed_at": "2026-05-19 10:00:00",
                "chunk_count": 5,
            }
        ]
        mock_chroma.get_by_metadata.return_value = [
            {"id": "c1", "metadata": {"collection": "test"}},
            {"id": "c2", "metadata": {"collection": "test"}},
        ]

        result = manager.list_documents()

        assert len(result) == 1
        doc = result[0]
        assert doc.source_path == "/path/to/doc.pdf"
        assert doc.file_hash == "abc123"
        assert doc.chunk_count == 2  # ChromaDB 统计优先
        assert doc.collection == "test"
        assert doc.file_size == 1024

    def test_multiple_documents(self, manager, mock_integrity, mock_chroma):
        """多文档返回正确数量。"""
        mock_integrity.list_processed.return_value = [
            {
                "file_hash": "hash1",
                "file_path": "/path/doc1.pdf",
                "file_size": 100,
                "processed_at": "2026-05-19 10:00:00",
                "chunk_count": 3,
            },
            {
                "file_hash": "hash2",
                "file_path": "/path/doc2.pdf",
                "file_size": 200,
                "processed_at": "2026-05-19 11:00:00",
                "chunk_count": 7,
            },
        ]
        mock_chroma.get_by_metadata.return_value = [
            {"id": "c1", "metadata": {"collection": "default"}},
        ]

        result = manager.list_documents()
        assert len(result) == 2

    def test_fallback_to_stored_chunk_count(
        self, manager, mock_integrity, mock_chroma
    ):
        """ChromaDB 无数据时回退到 FileIntegrity 记录的 chunk_count。"""
        mock_integrity.list_processed.return_value = [
            {
                "file_hash": "abc",
                "file_path": "/path/doc.pdf",
                "file_size": 100,
                "processed_at": "2026-05-19",
                "chunk_count": 10,
            }
        ]
        # ChromaDB 返回空（可能数据被清理）
        mock_chroma.get_by_metadata.return_value = []

        result = manager.list_documents()
        assert len(result) == 1
        assert result[0].chunk_count == 10  # 使用 FileIntegrity 的值

    def test_with_image_count(self, manager, mock_integrity, mock_chroma, mock_image_storage):
        """正确统计图片数量。"""
        mock_integrity.list_processed.return_value = [
            {
                "file_hash": "abc",
                "file_path": "/path/doc.pdf",
                "file_size": 100,
                "processed_at": "2026-05-19",
                "chunk_count": 3,
            }
        ]
        mock_chroma.get_by_metadata.return_value = [
            {"id": "c1", "metadata": {"collection": "default"}},
        ]
        mock_image_storage.list_by_doc_hash.return_value = [
            {"image_id": "img1"},
            {"image_id": "img2"},
        ]

        result = manager.list_documents()
        assert result[0].image_count == 2

    def test_chroma_error_graceful(self, manager, mock_integrity, mock_chroma):
        """ChromaDB 异常时不崩溃，回退到 FileIntegrity 数据。"""
        mock_integrity.list_processed.return_value = [
            {
                "file_hash": "abc",
                "file_path": "/path/doc.pdf",
                "file_size": 100,
                "processed_at": "2026-05-19",
                "chunk_count": 5,
            }
        ]
        mock_chroma.get_by_metadata.side_effect = RuntimeError("连接失败")

        result = manager.list_documents()
        assert len(result) == 1
        assert result[0].chunk_count == 5


# ============================================================
# get_document_detail 测试
# ============================================================


class TestGetDocumentDetail:
    """get_document_detail 方法测试。"""

    def test_document_not_found(self, manager, mock_integrity):
        """文档不存在返回 None。"""
        mock_integrity.list_processed.return_value = []
        result = manager.get_document_detail("/nonexistent.pdf")
        assert result is None

    def test_document_found(self, manager, mock_integrity, mock_chroma):
        """文档存在返回详情。"""
        mock_integrity.list_processed.return_value = [
            {
                "file_hash": "abc",
                "file_path": "/path/doc.pdf",
                "file_size": 1024,
                "processed_at": "2026-05-19",
                "chunk_count": 2,
            }
        ]
        mock_chroma.get_by_metadata.side_effect = [
            # 第一次调用: _query_document_chunks_info
            [
                {"id": "c1", "metadata": {"collection": "test"}},
                {"id": "c2", "metadata": {"collection": "test"}},
            ],
            # 第二次调用: _query_document_chunks (include_documents=True)
            [
                {"id": "c1", "text": "chunk 1 text", "metadata": {"collection": "test"}},
                {"id": "c2", "text": "chunk 2 text", "metadata": {"collection": "test"}},
            ],
        ]

        result = manager.get_document_detail("/path/doc.pdf")

        assert result is not None
        assert isinstance(result, DocumentDetail)
        assert result.info.source_path == "/path/doc.pdf"
        assert result.info.chunk_count == 2
        assert len(result.chunks) == 2
        assert result.chunks[0]["text"] == "chunk 1 text"

    def test_with_images(self, manager, mock_integrity, mock_chroma, mock_image_storage):
        """详情包含图片信息。"""
        mock_integrity.list_processed.return_value = [
            {
                "file_hash": "abc",
                "file_path": "/path/doc.pdf",
                "file_size": 100,
                "processed_at": "2026-05-19",
                "chunk_count": 1,
            }
        ]
        mock_chroma.get_by_metadata.return_value = [
            {"id": "c1", "text": "text", "metadata": {"collection": "default"}},
        ]
        mock_image_storage.list_by_doc_hash.return_value = [
            {"image_id": "img1", "file_path": "/images/img1.png"},
        ]

        result = manager.get_document_detail("/path/doc.pdf")
        assert result is not None
        assert result.info.image_count == 1
        assert len(result.images) == 1


# ============================================================
# delete_document 测试
# ============================================================


class TestDeleteDocument:
    """delete_document 方法测试。"""

    def test_delete_success(
        self, manager, mock_chroma, mock_bm25, mock_image_storage, mock_integrity
    ):
        """完整删除流程：Chroma + BM25 + Image + Integrity。"""
        # ChromaDB 查找 chunks
        mock_chroma.get_by_metadata.return_value = [
            {"id": "c1", "metadata": {"file_hash": "abc"}},
            {"id": "c2", "metadata": {"file_hash": "abc"}},
        ]
        mock_chroma.delete.return_value = 2
        mock_bm25.remove_document.return_value = True
        mock_image_storage.delete_by_doc_hash.return_value = 3
        mock_integrity.remove_record.return_value = True

        result = manager.delete_document("/path/doc.pdf")

        assert result.success is True
        assert result.chunks_deleted == 2
        assert result.bm25_deleted == 2
        assert result.images_deleted == 3
        assert result.integrity_removed is True
        assert len(result.errors) == 0

        # 验证调用
        mock_chroma.delete.assert_called_once_with(["c1", "c2"])
        assert mock_bm25.remove_document.call_count == 2
        mock_bm25.save.assert_called_once()
        mock_image_storage.delete_by_doc_hash.assert_called_once_with("abc")
        mock_integrity.remove_record.assert_called_once_with("abc")

    def test_delete_no_chunks(self, manager, mock_chroma, mock_integrity):
        """文档无 chunk 时仍能正常删除。"""
        mock_chroma.get_by_metadata.return_value = []
        mock_integrity.list_processed.return_value = [
            {
                "file_hash": "abc",
                "file_path": "/path/doc.pdf",
                "file_size": 100,
                "processed_at": "2026-05-19",
                "chunk_count": 0,
            }
        ]

        result = manager.delete_document("/path/doc.pdf")

        assert result.success is True
        assert result.chunks_deleted == 0
        assert result.bm25_deleted == 0

    def test_delete_chroma_error_continues(
        self, manager, mock_chroma, mock_bm25, mock_integrity
    ):
        """ChromaDB 删除失败时继续其他存储的删除。"""
        mock_chroma.get_by_metadata.side_effect = RuntimeError("连接断开")
        mock_integrity.list_processed.return_value = [
            {
                "file_hash": "abc",
                "file_path": "/path/doc.pdf",
                "file_size": 100,
                "processed_at": "2026-05-19",
                "chunk_count": 3,
            }
        ]

        result = manager.delete_document("/path/doc.pdf")

        assert result.success is False
        assert len(result.errors) == 1
        assert "ChromaDB" in result.errors[0]

    def test_delete_bm25_partial_failure(
        self, manager, mock_chroma, mock_bm25, mock_integrity
    ):
        """BM25 部分 chunk 删除失败时记录成功数。"""
        mock_chroma.get_by_metadata.return_value = [
            {"id": "c1", "metadata": {"file_hash": "abc"}},
            {"id": "c2", "metadata": {"file_hash": "abc"}},
        ]
        mock_chroma.delete.return_value = 2
        # 第一个成功，第二个失败
        mock_bm25.remove_document.side_effect = [True, False]
        mock_integrity.list_processed.return_value = [
            {
                "file_hash": "abc",
                "file_path": "/path/doc.pdf",
                "file_size": 100,
                "processed_at": "2026-05-19",
                "chunk_count": 2,
            }
        ]

        result = manager.delete_document("/path/doc.pdf")

        assert result.success is True  # BM25 部分失败不算整体失败
        assert result.bm25_deleted == 1

    def test_delete_image_storage_error(
        self, manager, mock_chroma, mock_image_storage, mock_integrity
    ):
        """ImageStorage 删除失败时记录错误但继续。"""
        mock_chroma.get_by_metadata.return_value = [
            {"id": "c1", "metadata": {"file_hash": "abc"}},
        ]
        mock_chroma.delete.return_value = 1
        mock_image_storage.delete_by_doc_hash.side_effect = RuntimeError("磁盘满")
        mock_integrity.list_processed.return_value = [
            {
                "file_hash": "abc",
                "file_path": "/path/doc.pdf",
                "file_size": 100,
                "processed_at": "2026-05-19",
                "chunk_count": 1,
            }
        ]

        result = manager.delete_document("/path/doc.pdf")

        assert result.success is False
        assert result.chunks_deleted == 1
        assert any("ImageStorage" in e for e in result.errors)


# ============================================================
# get_collection_stats 测试
# ============================================================


class TestGetCollectionStats:
    """get_collection_stats 方法测试。"""

    def test_empty_stats(self, manager, mock_integrity):
        """空集合返回零统计。"""
        mock_integrity.list_processed.return_value = []

        result = manager.get_collection_stats("test")

        assert result.collection == "test"
        assert result.document_count == 0
        assert result.chunk_count == 0
        assert result.image_count == 0

    def test_with_data(self, manager, mock_integrity, mock_chroma, mock_image_storage):
        """有数据时正确汇总。"""
        mock_integrity.list_processed.return_value = [
            {
                "file_hash": "h1",
                "file_path": "/doc1.pdf",
                "file_size": 100,
                "processed_at": "2026-05-19",
                "chunk_count": 5,
            },
            {
                "file_hash": "h2",
                "file_path": "/doc2.pdf",
                "file_size": 200,
                "processed_at": "2026-05-19",
                "chunk_count": 3,
            },
        ]
        mock_chroma.get_by_metadata.return_value = [
            {"id": "c1", "metadata": {"collection": "default"}},
        ]
        mock_image_storage.list_by_doc_hash.return_value = [{"image_id": "img1"}]

        result = manager.get_collection_stats()

        assert result.document_count == 2
        assert result.collection == "all"
        assert result.total_file_size == 300


# ============================================================
# get_by_metadata (ChromaStore) 测试
# ============================================================


class TestChromaGetByMetadata:
    """ChromaStore.get_by_metadata 方法测试。"""

    def test_empty_filters_raises(self):
        """空过滤条件抛出 ValueError。"""
        from src.libs.vector_store.chroma_store import ChromaStore

        store = ChromaStore.__new__(ChromaStore)
        store.collection_name = "test"
        store._client = None
        store._collection = None

        with pytest.raises(ValueError, match="不能为空"):
            store.get_by_metadata(filters={})

    def test_single_filter(self):
        """单条件过滤正确构建 where。"""
        from src.libs.vector_store.chroma_store import ChromaStore

        store = ChromaStore.__new__(ChromaStore)
        store.collection_name = "test"
        store._client = None

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["c1"],
            "metadatas": [{"source_path": "/doc.pdf"}],
        }
        store._collection = mock_collection

        result = store.get_by_metadata(
            filters={"source_path": "/doc.pdf"},
        )

        assert len(result) == 1
        assert result[0]["id"] == "c1"
        mock_collection.get.assert_called_once()
        call_kwargs = mock_collection.get.call_args
        assert call_kwargs[1]["where"] == {"source_path": "/doc.pdf"}

    def test_multiple_filters(self):
        """多条件过滤使用 $and 语法。"""
        from src.libs.vector_store.chroma_store import ChromaStore

        store = ChromaStore.__new__(ChromaStore)
        store.collection_name = "test"
        store._client = None

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": [],
            "metadatas": [],
        }
        store._collection = mock_collection

        store.get_by_metadata(
            filters={"source_path": "/doc.pdf", "collection": "test"},
        )

        call_kwargs = mock_collection.get.call_args
        where = call_kwargs[1]["where"]
        assert "$and" in where
        assert len(where["$and"]) == 2

    def test_include_documents(self):
        """include_documents=True 时返回 text 字段。"""
        from src.libs.vector_store.chroma_store import ChromaStore

        store = ChromaStore.__new__(ChromaStore)
        store.collection_name = "test"
        store._client = None

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["c1"],
            "metadatas": [{"source_path": "/doc.pdf"}],
            "documents": ["chunk text content"],
        }
        store._collection = mock_collection

        result = store.get_by_metadata(
            filters={"source_path": "/doc.pdf"},
            include_documents=True,
        )

        assert result[0]["text"] == "chunk text content"


# ============================================================
# DeleteResult / DocumentInfo 数据类测试
# ============================================================


class TestDataClasses:
    """数据类型测试。"""

    def test_delete_result_defaults(self):
        """DeleteResult 默认值正确。"""
        result = DeleteResult(source_path="/test.pdf")
        assert result.success is True
        assert result.chunks_deleted == 0
        assert result.errors == []

    def test_document_info_fields(self):
        """DocumentInfo 字段赋值正确。"""
        info = DocumentInfo(
            source_path="/test.pdf",
            file_hash="abc",
            collection="default",
            chunk_count=5,
            image_count=2,
            processed_at="2026-05-19",
            file_size=1024,
        )
        assert info.source_path == "/test.pdf"
        assert info.chunk_count == 5

    def test_collection_stats_fields(self):
        """CollectionStats 字段赋值正确。"""
        stats = CollectionStats(
            collection="test",
            document_count=3,
            chunk_count=15,
            image_count=5,
            total_file_size=4096,
        )
        assert stats.document_count == 3
        assert stats.chunk_count == 15
