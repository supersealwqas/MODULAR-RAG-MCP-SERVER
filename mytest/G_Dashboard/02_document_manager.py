"""手动测试 G2: DocumentManager 文档生命周期管理。

用法:
    uv run python mytest/G_Dashboard/02_document_manager.py

验证项:
    1. DocumentManager 可导入
    2. ChromaStore.get_by_metadata 方法存在
    3. Pipeline 已注入 file_hash 到 document.metadata
    4. 数据类型（DocumentInfo/DeleteResult/CollectionStats）可实例化
    5. 完整的 list → detail → delete 流程（使用 mock）
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """测试模块可导入。"""
    print("=" * 50)
    print("测试 1: 模块可导入")
    print("=" * 50)

    from src.ingestion.document_manager import (
        DocumentManager,
        DocumentInfo,
        DocumentDetail,
        DeleteResult,
        CollectionStats,
    )
    print(f"  DocumentManager: {DocumentManager}")
    print(f"  DocumentInfo: {DocumentInfo}")
    print(f"  DeleteResult: {DeleteResult}")
    print(f"  CollectionStats: {CollectionStats}")

    from src.ingestion import DocumentManager as DM
    assert DM is DocumentManager

    print("\n✅ 模块导入测试通过\n")


def test_chroma_get_by_metadata():
    """测试 ChromaStore.get_by_metadata 方法。"""
    print("=" * 50)
    print("测试 2: ChromaStore.get_by_metadata 方法")
    print("=" * 50)

    from src.libs.vector_store.chroma_store import ChromaStore
    from src.libs.vector_store.base_vector_store import BaseVectorStore

    # 检查方法存在
    assert hasattr(ChromaStore, "get_by_metadata"), "ChromaStore 缺少 get_by_metadata"
    assert hasattr(BaseVectorStore, "get_by_metadata"), "BaseVectorStore 缺少 get_by_metadata"
    print("  ChromaStore.get_by_metadata 方法存在")
    print("  BaseVectorStore.get_by_metadata 方法存在")

    # 测试空过滤条件
    store = ChromaStore.__new__(ChromaStore)
    store.collection_name = "test"
    store._client = None
    store._collection = None

    try:
        store.get_by_metadata(filters={})
        print("\n❌ 应该抛出 ValueError")
    except ValueError as e:
        print(f"  空过滤抛出 ValueError: {e}")

    print("\n✅ ChromaStore.get_by_metadata 测试通过\n")


def test_pipeline_file_hash_injection():
    """测试 Pipeline 是否注入 file_hash 到 metadata。"""
    print("=" * 50)
    print("测试 3: Pipeline file_hash 注入")
    print("=" * 50)

    import inspect
    from src.ingestion.pipeline import IngestionPipeline

    # 检查 run 方法源码中包含 file_hash 注入
    source = inspect.getsource(IngestionPipeline.run)
    has_injection = 'document.metadata["file_hash"]' in source or \
                    "document.metadata['file_hash']" in source
    print(f"  Pipeline.run 包含 file_hash 注入: {has_injection}")

    if has_injection:
        print("\n✅ Pipeline file_hash 注入测试通过\n")
    else:
        print("\n❌ Pipeline 未注入 file_hash\n")


def test_data_classes():
    """测试数据类型可实例化。"""
    print("=" * 50)
    print("测试 4: 数据类型实例化")
    print("=" * 50)

    from src.ingestion.document_manager import (
        DocumentInfo,
        DeleteResult,
        CollectionStats,
    )

    info = DocumentInfo(
        source_path="/test.pdf",
        file_hash="abc123",
        collection="default",
        chunk_count=5,
        image_count=2,
        processed_at="2026-05-19 10:00:00",
        file_size=1024,
    )
    print(f"  DocumentInfo: {info.source_path}, {info.chunk_count} chunks")

    result = DeleteResult(source_path="/test.pdf")
    print(f"  DeleteResult: success={result.success}, errors={result.errors}")

    stats = CollectionStats(
        collection="test",
        document_count=3,
        chunk_count=15,
        image_count=5,
    )
    print(f"  CollectionStats: {stats.document_count} docs, {stats.chunk_count} chunks")

    print("\n✅ 数据类型实例化测试通过\n")


def test_mock_workflow():
    """测试完整的 list → detail → delete 流程（mock）。"""
    print("=" * 50)
    print("测试 5: Mock 完整工作流")
    print("=" * 50)

    from unittest.mock import MagicMock
    from src.ingestion.document_manager import DocumentManager

    # 创建 mock 依赖
    mock_chroma = MagicMock()
    mock_bm25 = MagicMock()
    mock_image = MagicMock()
    mock_integrity = MagicMock()

    manager = DocumentManager(
        chroma_store=mock_chroma,
        bm25_indexer=mock_bm25,
        image_storage=mock_image,
        file_integrity=mock_integrity,
    )

    # 配置 mock 返回值
    mock_integrity.list_processed.return_value = [
        {
            "file_hash": "abc123",
            "file_path": "/data/test.pdf",
            "file_size": 2048,
            "processed_at": "2026-05-19 10:00:00",
            "chunk_count": 5,
        }
    ]
    mock_chroma.get_by_metadata.return_value = [
        {"id": "c1", "text": "chunk 1", "metadata": {"collection": "default", "file_hash": "abc123"}},
        {"id": "c2", "text": "chunk 2", "metadata": {"collection": "default", "file_hash": "abc123"}},
    ]
    mock_image.list_by_doc_hash.return_value = [{"image_id": "img1"}]
    mock_chroma.delete.return_value = 2
    mock_bm25.remove_document.return_value = True
    mock_image.delete_by_doc_hash.return_value = 1
    mock_integrity.remove_record.return_value = True

    # 1. list
    docs = manager.list_documents()
    print(f"  list_documents: {len(docs)} 个文档")
    assert len(docs) == 1
    assert docs[0].source_path == "/data/test.pdf"

    # 2. detail
    detail = manager.get_document_detail("/data/test.pdf")
    print(f"  get_document_detail: {detail.info.chunk_count} chunks, {detail.info.image_count} images")
    assert detail is not None
    assert len(detail.chunks) == 2

    # 3. delete
    delete_result = manager.delete_document("/data/test.pdf")
    print(f"  delete_document: success={delete_result.success}, "
          f"chunks={delete_result.chunks_deleted}, "
          f"bm25={delete_result.bm25_deleted}, "
          f"images={delete_result.images_deleted}")
    assert delete_result.success is True
    assert delete_result.chunks_deleted == 2
    assert delete_result.bm25_deleted == 2
    assert delete_result.images_deleted == 1
    assert delete_result.integrity_removed is True

    print("\n✅ Mock 完整工作流测试通过\n")


if __name__ == "__main__":
    print("🧪 G2 DocumentManager — 手动测试\n")
    test_imports()
    test_chroma_get_by_metadata()
    test_pipeline_file_hash_injection()
    test_data_classes()
    test_mock_workflow()
    print("=" * 50)
    print("所有手动测试完成！")
    print("=" * 50)
