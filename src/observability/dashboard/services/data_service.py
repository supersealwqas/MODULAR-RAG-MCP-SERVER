"""DataService 数据浏览服务。

封装 DocumentManager 与底层存储，为 Dashboard 数据浏览器页面
提供文档列表、详情、集合筛选等查询接口。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.settings import load_settings
from src.ingestion.document_manager import (
    CollectionStats,
    DocumentDetail,
    DocumentInfo,
    DocumentManager,
)
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.libs.vector_store.chroma_store import ChromaStore


class DataService:
    """数据浏览服务，封装 DocumentManager 的创建与调用。

    懒初始化 DocumentManager 及其四个依赖后端，
    供 Dashboard 页面通过简单方法调用获取数据。
    """

    def __init__(self) -> None:
        """初始化 DataService（延迟创建依赖）。"""
        self._document_manager: Optional[DocumentManager] = None

    def _get_document_manager(self) -> DocumentManager:
        """获取或创建 DocumentManager 实例。

        返回:
            DocumentManager 实例
        """
        if self._document_manager is None:
            settings = load_settings()
            persist_dir = settings.vector_store.persist_directory

            chroma_store = ChromaStore(
                persist_directory=persist_dir,
            )
            bm25_indexer = BM25Indexer()
            image_storage = ImageStorage()
            file_integrity = SQLiteIntegrityChecker()

            self._document_manager = DocumentManager(
                chroma_store=chroma_store,
                bm25_indexer=bm25_indexer,
                image_storage=image_storage,
                file_integrity=file_integrity,
            )
        return self._document_manager

    def list_documents(
        self,
        collection: Optional[str] = None,
    ) -> List[DocumentInfo]:
        """列出已摄入的文档。

        参数:
            collection: 可选的集合过滤

        返回:
            DocumentInfo 列表
        """
        manager = self._get_document_manager()
        return manager.list_documents(collection=collection)

    def get_document_detail(
        self,
        source_path: str,
    ) -> Optional[DocumentDetail]:
        """获取文档详情。

        参数:
            source_path: 文档源文件路径

        返回:
            DocumentDetail 或 None
        """
        manager = self._get_document_manager()
        return manager.get_document_detail(source_path)

    def get_collection_stats(
        self,
        collection: Optional[str] = None,
    ) -> CollectionStats:
        """获取集合统计。

        参数:
            collection: 可选的集合名称

        返回:
            CollectionStats 统计信息
        """
        manager = self._get_document_manager()
        return manager.get_collection_stats(collection=collection)

    def list_collections(self) -> List[str]:
        """列出所有集合名称。

        返回:
            集合名称列表
        """
        manager = self._get_document_manager()
        docs = manager.list_documents()
        collections = sorted(set(d.collection for d in docs))
        return collections
