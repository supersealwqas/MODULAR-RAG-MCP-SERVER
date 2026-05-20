"""DocumentManager 文档生命周期管理模块。

跨 ChromaDB、BM25、ImageStorage、FileIntegrity 四个存储后端协调文档的
列表查询、详情获取、删除操作与统计汇总。

供 Dashboard 数据浏览器与 CLI 管理命令调用。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.libs.loader.file_integrity import FileIntegrityChecker
from src.libs.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


# ============================================================
# 数据类型
# ============================================================


@dataclass
class DocumentInfo:
    """文档摘要信息（用于列表展示）。

    属性:
        source_path: 文档源文件路径
        file_hash: 文件 SHA256 哈希（完整）
        collection: 所属集合名称
        chunk_count: 该文档的 chunk 数量
        image_count: 该文档的图片数量
        processed_at: 处理完成时间
        file_size: 文件大小（字节）
    """

    source_path: str
    file_hash: str
    collection: str
    chunk_count: int
    image_count: int
    processed_at: str
    file_size: int = 0


@dataclass
class DocumentDetail:
    """文档详情（用于详情展示）。

    属性:
        info: 文档摘要信息
        chunks: chunk 列表（每个包含 id、text、metadata）
        images: 图片列表（每个包含 image_id、file_path 等）
    """

    info: DocumentInfo
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DeleteResult:
    """删除操作结果。

    属性:
        source_path: 被删除文档的路径
        chunks_deleted: 从 ChromaDB 删除的 chunk 数
        bm25_deleted: 从 BM25 索引删除的 chunk 数
        images_deleted: 删除的图片数
        integrity_removed: 是否移除了完整性记录
        success: 是否全部成功
        errors: 错误信息列表
    """

    source_path: str
    chunks_deleted: int = 0
    bm25_deleted: int = 0
    images_deleted: int = 0
    integrity_removed: bool = False
    success: bool = True
    errors: List[str] = field(default_factory=list)


@dataclass
class CollectionStats:
    """集合统计信息。

    属性:
        collection: 集合名称
        document_count: 文档数量
        chunk_count: chunk 总数
        image_count: 图片总数
        total_file_size: 文件总大小（字节）
    """

    collection: str
    document_count: int
    chunk_count: int
    image_count: int
    total_file_size: int = 0


# ============================================================
# DocumentManager
# ============================================================


class DocumentManager:
    """文档生命周期管理器。

    协调四个存储后端（ChromaDB、BM25、ImageStorage、FileIntegrity）
    实现文档的统一查询与删除。

    属性:
        _settings: 全局配置对象
        _chroma_store: ChromaDB 向量存储实例（默认集合）
        _bm25_indexer: BM25 索引器实例
        _image_storage: 图片存储实例
        _file_integrity: 文件完整性检查器实例
    """

    def __init__(
        self,
        chroma_store: BaseVectorStore,
        bm25_indexer: BM25Indexer,
        image_storage: ImageStorage,
        file_integrity: FileIntegrityChecker,
        settings: Optional[Settings] = None,
    ) -> None:
        """初始化 DocumentManager。

        参数:
            chroma_store: ChromaDB 向量存储实例
            bm25_indexer: BM25 索引器实例
            image_storage: 图片存储实例
            file_integrity: 文件完整性检查器实例
            settings: 全局配置对象
        """
        self._chroma_store = chroma_store
        self._bm25_indexer = bm25_indexer
        self._image_storage = image_storage
        self._file_integrity = file_integrity
        self._settings = settings

    def _get_collection_store(self, collection_name: str) -> BaseVectorStore:
        """获取特定集合的 VectorStore。"""
        if self._chroma_store.collection_name == collection_name:
            return self._chroma_store
        
        if self._settings:
            from src.libs.vector_store.vector_store_factory import VectorStoreFactory
            return VectorStoreFactory.create(
                self._settings.vector_store,
                collection_name=collection_name
            )
        return self._chroma_store

    def list_documents(
        self,
        collection: Optional[str] = None,
    ) -> List[DocumentInfo]:
        """列出已摄入的文档列表。

        参数:
            collection: 可选的集合过滤

        返回:
            DocumentInfo 列表，按处理时间倒序排列
        """
        # 从 FileIntegrity 获取已处理文件列表
        processed = self._file_integrity.list_processed()
        if not processed:
            return []

        documents: List[DocumentInfo] = []
        for record in processed:
            source_path = record.get("file_path", "")
            file_hash = record.get("file_hash", "")
            processed_at = record.get("processed_at", "")
            file_size = record.get("file_size", 0)
            stored_chunk_count = record.get("chunk_count", 0)
            doc_collection = record.get("collection", "default")

            # 集合过滤
            if collection and doc_collection != collection:
                continue

            # 从 ChromaDB 查询该文档的 chunk 数量（使用正确的 collection）
            store = self._get_collection_store(doc_collection)
            try:
                records = store.get_by_metadata(filters={"source_path": source_path})
                chunk_count = len(records)
            except Exception:
                chunk_count = stored_chunk_count

            # 优先使用 ChromaDB 统计，否则使用 FileIntegrity 记录的值
            if chunk_count == 0 and stored_chunk_count > 0:
                chunk_count = stored_chunk_count

            # 查询图片数量
            image_count = 0
            if file_hash:
                try:
                    images = self._image_storage.list_by_doc_hash(file_hash)
                    image_count = len(images)
                except Exception:
                    pass

            documents.append(DocumentInfo(
                source_path=source_path,
                file_hash=file_hash,
                collection=doc_collection,
                chunk_count=chunk_count,
                image_count=image_count,
                processed_at=str(processed_at),
                file_size=file_size,
            ))

        return documents

    def get_document_detail(
        self,
        source_path: str,
    ) -> Optional[DocumentDetail]:
        """获取文档详细信息。

        包含文档摘要、所有 chunk 内容与 metadata、关联图片列表。

        参数:
            source_path: 文档源文件路径

        返回:
            DocumentDetail 或 None（文档不存在时）
        """
        # 从 FileIntegrity 获取基本信息
        processed = self._file_integrity.list_processed()
        record = None
        for r in processed:
            if r.get("file_path") == source_path:
                record = r
                break

        if record is None:
            return None

        file_hash = record.get("file_hash", "")

        # 从 ChromaDB 查询 chunks
        chunk_count, doc_collection = self._query_document_chunks_info(source_path)
        chunks = self._query_document_chunks(source_path)

        # 从 ImageStorage 查询图片
        images: List[Dict[str, Any]] = []
        if file_hash:
            try:
                images = self._image_storage.list_by_doc_hash(file_hash)
            except Exception:
                pass

        info = DocumentInfo(
            source_path=source_path,
            file_hash=file_hash,
            collection=doc_collection or "default",
            chunk_count=chunk_count,
            image_count=len(images),
            processed_at=str(record.get("processed_at", "")),
            file_size=record.get("file_size", 0),
        )

        return DocumentDetail(info=info, chunks=chunks, images=images)

    def delete_document(
        self,
        source_path: str,
    ) -> DeleteResult:
        """删除文档及其关联的所有数据。

        参数:
            source_path: 文档源文件路径

        返回:
            DeleteResult 删除操作结果
        """
        result = DeleteResult(source_path=source_path)
        errors: List[str] = []

        # 0. 确定文档所属集合
        doc_collection = "default"
        file_hash = ""
        processed = self._file_integrity.list_processed()
        for r in processed:
            if r.get("file_path") == source_path:
                doc_collection = r.get("collection", "default")
                file_hash = r.get("file_hash", "")
                break

        # 1. 从 ChromaDB 查找并删除 chunks
        chunk_ids: List[str] = []
        try:
            store = self._get_collection_store(doc_collection)
            records = store.get_by_metadata(
                filters={"source_path": source_path},
            )
            chunk_ids = [r["id"] for r in records]
            if chunk_ids:
                deleted = store.delete(chunk_ids)
                result.chunks_deleted = deleted
                logger.info(
                    "ChromaDB [%s] 删除完成: %s, %d chunks",
                    doc_collection, source_path, deleted,
                )
        except Exception as e:
            msg = f"ChromaDB [{doc_collection}] 删除失败: {e}"
            errors.append(msg)
            logger.error(msg)

        # 2. 从 BM25 删除 chunks
        bm25_deleted = 0
        if chunk_ids:
            try:
                # 针对该 collection 加载索引
                self._bm25_indexer.load(collection=doc_collection)
                for chunk_id in chunk_ids:
                    if self._bm25_indexer.remove_document(chunk_id):
                        bm25_deleted += 1
                
                result.bm25_deleted = bm25_deleted
                self._bm25_indexer.save(collection=doc_collection)
                logger.info("BM25 [%s] 索引更新完成", doc_collection)
            except Exception as e:
                msg = f"BM25 [{doc_collection}] 删除失败: {e}"
                errors.append(msg)
                logger.error(msg)

        # 3. 从 ImageStorage 删除图片（需要 file_hash）
        file_hash = self._find_file_hash(source_path)
        if file_hash:
            try:
                images_deleted = self._image_storage.delete_by_doc_hash(file_hash)
                result.images_deleted = images_deleted
                logger.info(
                    "ImageStorage 删除完成: %s, %d 图片",
                    source_path, images_deleted,
                )
            except Exception as e:
                msg = f"ImageStorage 删除失败: {e}"
                errors.append(msg)
                logger.error(msg)

        # 4. 从 FileIntegrity 移除记录
        if file_hash:
            try:
                removed = self._file_integrity.remove_record(file_hash)
                result.integrity_removed = removed
                logger.info(
                    "FileIntegrity 记录移除: %s, removed=%s",
                    source_path, removed,
                )
            except Exception as e:
                msg = f"FileIntegrity 记录移除失败: {e}"
                errors.append(msg)
                logger.error(msg)

        result.errors = errors
        result.success = len(errors) == 0
        return result

    def get_collection_stats(
        self,
        collection: Optional[str] = None,
    ) -> CollectionStats:
        """获取集合统计信息。

        参数:
            collection: 集合名称（None 时统计全部）

        返回:
            CollectionStats 统计信息
        """
        documents = self.list_documents(collection=collection)
        total_chunks = sum(d.chunk_count for d in documents)
        total_images = sum(d.image_count for d in documents)
        total_size = sum(d.file_size for d in documents)

        return CollectionStats(
            collection=collection or "all",
            document_count=len(documents),
            chunk_count=total_chunks,
            image_count=total_images,
            total_file_size=total_size,
        )

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    def _query_document_chunks_info(
        self,
        source_path: str,
    ) -> tuple:
        """查询文档的 chunk 数量与集合名。

        参数:
            source_path: 文档源文件路径

        返回:
            (chunk_count, collection) 元组
        """
        try:
            records = self._chroma_store.get_by_metadata(
                filters={"source_path": source_path},
            )
            chunk_count = len(records)
            # 从第一个 chunk 的 metadata 获取 collection
            collection = ""
            if records:
                collection = records[0].get("metadata", {}).get("collection", "")
            return chunk_count, collection
        except Exception as e:
            logger.warning("查询文档 chunk 信息失败: %s - %s", source_path, e)
            return 0, ""

    def _query_document_chunks(
        self,
        source_path: str,
    ) -> List[Dict[str, Any]]:
        """查询文档的所有 chunk 详情。

        参数:
            source_path: 文档源文件路径

        返回:
            chunk 字典列表（包含 id、text、metadata）
        """
        try:
            return self._chroma_store.get_by_metadata(
                filters={"source_path": source_path},
                include_documents=True,
            )
        except Exception as e:
            logger.warning("查询文档 chunks 失败: %s - %s", source_path, e)
            return []

    def _find_file_hash(self, source_path: str) -> str:
        """根据 source_path 在 FileIntegrity 中查找 file_hash。

        参数:
            source_path: 文档源文件路径

        返回:
            file_hash 字符串（未找到时返回空字符串）
        """
        # 优先从 chunk metadata 获取 file_hash（更高效）
        try:
            records = self._chroma_store.get_by_metadata(
                filters={"source_path": source_path},
            )
            if records:
                file_hash = records[0].get("metadata", {}).get("file_hash", "")
                if file_hash:
                    return file_hash
        except Exception:
            pass

        # 降级：遍历 FileIntegrity 记录查找
        try:
            processed = self._file_integrity.list_processed()
            for record in processed:
                if record.get("file_path") == source_path:
                    return record.get("file_hash", "")
        except Exception:
            pass

        return ""
