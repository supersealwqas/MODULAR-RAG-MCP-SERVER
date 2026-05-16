"""Ingestion Chunking 模块。

提供 DocumentChunker 适配器，将 libs.splitter 的纯文本切分结果
转换为符合 core.types 契约的 Chunk 业务对象。
"""

from src.ingestion.chunking.document_chunker import DocumentChunker

__all__ = ["DocumentChunker"]
