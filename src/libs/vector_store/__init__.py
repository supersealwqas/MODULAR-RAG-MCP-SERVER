"""向量存储模块。

提供可插拔的向量存储后端：
- ChromaStore: ChromaDB 本地持久化存储
"""

from src.libs.vector_store.chroma_store import ChromaStore

__all__ = ["ChromaStore"]
