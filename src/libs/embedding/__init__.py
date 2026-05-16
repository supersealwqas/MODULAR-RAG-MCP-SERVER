"""Embedding 模块。

提供可插拔的向量编码后端：
- BGEEmbedding: BGE-M3 本地模型
"""

from src.libs.embedding.bge_embedding import BGEEmbedding

__all__ = ["BGEEmbedding"]
