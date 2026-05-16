"""Ingestion Embedding 模块。

提供稠密向量编码（DenseEncoder）、稀疏向量编码（SparseEncoder）
和批处理编排（BatchProcessor）。
"""

from src.ingestion.embedding.batch_processor import BatchProcessor
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder

__all__ = ["BatchProcessor", "DenseEncoder", "SparseEncoder"]
