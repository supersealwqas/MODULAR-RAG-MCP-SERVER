"""Ingestion Pipeline 模块。

提供完整摄取流程编排：integrity → load → split → transform → encode → store。
"""

from src.ingestion.document_manager import (
    CollectionStats,
    DeleteResult,
    DocumentDetail,
    DocumentInfo,
    DocumentManager,
)
from src.ingestion.pipeline import IngestionPipeline, PipelineError, PipelineResult

__all__ = [
    "CollectionStats",
    "DeleteResult",
    "DocumentDetail",
    "DocumentInfo",
    "DocumentManager",
    "IngestionPipeline",
    "PipelineError",
    "PipelineResult",
]
