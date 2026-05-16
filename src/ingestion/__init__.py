"""Ingestion Pipeline 模块。

提供完整摄取流程编排：integrity → load → split → transform → encode → store。
"""

from src.ingestion.pipeline import IngestionPipeline, PipelineError, PipelineResult

__all__ = ["IngestionPipeline", "PipelineError", "PipelineResult"]
