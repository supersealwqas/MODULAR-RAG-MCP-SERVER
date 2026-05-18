"""Pipeline 进度回调单元测试。

验证 F5 阶段实现的 on_progress 回调功能：
- 回调被正确调用
- 回调参数正确
- 不传回调时无异常
- 跳过时不触发回调
"""

from __future__ import annotations

from typing import List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from src.core.settings import (
    EmbeddingConfig,
    EvaluationConfig,
    LLMConfig,
    ObservabilityConfig,
    OllamaConfig,
    RetrievalConfig,
    RerankConfig,
    Settings,
    SplitterConfig,
    VectorStoreConfig,
    VisionLLMConfig,
)
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk, ChunkRecord, Document
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.libs.loader.base_loader import BaseLoader


# ============================================================
# Fake 组件
# ============================================================


class FakeLoader(BaseLoader):
    """Fake 文件加载器。"""

    def load(self, path: str, collection: str = "default") -> Document:
        return Document(
            id="test_doc",
            text="这是测试文档内容。" * 30,
            metadata={"source_path": path, "doc_type": "pdf"},
        )


class FakeBatchProcessor:
    """Fake 批量编码器。"""

    def process(self, chunks, trace=None):
        return [
            ChunkRecord(
                id=c.id,
                text=c.text,
                metadata=c.metadata.copy(),
                dense_vector=[0.1, 0.2],
                sparse_vector={"test": 1.0},
            )
            for c in chunks
        ]


class FakeVectorUpserter:
    """Fake 向量写入器。"""

    def upsert(self, records, trace=None):
        return len(records)

    def delete(self, chunk_ids, trace=None):
        return len(chunk_ids)


class FakeBM25Indexer:
    """Fake BM25 索引器。"""

    def build(self, records, trace=None):
        pass

    def save(self, path=None):
        return "fake_path"

    def get_vocabulary_size(self):
        return 10


class FakeIntegrityChecker:
    """Fake 文件完整性检查器。"""

    def __init__(self):
        self._processed = {}

    def compute_hash(self, file_path):
        return f"hash_{file_path}"

    def should_skip(self, file_hash):
        return file_hash in self._processed

    def mark_success(self, file_hash, file_path, **kwargs):
        self._processed[file_hash] = "success"

    def mark_failed(self, file_hash, file_path, error_msg):
        self._processed[file_hash] = "failed"


# ============================================================
# 辅助工具
# ============================================================


def _make_settings() -> Settings:
    """创建最小测试配置。"""
    return Settings(
        llm=LLMConfig(provider="fake", model="fake"),
        vision_llm=VisionLLMConfig(),
        ollama=OllamaConfig(),
        embedding=EmbeddingConfig(provider="fake", model="fake", dimensions=2),
        splitter=SplitterConfig(strategy="recursive", chunk_size=500, chunk_overlap=50),
        vector_store=VectorStoreConfig(provider="fake"),
        retrieval=RetrievalConfig(),
        rerank=RerankConfig(),
        evaluation=EvaluationConfig(),
        observability=ObservabilityConfig(),
        pipeline=MagicMock(),
    )


def _make_pipeline() -> IngestionPipeline:
    """创建注入 Fake 组件的 Pipeline。"""
    settings = _make_settings()
    chunker = DocumentChunker(settings)
    refiner = ChunkRefiner(settings, use_llm=False)

    return IngestionPipeline(
        settings,
        integrity_checker=FakeIntegrityChecker(),
        loader=FakeLoader(),
        chunker=chunker,
        refiner=refiner,
        batch_processor=FakeBatchProcessor(),
        vector_upserter=FakeVectorUpserter(),
        bm25_indexer=FakeBM25Indexer(),
        compute_hash=lambda p: f"hash_{p}",
    )


# ============================================================
# 测试用例
# ============================================================


class TestProgressCallback:
    """进度回调测试。"""

    def test_callback_called_for_each_stage(self):
        """on_progress 回调应被每个阶段调用。"""
        pipeline = _make_pipeline()
        calls: List[Tuple[str, int, int]] = []

        def on_progress(stage: str, current: int, total: int):
            calls.append((stage, current, total))

        pipeline.run("test.pdf", on_progress=on_progress)

        assert len(calls) == 6
        stages = [c[0] for c in calls]
        assert stages == ["integrity", "load", "split", "transform", "encode", "store"]

    def test_callback_args_correct(self):
        """回调参数应正确：current 递增，total 固定为 6。"""
        pipeline = _make_pipeline()
        calls: List[dict] = []

        def on_progress(stage: str, current: int, total: int):
            calls.append({"stage": stage, "current": current, "total": total})

        pipeline.run("test.pdf", on_progress=on_progress)

        for i, call in enumerate(calls):
            assert call["current"] == i + 1
            assert call["total"] == 6

    def test_callback_none_no_error(self):
        """不传回调时不应报错。"""
        pipeline = _make_pipeline()

        result = pipeline.run("test.pdf")
        assert result.skipped is False

    def test_callback_integrity_called_on_skip(self):
        """文件被跳过时仍会调用 integrity 回调（检查在回调之后）。"""
        pipeline = _make_pipeline()
        calls: List[str] = []

        def on_progress(stage: str, current: int, total: int):
            calls.append(stage)

        # 第一次运行
        pipeline.run("test.pdf", on_progress=on_progress)
        assert len(calls) == 6

        # 第二次运行（跳过，但 integrity 回调仍被调用）
        calls.clear()
        pipeline.run("test.pdf", on_progress=on_progress)
        assert calls == ["integrity"]

    def test_callback_stage_order(self):
        """回调阶段顺序应为 integrity → load → split → transform → encode → store。"""
        pipeline = _make_pipeline()
        stages: List[str] = []

        def on_progress(stage: str, current: int, total: int):
            stages.append(stage)

        pipeline.run("test.pdf", on_progress=on_progress)

        expected = ["integrity", "load", "split", "transform", "encode", "store"]
        assert stages == expected
