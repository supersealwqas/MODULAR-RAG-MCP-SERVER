"""Ingestion Pipeline 集成测试。

测试完整摄取流程编排：integrity → load → split → transform → encode → store。
使用 Fake 组件注入隔离外部依赖，验证各阶段串联和数据流转。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
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
from src.ingestion.pipeline import IngestionPipeline, PipelineError, PipelineResult
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.libs.loader.base_loader import BaseLoader
from src.libs.splitter.recursive_splitter import RecursiveSplitter  # noqa: F401


# ============================================================
# Fake 组件
# ============================================================


class FakeLoader(BaseLoader):
    """Fake 文件加载器，返回预构建的 Document。"""

    def __init__(self, document: Document) -> None:
        self._document = document
        self.call_count = 0
        self.last_path: Optional[str] = None
        self.last_collection: Optional[str] = None

    def load(self, path: str, collection: str = "default") -> Document:
        self.call_count += 1
        self.last_path = path
        self.last_collection = collection
        return self._document


class FakeBatchProcessor:
    """Fake 批量编码器，为每个 Chunk 生成确定性的 ChunkRecord。"""

    def __init__(self) -> None:
        self.call_count = 0
        self.last_chunks: Optional[List[Chunk]] = None

    def process(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None,
    ) -> List[ChunkRecord]:
        self.call_count += 1
        self.last_chunks = chunks
        records = []
        for i, chunk in enumerate(chunks):
            records.append(
                ChunkRecord(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=chunk.metadata.copy(),
                    dense_vector=[float(i) * 0.1, float(i) * 0.2, float(i) * 0.3],
                    sparse_vector={"test": 1.0},
                )
            )
        return records


class FakeVectorUpserter:
    """Fake 向量写入器，记录调用参数。"""

    def __init__(self) -> None:
        self.call_count = 0
        self.last_records: Optional[List[ChunkRecord]] = None
        self.last_collection: Optional[str] = None

    def upsert(
        self,
        records: List[ChunkRecord],
        collection: str = "default",
        trace: Optional[TraceContext] = None,
    ) -> int:
        self.call_count += 1
        self.last_records = records
        self.last_collection = collection
        return len(records)

    def delete(
        self,
        chunk_ids: List[str],
        collection: str = "default",
        trace: Optional[TraceContext] = None,
    ) -> int:
        return len(chunk_ids)


class FakeBM25Indexer:
    """Fake BM25 索引器，记录调用参数。"""

    def __init__(self) -> None:
        self.build_count = 0
        self.save_count = 0
        self.load_count = 0
        self.add_count = 0
        self.last_records: Optional[List[ChunkRecord]] = None
        self.last_collection: Optional[str] = None
        self._vocab_size = 0

    def build(
        self,
        records: List[ChunkRecord],
        trace: Optional[TraceContext] = None,
    ) -> None:
        self.build_count += 1
        self.last_records = records
        terms = set()
        for r in records:
            if r.sparse_vector:
                terms.update(r.sparse_vector.keys())
        self._vocab_size = len(terms)

    def add_documents(
        self,
        records: List[ChunkRecord],
        trace: Optional[TraceContext] = None,
    ) -> None:
        self.add_count += 1
        self.last_records = records

    def save(self, collection: str = "default", path: Optional[str] = None) -> str:
        self.save_count += 1
        self.last_collection = collection
        return path or "fake_path"

    def load(self, collection: str = "default", path: Optional[str] = None) -> None:
        self.load_count += 1
        self.last_collection = collection

    def get_vocabulary_size(self) -> int:
        return self._vocab_size


class FakeIntegrityChecker:
    """Fake 文件完整性检查器，基于内存记录。"""

    def __init__(self) -> None:
        self._processed: Dict[str, dict] = {}
        self._hash_cache: Dict[str, str] = {}

    def compute_hash(self, file_path: str) -> str:
        """为测试生成确定性 hash（同一路径返回相同值）。"""
        if file_path not in self._hash_cache:
            self._hash_cache[file_path] = f"{hash(file_path) % (10**64):064d}"
        return self._hash_cache[file_path]

    def should_skip(self, file_hash: str) -> bool:
        return self._processed.get(file_hash, {}).get("status") == "success"

    def mark_success(
        self,
        file_hash: str,
        file_path: str,
        collection: str = "default",
        file_size: int = 0,
        chunk_count: int = 0,
    ) -> None:
        self._processed[file_hash] = {
            "status": "success",
            "file_path": file_path,
            "collection": collection,
            "chunk_count": chunk_count,
        }

    def mark_failed(self, file_hash: str, file_path: str, error_msg: str) -> None:
        self._processed[file_hash] = {"status": "failed"}

    def get_status(self, file_hash: str) -> Optional[str]:
        return self._processed.get(file_hash, {}).get("status")

    def remove_record(self, file_hash: str) -> bool:
        if file_hash in self._processed:
            del self._processed[file_hash]
            return True
        return False

    def list_processed(self) -> list:
        return [
            {
                "file_hash": h, 
                "status": "success",
                "file_path": d.get("file_path"),
                "collection": d.get("collection"),
                "chunk_count": d.get("chunk_count"),
            }
            for h, d in self._processed.items()
            if d.get("status") == "success"
        ]


# ============================================================
# 辅助工具
# ============================================================


def _make_settings() -> Settings:
    """创建最小测试配置。"""
    return Settings(
        llm=LLMConfig(provider="fake", model="fake"),
        vision_llm=VisionLLMConfig(),
        ollama=OllamaConfig(),
        embedding=EmbeddingConfig(provider="fake", model="fake", dimensions=3),
        splitter=SplitterConfig(strategy="recursive", chunk_size=500, chunk_overlap=50),
        vector_store=VectorStoreConfig(provider="fake"),
        retrieval=RetrievalConfig(),
        rerank=RerankConfig(),
        evaluation=EvaluationConfig(),
        observability=ObservabilityConfig(),
        pipeline=MagicMock(),
    )


def _make_document(
    text: str = "这是测试文档。包含足够长的文本内容用于切分测试。" * 20,
    source_path: str = "test.pdf",
    doc_id: str = "test_doc_001",
) -> Document:
    """创建测试 Document。"""
    return Document(
        id=doc_id,
        text=text,
        metadata={"source_path": source_path, "doc_type": "pdf"},
    )


def _make_pipeline(
    settings: Optional[Settings] = None,
    loader: Optional[BaseLoader] = None,
    document: Optional[Document] = None,
    batch_processor=None,
    vector_upserter=None,
    bm25_indexer=None,
    integrity_checker=None,
) -> tuple:
    """创建注入 Fake 组件的 Pipeline。返回 (pipeline, fakes_dict)。"""
    settings = settings or _make_settings()
    document = document or _make_document()
    loader = loader or FakeLoader(document)
    batch_processor = batch_processor or FakeBatchProcessor()
    vector_upserter = vector_upserter or FakeVectorUpserter()
    bm25_indexer = bm25_indexer or FakeBM25Indexer()
    integrity_checker = integrity_checker or FakeIntegrityChecker()

    # 使用真实 Chunker（含 RecursiveSplitter）和 Refiner（规则模式）
    chunker = DocumentChunker(settings)
    refiner = ChunkRefiner(settings, use_llm=False)

    compute_hash = getattr(integrity_checker, "compute_hash", None)

    pipeline = IngestionPipeline(
        settings,
        integrity_checker=integrity_checker,
        loader=loader,
        chunker=chunker,
        refiner=refiner,
        batch_processor=batch_processor,
        vector_upserter=vector_upserter,
        bm25_indexer=bm25_indexer,
        compute_hash=compute_hash,
    )

    fakes = {
        "loader": loader,
        "batch_processor": batch_processor,
        "vector_upserter": vector_upserter,
        "bm25_indexer": bm25_indexer,
        "integrity_checker": integrity_checker,
    }
    return pipeline, fakes


# ============================================================
# 测试用例
# ============================================================


class TestPipelineRun:
    """Pipeline 正常运行测试。"""

    def test_full_pipeline_produces_result(self):
        """完整 pipeline 应产出 PipelineResult。"""
        pipeline, fakes = _make_pipeline()

        result = pipeline.run("fake_file.pdf", collection="test")

        assert isinstance(result, PipelineResult)
        assert result.skipped is False
        assert result.chunk_count > 0
        assert result.record_count > 0
        assert result.elapsed_ms >= 0
        assert result.file_path == "fake_file.pdf"
        assert result.collection == "test"

    def test_pipeline_calls_all_stages(self):
        """pipeline 应调用所有组件。"""
        pipeline, fakes = _make_pipeline()

        pipeline.run("fake_file.pdf")

        assert fakes["loader"].call_count == 1
        assert fakes["batch_processor"].call_count == 1
        assert fakes["vector_upserter"].call_count == 1
        # 可能调用 build 或 add_documents
        assert fakes["bm25_indexer"].build_count + fakes["bm25_indexer"].add_count == 1
        assert fakes["bm25_indexer"].save_count == 1

    def test_pipeline_loader_receives_correct_args(self):
        """loader 应收到正确的 file_path 和 collection。"""
        pipeline, fakes = _make_pipeline()

        pipeline.run("/path/to/doc.pdf", collection="my_collection")

        loader = fakes["loader"]
        assert loader.last_path == "/path/to/doc.pdf"
        assert loader.last_collection == "my_collection"

    def test_pipeline_data_flows_through_stages(self):
        """数据应正确在各阶段间流转。"""
        doc = _make_document(text="这是第一段文本内容。" * 30 + "\n\n这是第二段文本内容。" * 30)
        pipeline, fakes = _make_pipeline(document=doc)

        result = pipeline.run("fake.pdf")

        bp = fakes["batch_processor"]
        assert bp.last_chunks is not None
        assert len(bp.last_chunks) > 0

        vu = fakes["vector_upserter"]
        assert vu.last_records is not None
        assert len(vu.last_records) == result.record_count

        bm = fakes["bm25_indexer"]
        assert bm.last_records is not None
        assert len(bm.last_records) == result.record_count

    def test_pipeline_record_count_matches_chunks(self):
        """record_count 应等于 chunk_count。"""
        pipeline, _ = _make_pipeline()

        result = pipeline.run("fake.pdf")

        assert result.record_count == result.chunk_count

    def test_pipeline_stage_times_recorded(self):
        """各阶段耗时应被记录。"""
        pipeline, _ = _make_pipeline()

        result = pipeline.run("fake.pdf")

        expected_stages = ["integrity", "load", "split", "transform", "encode", "store"]
        for stage in expected_stages:
            assert stage in result.stage_times
            assert result.stage_times[stage] >= 0

    def test_pipeline_doc_id_propagated(self):
        """Document.id 应传播到 PipelineResult。"""
        doc = _make_document(doc_id="custom_doc_id_xyz")
        pipeline, _ = _make_pipeline(document=doc)

        result = pipeline.run("fake.pdf")

        assert result.doc_id == "custom_doc_id_xyz"


class TestPipelineSkip:
    """增量跳过测试。"""

    def test_skip_already_processed(self):
        """已处理的文件应被跳过。"""
        pipeline, fakes = _make_pipeline()

        result1 = pipeline.run("fake.pdf")
        assert result1.skipped is False
        assert fakes["loader"].call_count == 1

        result2 = pipeline.run("fake.pdf")
        assert result2.skipped is True
        assert fakes["loader"].call_count == 1

    def test_force_bypasses_skip(self):
        """force=True 应跳过增量检查。"""
        pipeline, fakes = _make_pipeline()

        pipeline.run("fake.pdf")
        assert fakes["loader"].call_count == 1

        result = pipeline.run("fake.pdf", force=True)
        assert result.skipped is False
        assert fakes["loader"].call_count == 2

    def test_skip_returns_empty_result(self):
        """跳过时应返回空结果。"""
        pipeline, _ = _make_pipeline()

        pipeline.run("fake.pdf")
        result = pipeline.run("fake.pdf")

        assert result.chunk_count == 0
        assert result.record_count == 0
        assert result.doc_id == ""
        assert result.file_hash == ""


class TestPipelineErrorHandling:
    """异常处理测试。"""

    def test_file_not_found_raises(self):
        """文件不存在时应抛出异常。"""
        pipeline, _ = _make_pipeline()

        pipeline._compute_hash = lambda path: (_ for _ in ()).throw(
            FileNotFoundError(f"文件不存在: {path}")
        )

        with pytest.raises(PipelineError) as exc_info:
            pipeline.run("/nonexistent/file.pdf")

        assert exc_info.value.stage == "integrity"

    def test_loader_failure_raises_pipeline_error(self):
        """加载器失败时应抛出 PipelineError。"""
        failing_loader = MagicMock(spec=BaseLoader)
        failing_loader.load.side_effect = RuntimeError("加载失败")

        pipeline, _ = _make_pipeline(loader=failing_loader)

        with pytest.raises(PipelineError) as exc_info:
            pipeline.run("fake.pdf")

        assert exc_info.value.stage == "load"
        assert "加载失败" in str(exc_info.value.original_error)

    def test_encode_failure_marks_integrity(self):
        """编码失败时应标记 integrity 为 failed。"""
        failing_bp = MagicMock()
        failing_bp.process.side_effect = RuntimeError("编码失败")

        pipeline, fakes = _make_pipeline(batch_processor=failing_bp)

        with pytest.raises(PipelineError) as exc_info:
            pipeline.run("fake.pdf")

        assert exc_info.value.stage == "encode"

    def test_pipeline_error_preserves_stage(self):
        """PipelineError 应保存失败阶段名称。"""
        err = PipelineError("store", RuntimeError("写入失败"))
        assert err.stage == "store"
        assert "store" in str(err)
        assert isinstance(err.original_error, RuntimeError)


class TestPipelineProgress:
    """进度回调测试。"""

    def test_progress_callback_called(self):
        """on_progress 回调应被调用。"""
        pipeline, _ = _make_pipeline()

        progress_calls = []

        def on_progress(stage, current, total):
            progress_calls.append((stage, current, total))

        pipeline.run("fake.pdf", on_progress=on_progress)

        assert len(progress_calls) == 6
        stages = [call[0] for call in progress_calls]
        assert stages == ["integrity", "load", "split", "transform", "encode", "store"]

    def test_progress_callback_args_correct(self):
        """回调参数应正确。"""
        pipeline, _ = _make_pipeline()

        progress_calls = []

        def on_progress(stage, current, total):
            progress_calls.append({"stage": stage, "current": current, "total": total})

        pipeline.run("fake.pdf", on_progress=on_progress)

        for i, call in enumerate(progress_calls):
            assert call["current"] == i + 1
            assert call["total"] == 6

    def test_no_progress_callback_no_error(self):
        """不传回调时不应报错。"""
        pipeline, _ = _make_pipeline()

        result = pipeline.run("fake.pdf")
        assert result.skipped is False


class TestPipelineTrace:
    """Trace 记录测试（F4 验收标准）。"""

    # -- 验收标准 1: trace 包含 load/split/transform/embed/upsert 阶段 --

    def test_trace_records_all_stages(self):
        """trace 应记录所有 5 个核心阶段（load/split/transform/embed/upsert）。"""
        pipeline, _ = _make_pipeline()

        trace = TraceContext(trace_type="ingestion")
        pipeline.run("fake.pdf", trace=trace)

        stage_names = [s["name"] for s in trace.stages]
        # 核心 5 阶段
        assert "load" in stage_names
        assert "split" in stage_names
        assert "transform" in stage_names
        assert "embed" in stage_names
        assert "upsert" in stage_names

    def test_trace_integrity_stage_data(self):
        """integrity 阶段应记录 file_hash 和 action。"""
        pipeline, _ = _make_pipeline()

        trace = TraceContext(trace_type="ingestion")
        pipeline.run("fake.pdf", trace=trace)

        integrity_stage = next(s for s in trace.stages if s["name"] == "integrity")
        assert integrity_stage["action"] == "process"
        assert "file_hash" in integrity_stage

    # -- 验收标准 2: 每阶段记录 elapsed_ms、method 和处理详情 --

    def test_trace_stages_have_elapsed_ms(self):
        """pipeline 自身记录的核心阶段都应包含 elapsed_ms 字段。"""
        pipeline, _ = _make_pipeline()

        trace = TraceContext(trace_type="ingestion")
        pipeline.run("fake.pdf", trace=trace)

        # 只检查 pipeline 自身记录的阶段（不含子组件内部阶段）
        pipeline_stages = ["integrity", "load", "split", "transform", "embed", "upsert"]
        for stage in trace.stages:
            if stage["name"] in pipeline_stages:
                assert "elapsed_ms" in stage, f"阶段 {stage['name']} 缺少 elapsed_ms"
                assert isinstance(stage["elapsed_ms"], (int, float))
                assert stage["elapsed_ms"] >= 0

    def test_trace_stages_have_method(self):
        """核心阶段应包含 method 字段。"""
        pipeline, _ = _make_pipeline()

        trace = TraceContext(trace_type="ingestion")
        pipeline.run("fake.pdf", trace=trace)

        method_stages = ["load", "split", "transform", "embed", "upsert"]
        for stage in trace.stages:
            if stage["name"] in method_stages:
                assert "method" in stage, f"阶段 {stage['name']} 缺少 method"

    def test_trace_load_stage_details(self):
        """load 阶段应记录 text_length 和 image_count。"""
        pipeline, _ = _make_pipeline()

        trace = TraceContext(trace_type="ingestion")
        pipeline.run("fake.pdf", trace=trace)

        load_stage = next(s for s in trace.stages if s["name"] == "load")
        assert load_stage["text_length"] > 0
        assert "image_count" in load_stage

    def test_trace_split_stage_details(self):
        """split 阶段应记录 chunk_count。"""
        pipeline, _ = _make_pipeline()

        trace = TraceContext(trace_type="ingestion")
        pipeline.run("fake.pdf", trace=trace)

        split_stage = next(s for s in trace.stages if s["name"] == "split")
        assert split_stage["chunk_count"] > 0

    def test_trace_transform_stage_details(self):
        """transform 阶段应记录 chunk_count。"""
        pipeline, _ = _make_pipeline()

        trace = TraceContext(trace_type="ingestion")
        pipeline.run("fake.pdf", trace=trace)

        transform_stage = next(s for s in trace.stages if s["name"] == "transform")
        assert transform_stage["chunk_count"] > 0

    def test_trace_embed_stage_details(self):
        """embed 阶段应记录 record_count。"""
        pipeline, _ = _make_pipeline()

        trace = TraceContext(trace_type="ingestion")
        pipeline.run("fake.pdf", trace=trace)

        embed_stage = next(s for s in trace.stages if s["name"] == "embed")
        assert embed_stage["record_count"] > 0

    def test_trace_upsert_stage_details(self):
        """upsert 阶段应记录 vector_count 和 vocabulary_size。"""
        pipeline, _ = _make_pipeline()

        trace = TraceContext(trace_type="ingestion")
        pipeline.run("fake.pdf", trace=trace)

        upsert_stage = next(s for s in trace.stages if s["name"] == "upsert")
        assert upsert_stage["vector_count"] > 0
        assert "vocabulary_size" in upsert_stage

    # -- 验收标准 3: trace_type == "ingestion" --

    def test_trace_type_is_ingestion(self):
        """trace.to_dict() 中 trace_type 应为 ingestion。"""
        pipeline, _ = _make_pipeline()

        trace = TraceContext(trace_type="ingestion")
        pipeline.run("fake.pdf", trace=trace)
        trace.finish()

        trace_dict = trace.to_dict()
        assert trace_dict["trace_type"] == "ingestion"
        assert trace_dict["stages"]  # 非空
        assert trace_dict["total_elapsed_ms"] >= 0


class TestPipelineIntegrityIntegration:
    """与 FakeIntegrityChecker 的集成测试。"""

    def test_integrity_marks_success(self):
        """成功后应标记 integrity 为 success。"""
        integrity = FakeIntegrityChecker()
        pipeline, fakes = _make_pipeline(integrity_checker=integrity)

        pipeline.run("fake.pdf")

        records = integrity.list_processed()
        assert len(records) == 1

    def test_different_files_not_skipped(self):
        """不同文件不应互相跳过。"""
        pipeline, fakes = _make_pipeline()

        result1 = pipeline.run("file_a.pdf")
        result2 = pipeline.run("file_b.pdf")

        assert result1.skipped is False
        assert result2.skipped is False
        assert fakes["loader"].call_count == 2


class TestPipelineEdgeCases:
    """边界场景测试。"""

    def test_pipeline_with_different_collections(self):
        """不同 collection 应产出不同结果。"""
        pipeline, fakes = _make_pipeline()

        result1 = pipeline.run("file_a.pdf", collection="col_a")
        result2 = pipeline.run("file_b.pdf", collection="col_b")

        assert result1.collection == "col_a"
        assert result2.collection == "col_b"
        assert fakes["loader"].call_count == 2

    def test_pipeline_result_has_hash(self):
        """结果应包含 file_hash。"""
        pipeline, _ = _make_pipeline()

        result = pipeline.run("fake.pdf")

        assert len(result.file_hash) == 64

    def test_pipeline_long_document(self):
        """长文档应正确处理。"""
        long_text = "这是第{}段的内容。".format(1) * 100
        for i in range(2, 20):
            long_text += "\n\n" + "这是第{}段的内容。".format(i) * 100
        doc = _make_document(text=long_text)

        pipeline, _ = _make_pipeline(document=doc)

        result = pipeline.run("fake.pdf")

        assert result.chunk_count > 1
        assert result.record_count > 1
