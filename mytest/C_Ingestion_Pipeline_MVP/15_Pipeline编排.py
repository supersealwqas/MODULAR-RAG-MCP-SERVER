"""C14 IngestionPipeline 手动测试脚本。

测试完整摄取流程编排：integrity → load → split → transform → encode → store。
使用 Fake 组件注入隔离外部依赖。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.libs.loader.base_loader import BaseLoader
from src.libs.splitter.recursive_splitter import RecursiveSplitter  # noqa: F401


def safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("gbk", errors="replace").decode("gbk", errors="replace"))


def section(title: str):
    safe_print(f"\n{'='*60}")
    safe_print(f"  {title}")
    safe_print(f"{'='*60}")


class FakeLoader(BaseLoader):
    def __init__(self, document: Document):
        self._document = document
        self.call_count = 0

    def load(self, path: str, collection: str = "default") -> Document:
        self.call_count += 1
        return self._document


class FakeBatchProcessor:
    def __init__(self):
        self.call_count = 0

    def process(self, chunks, trace=None):
        self.call_count += 1
        return [
            ChunkRecord(
                id=chunk.id, text=chunk.text, metadata=chunk.metadata.copy(),
                dense_vector=[float(i) * 0.1, float(i) * 0.2, float(i) * 0.3],
                sparse_vector={"test": 1.0},
            )
            for i, chunk in enumerate(chunks)
        ]


class FakeVectorUpserter:
    def __init__(self):
        self.call_count = 0

    def upsert(self, records, trace=None):
        self.call_count += 1
        return len(records)

    def delete(self, chunk_ids, trace=None):
        return len(chunk_ids)


class FakeBM25Indexer:
    def __init__(self):
        self.build_count = 0
        self.save_count = 0

    def build(self, records, trace=None):
        self.build_count += 1

    def save(self, path=None):
        self.save_count += 1
        return path or "fake"

    def get_vocabulary_size(self):
        return 10


class FakeIntegrityChecker:
    def __init__(self):
        self._processed = {}
        self._hash_cache = {}

    def compute_hash(self, file_path):
        if file_path not in self._hash_cache:
            self._hash_cache[file_path] = f"{hash(file_path) % (10**64):064d}"
        return self._hash_cache[file_path]

    def should_skip(self, file_hash):
        return self._processed.get(file_hash) == "success"

    def mark_success(self, file_hash, file_path, file_size=0, chunk_count=0):
        self._processed[file_hash] = "success"

    def mark_failed(self, file_hash, file_path, error_msg):
        self._processed[file_hash] = "failed"

    def get_status(self, file_hash):
        return self._processed.get(file_hash)


def _make_settings():
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
    )


def _make_pipeline(document, collection="default"):
    settings = _make_settings()
    integrity = FakeIntegrityChecker()
    return IngestionPipeline(
        settings,
        integrity_checker=integrity,
        loader=FakeLoader(document),
        chunker=DocumentChunker(settings),
        refiner=ChunkRefiner(settings, use_llm=False),
        batch_processor=FakeBatchProcessor(),
        vector_upserter=FakeVectorUpserter(),
        bm25_indexer=FakeBM25Indexer(),
        compute_hash=integrity.compute_hash,
    )


def test_full_pipeline():
    section("完整 Pipeline 运行")
    doc = Document(
        id="test_doc_001",
        text="这是测试文档的第一段内容。" * 30 + "\n\n这是第二段内容。" * 30,
        metadata={"source_path": "test.pdf", "doc_type": "pdf"},
    )
    pipeline = _make_pipeline(doc)
    result = pipeline.run("test.pdf", collection="test_collection")

    safe_print(f"  skipped: {result.skipped}")
    safe_print(f"  chunk_count: {result.chunk_count}")
    safe_print(f"  record_count: {result.record_count}")
    safe_print(f"  doc_id: {result.doc_id}")
    safe_print(f"  elapsed_ms: {result.elapsed_ms:.1f}")
    safe_print(f"  stage_times: {result.stage_times}")


def test_skip_behavior():
    section("增量跳过")
    doc = Document(
        id="test_doc_002",
        text="用于跳过测试的文档内容。" * 20,
        metadata={"source_path": "skip_test.pdf"},
    )
    settings = _make_settings()
    integrity = FakeIntegrityChecker()
    pipeline = IngestionPipeline(
        settings,
        integrity_checker=integrity,
        loader=FakeLoader(doc),
        chunker=DocumentChunker(settings),
        refiner=ChunkRefiner(settings, use_llm=False),
        batch_processor=FakeBatchProcessor(),
        vector_upserter=FakeVectorUpserter(),
        bm25_indexer=FakeBM25Indexer(),
        compute_hash=integrity.compute_hash,
    )

    r1 = pipeline.run("skip_test.pdf")
    safe_print(f"  第一次: skipped={r1.skipped}, chunks={r1.chunk_count}")

    r2 = pipeline.run("skip_test.pdf")
    safe_print(f"  第二次: skipped={r2.skipped}, chunks={r2.chunk_count}")

    r3 = pipeline.run("skip_test.pdf", force=True)
    safe_print(f"  force=True: skipped={r3.skipped}, chunks={r3.chunk_count}")


def test_trace_recording():
    section("Trace 记录")
    doc = Document(
        id="test_doc_003",
        text="Trace 测试文档。" * 20,
        metadata={"source_path": "trace_test.pdf"},
    )
    pipeline = _make_pipeline(doc)
    trace = TraceContext(trace_type="ingestion")
    pipeline.run("trace_test.pdf", trace=trace)

    safe_print(f"  trace_id: {trace.trace_id}")
    safe_print(f"  stages: {len(trace.stages)}")
    for stage in trace.stages:
        safe_print(f"    {stage['name']}: {stage}")


def test_progress_callback():
    section("进度回调")
    doc = Document(
        id="test_doc_004",
        text="进度测试文档。" * 20,
        metadata={"source_path": "progress_test.pdf"},
    )
    pipeline = _make_pipeline(doc)

    def on_progress(stage, current, total):
        safe_print(f"  [{current}/{total}] {stage}")

    pipeline.run("progress_test.pdf", on_progress=on_progress)


def test_error_handling():
    section("异常处理")
    settings = _make_settings()
    integrity = FakeIntegrityChecker()

    failing_loader = type("FailingLoader", (), {
        "load": lambda self, *a, **kw: (_ for _ in ()).throw(RuntimeError("模拟加载失败"))
    })()

    pipeline = IngestionPipeline(
        settings,
        integrity_checker=integrity,
        loader=failing_loader,
        chunker=DocumentChunker(settings),
        refiner=ChunkRefiner(settings, use_llm=False),
        batch_processor=FakeBatchProcessor(),
        vector_upserter=FakeVectorUpserter(),
        bm25_indexer=FakeBM25Indexer(),
        compute_hash=integrity.compute_hash,
    )

    try:
        pipeline.run("error_test.pdf")
        safe_print("  ERROR: 应该抛出异常!")
    except PipelineError as e:
        safe_print(f"  PipelineError: stage={e.stage}, error={e.original_error}")


def test_metadata_enricher_integration():
    section("MetadataEnricher 集成验证")
    doc = Document(
        id="test_doc_enrich",
        text="# RAG 技术概述\n\nRAG 是检索增强生成技术。它结合了检索和生成两种方法。" * 10,
        metadata={"source_path": "enrich_test.pdf"},
    )
    pipeline = _make_pipeline(doc)

    # 验证 Pipeline 使用了 MetadataEnricher
    enricher = pipeline._get_enricher()
    safe_print(f"  enricher 类型: {type(enricher).__name__}")
    safe_print(f"  是 MetadataEnricher: {type(enricher).__name__ == 'MetadataEnricher'}")

    # 运行 Pipeline 并检查 chunk metadata
    result = pipeline.run("enrich_test.pdf", collection="enrich_test")
    safe_print(f"  chunk_count: {result.chunk_count}")


def main():
    test_full_pipeline()
    test_skip_behavior()
    test_trace_recording()
    test_progress_callback()
    test_error_handling()
    test_metadata_enricher_integration()

    safe_print(f"\n{'='*60}")
    safe_print("  全部手动测试完成!")
    safe_print(f"{'='*60}")


if __name__ == "__main__":
    main()
