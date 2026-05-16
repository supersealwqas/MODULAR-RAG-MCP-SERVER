"""C10 BatchProcessor 手动测试脚本。

测试 BatchProcessor 的批处理编排、Dense/Sparse 编码协调、Trace 记录等。
使用真实 Embedding 进行测试。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.settings import load_settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk
from src.ingestion.embedding.batch_processor import BatchProcessor
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
# 导入 BGE Embedding 以触发 @register_embedding 注册
from src.libs.embedding.bge_embedding import BGEEmbedding  # noqa: F401


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def safe_print(text: str):
    """安全打印，忽略无法编码的字符。"""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))


def _make_chunks(count: int = 5) -> list:
    """创建测试用 Chunk 列表。"""
    texts = [
        "RAG 检索增强生成是一种结合信息检索与大语言模型的技术框架。",
        "向量数据库是 RAG 系统的核心组件，负责存储和检索文档的稠密向量表示。",
        "BM25 是一种经典的稀疏检索算法，基于词频和逆文档频率进行文档排序。",
        "混合检索结合了稠密向量检索和稀疏关键词检索的优势，通过 RRF 算法融合结果。",
        "Chunk 切分策略直接影响检索质量，RecursiveCharacterTextSplitter 是常用的切分方案。",
    ]
    return [
        Chunk(
            id=f"test_{i:04d}_abcd1234",
            text=texts[i % len(texts)],
            metadata={"source_path": "/test.pdf", "chunk_index": i},
            source_ref="doc_001",
        )
        for i in range(count)
    ]


def test_basic_processing():
    """测试基本批处理。"""
    section("基本批处理测试")

    settings = load_settings()
    dense = DenseEncoder(settings=settings)
    sparse = SparseEncoder(settings=settings)
    processor = BatchProcessor(
        settings=settings,
        dense_encoder=dense,
        sparse_encoder=sparse,
    )

    chunks = _make_chunks(3)
    records = processor.process(chunks)

    print(f"  输入 chunks: {len(chunks)}")
    print(f"  输出 records: {len(records)}")
    print(f"  有 dense_vector: {all(r.dense_vector is not None for r in records)}")
    print(f"  有 sparse_vector: {all(r.sparse_vector is not None for r in records)}")
    if records[0].dense_vector:
        print(f"  dense 维度: {len(records[0].dense_vector)}")
    if records[0].sparse_vector:
        print(f"  sparse term 数: {len(records[0].sparse_vector)}")


def test_batch_splitting():
    """测试分批逻辑。"""
    section("分批逻辑测试")

    settings = load_settings()
    dense = DenseEncoder(settings=settings)
    sparse = SparseEncoder(settings=settings)
    processor = BatchProcessor(
        settings=settings,
        dense_encoder=dense,
        sparse_encoder=sparse,
        batch_size=2,
    )

    chunks = _make_chunks(5)
    batches = processor.split_into_batches(chunks)

    print(f"  输入 chunks: {len(chunks)}")
    print(f"  batch_size: {processor.batch_size}")
    print(f"  批次数: {len(batches)}")
    for i, batch in enumerate(batches):
        print(f"    批次 {i+1}: {len(batch)} chunks")


def test_order_stability():
    """测试顺序稳定性。"""
    section("顺序稳定性测试")

    settings = load_settings()
    dense = DenseEncoder(settings=settings)
    sparse = SparseEncoder(settings=settings)
    processor = BatchProcessor(
        settings=settings,
        dense_encoder=dense,
        sparse_encoder=sparse,
        batch_size=2,
    )

    chunks = _make_chunks(7)
    records = processor.process(chunks)

    input_ids = [c.id for c in chunks]
    output_ids = [r.id for r in records]
    print(f"  输入 ID 顺序: {input_ids}")
    print(f"  输出 ID 顺序: {output_ids}")
    print(f"  顺序一致: {input_ids == output_ids}")


def test_trace_recording():
    """测试 Trace 记录。"""
    section("Trace 记录测试")

    settings = load_settings()
    dense = DenseEncoder(settings=settings)
    sparse = SparseEncoder(settings=settings)
    processor = BatchProcessor(
        settings=settings,
        dense_encoder=dense,
        sparse_encoder=sparse,
        batch_size=2,
    )

    trace = TraceContext(trace_type="ingestion")
    chunks = _make_chunks(5)
    processor.process(chunks, trace=trace)

    print(f"  trace_id: {trace.trace_id}")
    print(f"  stages: {len(trace.stages)}")
    for stage in trace.stages:
        safe_print(f"  - {stage['name']} chunks={stage.get('chunk_count', '-')} "
                   f"elapsed={stage.get('elapsed_ms', '-')}ms")


def main():
    test_basic_processing()
    test_batch_splitting()
    test_order_stability()
    test_trace_recording()
    print(f"\n{'='*60}")
    print("  全部手动测试完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
