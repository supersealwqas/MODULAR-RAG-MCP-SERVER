"""C8 DenseEncoder 手动测试脚本。

测试 DenseEncoder 的向量编码、批量处理、降级机制、Trace 记录等。
使用真实 Embedding 模型进行测试。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.settings import load_settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk
from src.ingestion.embedding.dense_encoder import DenseEncoder
# 导入 BGE Embedding 以触发 @register_embedding 注册
from src.libs.embedding.bge_embedding import BGEEmbedding  # noqa: F401
from src.libs.embedding.embedding_factory import EmbeddingFactory


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


def _make_chunks(count: int = 3) -> list:
    """创建测试用 Chunk 列表。"""
    texts = [
        "RAG（检索增强生成）是一种结合信息检索与大语言模型的技术框架。",
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


def test_basic_encoding():
    """测试基本向量编码。"""
    section("基本向量编码测试")

    settings = load_settings()
    try:
        embedding = EmbeddingFactory.create(settings.embedding)
    except Exception as e:
        print(f"  Embedding 创建失败: {e}")
        return

    encoder = DenseEncoder(settings=settings, embedding=embedding)
    chunks = _make_chunks(3)

    records = encoder.encode(chunks)
    print(f"  输入 chunks: {len(chunks)}")
    print(f"  输出 records: {len(records)}")
    print(f"  向量维度: {len(records[0].dense_vector) if records[0].dense_vector else 'None'}")
    print(f"  ID 保留: {records[0].id == chunks[0].id}")
    print(f"  文本保留: {records[0].text == chunks[0].text}")

    safe_print(f"  向量预览 (前5维): {records[0].dense_vector[:5]}")


def test_batch_processing():
    """测试批量处理。"""
    section("批量处理测试")

    settings = load_settings()
    try:
        embedding = EmbeddingFactory.create(settings.embedding)
    except Exception as e:
        print(f"  Embedding 创建失败: {e}")
        return

    encoder = DenseEncoder(settings=settings, embedding=embedding, batch_size=2)
    chunks = _make_chunks(5)

    records = encoder.encode(chunks)
    print(f"  输入 chunks: {len(chunks)}")
    print(f"  batch_size: {encoder.batch_size}")
    print(f"  输出 records: {len(records)}")
    print(f"  全部有向量: {all(r.dense_vector is not None for r in records)}")
    print(f"  维度一致: {len(set(len(r.dense_vector) for r in records)) == 1}")


def test_fallback():
    """测试 Embedding 不可用时的降级行为。"""
    section("降级测试（Embedding 不可用）")

    from src.core.settings import EmbeddingConfig

    bad_settings = load_settings()
    bad_settings.embedding = EmbeddingConfig(
        provider="nonexistent",
        model="nonexistent-model",
    )

    encoder = DenseEncoder(settings=bad_settings, embedding=None)
    chunks = _make_chunks(2)
    records = encoder.encode(chunks)

    print(f"  输入 chunks: {len(chunks)}")
    print(f"  输出 records: {len(records)}")
    print(f"  dense_vector 为 None: {records[0].dense_vector is None}")
    print(f"  ID 保留: {records[0].id == chunks[0].id}")
    print(f"  降级成功，未崩溃: True")


def test_trace_recording():
    """测试 Trace 记录。"""
    section("Trace 记录测试")

    settings = load_settings()
    try:
        embedding = EmbeddingFactory.create(settings.embedding)
    except Exception as e:
        print(f"  Embedding 创建失败: {e}")
        return

    encoder = DenseEncoder(settings=settings, embedding=embedding)
    trace = TraceContext(trace_type="ingestion")

    chunks = _make_chunks(3)
    encoder.encode(chunks, trace=trace)

    print(f"  trace_id: {trace.trace_id}")
    print(f"  stages: {len(trace.stages)}")
    for stage in trace.stages:
        safe_print(f"  - {stage['name']} method={stage.get('method')} "
                   f"chunks={stage.get('chunk_count')} "
                   f"dim={stage.get('vector_dim')} "
                   f"elapsed={stage.get('elapsed_ms')}ms")


def main():
    test_basic_encoding()
    test_batch_processing()
    test_fallback()
    test_trace_recording()
    print(f"\n{'='*60}")
    print("  全部手动测试完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
