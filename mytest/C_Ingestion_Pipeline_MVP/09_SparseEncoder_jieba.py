"""C9 SparseEncoder 手动测试脚本。

测试 SparseEncoder 的分词、TF 计算、停用词过滤、Trace 记录等。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.settings import load_settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk
from src.ingestion.embedding.sparse_encoder import SparseEncoder


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


def test_basic_encoding():
    """测试基本稀疏向量编码。"""
    section("基本稀疏向量编码测试")

    settings = load_settings()
    encoder = SparseEncoder(settings=settings)

    chunks = [
        Chunk(
            id=f"test_{i:04d}_abcd1234",
            text=text,
            metadata={"source_path": "/test.pdf", "chunk_index": i},
            source_ref="doc_001",
        )
        for i, text in enumerate([
            "RAG 检索增强生成是一种结合信息检索与大语言模型的技术框架。",
            "向量数据库是 RAG 系统的核心组件，负责存储和检索文档的稠密向量。",
            "BM25 是一种经典的稀疏检索算法，基于词频和逆文档频率进行排序。",
        ])
    ]

    records = encoder.encode(chunks)
    print(f"  输入 chunks: {len(chunks)}")
    print(f"  输出 records: {len(records)}")

    for i, record in enumerate(records):
        sv = record.sparse_vector
        safe_print(f"  [{i}] term 数量: {len(sv)}")
        # 显示 top 5 terms
        sorted_terms = sorted(sv.items(), key=lambda x: x[1], reverse=True)[:5]
        safe_print(f"      top 5: {sorted_terms}")


def test_stopwords():
    """测试停用词过滤。"""
    section("停用词过滤测试")

    settings = load_settings()
    encoder = SparseEncoder(settings=settings)

    chunk = Chunk(
        id="stopword_0000",
        text="这是一个测试，用于验证停用词的过滤效果。",
        metadata={"source_path": "/test.pdf", "chunk_index": 0},
        source_ref="doc_001",
    )
    records = encoder.encode([chunk])
    sv = records[0].sparse_vector

    safe_print(f"  term 数量: {len(sv)}")
    safe_print(f"  所有 terms: {list(sv.keys())}")
    # 验证停用词被过滤
    for word in ["的", "是", "在", "这", "一"]:
        print(f"  '{word}' 被过滤: {word not in sv}")


def test_empty_text():
    """测试空文本处理。"""
    section("空文本处理测试")

    settings = load_settings()
    encoder = SparseEncoder(settings=settings)

    test_cases = [
        ("空字符串", ""),
        ("纯空白", "   \n\t  "),
        ("全停用词", "的 是 在 了"),
    ]

    for name, text in test_cases:
        chunk = Chunk(
            id=f"empty_{name}",
            text=text,
            metadata={"source_path": "/test.pdf", "chunk_index": 0},
            source_ref="doc_001",
        )
        records = encoder.encode([chunk])
        sv = records[0].sparse_vector
        print(f"  {name}: term 数量 = {len(sv)}, 空 dict = {sv == {}}")


def test_trace_recording():
    """测试 Trace 记录。"""
    section("Trace 记录测试")

    settings = load_settings()
    encoder = SparseEncoder(settings=settings)
    trace = TraceContext(trace_type="ingestion")

    chunk = Chunk(
        id="trace_0000",
        text="RAG 检索增强生成技术测试。",
        metadata={"source_path": "/test.pdf", "chunk_index": 0},
        source_ref="doc_001",
    )
    encoder.encode([chunk], trace=trace)

    print(f"  trace_id: {trace.trace_id}")
    print(f"  stages: {len(trace.stages)}")
    for stage in trace.stages:
        safe_print(f"  - {stage['name']} method={stage.get('method')} "
                   f"chunks={stage.get('chunk_count')} "
                   f"terms={stage.get('total_terms')} "
                   f"elapsed={stage.get('elapsed_ms')}ms")


def main():
    test_basic_encoding()
    test_stopwords()
    test_empty_text()
    test_trace_recording()
    print(f"\n{'='*60}")
    print("  全部手动测试完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
