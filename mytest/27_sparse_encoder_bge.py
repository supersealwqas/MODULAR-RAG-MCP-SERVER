"""SparseEncoder 双后端手动测试脚本。

对比 jieba 和 BGE-M3 两种稀疏编码后端的输出差异。
使用真实 BGE-M3 模型进行测试。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.settings import load_settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk
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


def _make_chunks() -> list:
    """创建测试用 Chunk 列表。"""
    texts = [
        "RAG 检索增强生成是一种结合信息检索与大语言模型的技术框架。",
        "向量数据库是 RAG 系统的核心组件，负责存储和检索文档的稠密向量表示。",
        "BM25 是一种经典的稀疏检索算法，基于词频和逆文档频率进行文档排序。",
    ]
    return [
        Chunk(
            id=f"test_{i:04d}_abcd1234",
            text=text,
            metadata={"source_path": "/test.pdf", "chunk_index": i},
            source_ref="doc_001",
        )
        for i, text in enumerate(texts)
    ]


def test_jieba_backend():
    """测试 jieba 后端。"""
    section("jieba 后端")

    settings = load_settings()
    encoder = SparseEncoder(settings=settings, backend="jieba")
    chunks = _make_chunks()

    trace = TraceContext(trace_type="ingestion")
    records = encoder.encode(chunks, trace=trace)

    for record in records:
        safe_print(f"\n  [{record.id}]")
        safe_print(f"    terms: {len(record.sparse_vector)}")
        # 显示前 5 个 term
        sorted_terms = sorted(record.sparse_vector.items(), key=lambda x: -x[1])
        for term, weight in sorted_terms[:5]:
            safe_print(f"    {term}: {weight:.4f}")

    safe_print(f"\n  Trace stages: {[s['name'] for s in trace.stages]}")
    return records


def test_bge_backend():
    """测试 BGE-M3 后端。"""
    section("BGE-M3 后端")

    settings = load_settings()
    encoder = SparseEncoder(settings=settings, backend="bge")
    chunks = _make_chunks()

    trace = TraceContext(trace_type="ingestion")
    records = encoder.encode(chunks, trace=trace)

    for record in records:
        safe_print(f"\n  [{record.id}]")
        safe_print(f"    terms: {len(record.sparse_vector)}")
        # 显示前 5 个 term
        sorted_terms = sorted(record.sparse_vector.items(), key=lambda x: -x[1])
        for term, weight in sorted_terms[:5]:
            safe_print(f"    {term}: {weight:.4f}")

    safe_print(f"\n  Trace stages: {[s['name'] for s in trace.stages]}")
    return records


def compare_results(jieba_records, bge_records):
    """对比两种后端的输出差异。"""
    section("对比分析")

    for j_rec, b_rec in zip(jieba_records, bge_records):
        j_terms = set(j_rec.sparse_vector.keys())
        b_terms = set(b_rec.sparse_vector.keys())

        common = j_terms & b_terms
        j_only = j_terms - b_terms
        b_only = b_terms - j_terms

        safe_print(f"\n  [{j_rec.id}]")
        safe_print(f"    jieba terms: {len(j_terms)}, BGE terms: {len(b_terms)}")
        safe_print(f"    共有: {len(common)}, jieba 独有: {len(j_only)}, BGE 独有: {len(b_only)}")

        if j_only:
            safe_print(f"    jieba 独有: {list(j_only)[:5]}")
        if b_only:
            safe_print(f"    BGE 独有: {list(b_only)[:5]}")

        # 对共有 term 的权重比较
        if common:
            safe_print("    共有 term 权重对比:")
            for term in sorted(common)[:3]:
                j_w = j_rec.sparse_vector[term]
                b_w = b_rec.sparse_vector[term]
                safe_print(f"      {term}: jieba={j_w:.4f}, BGE={b_w:.4f}")


def test_bm25_compatibility():
    """测试 BGE 输出与 BM25Indexer 的兼容性。"""
    section("BM25Indexer 兼容性测试")

    from src.ingestion.storage.bm25_indexer import BM25Indexer

    settings = load_settings()
    encoder = SparseEncoder(settings=settings, backend="bge")
    chunks = _make_chunks()
    records = encoder.encode(chunks)

    # 转换为 ChunkRecord（encode 已经返回 ChunkRecord）
    indexer = BM25Indexer()
    indexer.build(records)

    safe_print(f"  文档数: {indexer.corpus_size}")
    safe_print(f"  词汇量: {indexer.get_vocabulary_size()}")

    # 查询测试
    vocab = list(indexer.inverted_index.keys())
    safe_print(f"  词汇表（前10）: {vocab[:10]}")

    # 用实际存在的 term 查询
    if vocab:
        query_term = vocab[0]
        results = indexer.query([query_term], top_k=3)
        safe_print(f"\n  查询 ['{query_term}']:")
        for doc_id, score in results:
            safe_print(f"    {doc_id}: {score:.4f}")


def main():
    jieba_records = test_jieba_backend()
    bge_records = test_bge_backend()
    compare_results(jieba_records, bge_records)
    test_bm25_compatibility()
    print(f"\n{'='*60}")
    print("  全部手动测试完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
