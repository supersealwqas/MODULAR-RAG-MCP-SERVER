"""C11 BM25Indexer 手动测试脚本。

测试 BM25 倒排索引的构建、查询、持久化、增量更新、文档移除等。
使用 SparseEncoder 生成真实 sparse_vector 进行端到端测试。
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.settings import load_settings
from src.core.trace.trace_context import TraceContext
from src.core.types import ChunkRecord
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.storage.bm25_indexer import BM25Indexer


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


def _make_chunks_data() -> list:
    """创建测试用文本数据。"""
    return [
        ("doc1", "RAG 检索增强生成是一种结合信息检索与大语言模型的技术框架。"),
        ("doc2", "向量数据库是 RAG 系统的核心组件，负责存储和检索文档的稠密向量表示。"),
        ("doc3", "BM25 是一种经典的稀疏检索算法，基于词频和逆文档频率进行文档排序。"),
        ("doc4", "混合检索结合了稠密向量检索和稀疏关键词检索的优势，通过 RRF 算法融合结果。"),
        ("doc5", "Chunk 切分策略直接影响检索质量，RecursiveCharacterTextSplitter 是常用的切分方案。"),
    ]


def _encode_to_records(texts: list) -> list:
    """使用 SparseEncoder 将文本编码为 ChunkRecord 列表。"""
    settings = load_settings()
    encoder = SparseEncoder(settings=settings)

    chunks_data = texts
    records = []
    for doc_id, text in chunks_data:
        from src.core.types import Chunk
        chunk = Chunk(
            id=doc_id,
            text=text,
            metadata={"source_path": "/test.pdf"},
            source_ref="doc_001",
        )
        encoded = encoder.encode([chunk])
        if encoded:
            records.append(encoded[0])

    return records


def test_build_and_query():
    """测试索引构建和查询。"""
    section("索引构建与查询")

    records = _encode_to_records(_make_chunks_data())
    indexer = BM25Indexer()
    indexer.build(records)

    safe_print(f"  文档数: {indexer.corpus_size}")
    safe_print(f"  词汇量: {indexer.get_vocabulary_size()}")
    safe_print(f"  平均文档长度: {indexer.avg_doc_length:.2f}")

    # 展示部分词汇表，方便选择查询词
    vocab = list(indexer.inverted_index.keys())
    safe_print(f"\n  词汇表（前20）: {vocab[:20]}")

    # 使用实际存在的 term 进行查询
    test_terms = []
    for candidate in ["rag", "检索", "bm25", "向量", "算法", "数据库", "混合"]:
        if candidate in indexer.inverted_index:
            test_terms.append(candidate)

    # 单 term 查询
    for term in test_terms[:3]:
        results = indexer.query([term], top_k=3)
        safe_print(f"\n  查询 ['{term}']:")
        for doc_id, score in results:
            safe_print(f"    {doc_id}: {score:.4f}")

    # 多 term 查询
    if len(test_terms) >= 2:
        results = indexer.query(test_terms[:2], top_k=3)
        safe_print(f"\n  查询 {test_terms[:2]}:")
        for doc_id, score in results:
            safe_print(f"    {doc_id}: {score:.4f}")


def test_idf_analysis():
    """测试 IDF 分析。"""
    section("IDF 分析")

    records = _encode_to_records(_make_chunks_data())
    indexer = BM25Indexer()
    indexer.build(records)

    safe_print(f"  {'term':<15} {'df':>5} {'idf':>10}")
    safe_print(f"  {'-'*32}")

    sorted_terms = sorted(
        indexer.inverted_index.items(),
        key=lambda x: x[1]["idf"],
        reverse=True,
    )
    for term, entry in sorted_terms[:15]:
        df = len(entry["postings"])
        safe_print(f"  {term:<15} {df:>5} {entry['idf']:>10.4f}")


def test_persistence_roundtrip():
    """测试持久化 roundtrip。"""
    section("持久化 Roundtrip")

    records = _encode_to_records(_make_chunks_data())
    indexer = BM25Indexer()
    indexer.build(records)

    # 使用实际存在的 term 查询
    query_terms = [t for t in ["检索", "向量", "rag"] if t in indexer.inverted_index][:2]
    original_results = indexer.query(query_terms)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bm25_index.pkl")
        indexer.save(path)
        safe_print(f"  保存到: {path}")
        safe_print(f"  文件大小: {os.path.getsize(path)} bytes")

        loaded = BM25Indexer()
        loaded.load(path)
        loaded_results = loaded.query(query_terms)

        safe_print(f"\n  查询: {query_terms}")
        safe_print(f"  原始结果: {original_results[:3]}")
        safe_print(f"  加载结果: {loaded_results[:3]}")
        safe_print(f"  结果一致: {original_results == loaded_results}")


def test_incremental_update():
    """测试增量更新。"""
    section("增量更新")

    records = _encode_to_records(_make_chunks_data()[:3])
    indexer = BM25Indexer()
    indexer.build(records)
    safe_print(f"  初始文档数: {indexer.corpus_size}")

    # 添加新文档
    new_texts = [
        ("doc6", "Prompt Engineering 是与大语言模型交互的关键技术。"),
        ("doc7", "Fine-tuning 可以让预训练模型适应特定领域的任务。"),
    ]
    new_records = _encode_to_records(new_texts)
    indexer.add_documents(new_records)
    safe_print(f"  增量更新后: {indexer.corpus_size}")

    # 使用实际存在的 term 查询
    for term in ["Prompt", "技术", "模型"]:
        if term in indexer.inverted_index:
            results = indexer.query([term], top_k=3)
            safe_print(f"\n  查询 ['{term}']:")
            for doc_id, score in results:
                safe_print(f"    {doc_id}: {score:.4f}")
            break


def test_document_removal():
    """测试文档移除。"""
    section("文档移除")

    records = _encode_to_records(_make_chunks_data())
    indexer = BM25Indexer()
    indexer.build(records)
    safe_print(f"  移除前: {indexer.corpus_size} 文档")

    # 找到 doc3 独有的 term
    doc3_terms = [
        t for t, e in indexer.inverted_index.items()
        if any(p["chunk_id"] == "doc3" for p in e["postings"])
        and len(e["postings"]) == 1
    ]

    indexer.remove_document("doc3")
    safe_print(f"  移除 doc3 后: {indexer.corpus_size} 文档")

    if doc3_terms:
        query_term = doc3_terms[0]
        results = indexer.query([query_term])
        chunk_ids = [r[0] for r in results]
        safe_print(f"  查询 ['{query_term}'] 结果: {chunk_ids}")
        safe_print(f"  doc3 已移除: {'doc3' not in chunk_ids}")
    else:
        safe_print("  doc3 无独有 term，跳过查询验证")


def test_trace_recording():
    """测试 Trace 记录。"""
    section("Trace 记录")

    records = _encode_to_records(_make_chunks_data())
    trace = TraceContext(trace_type="ingestion")

    indexer = BM25Indexer()
    indexer.build(records, trace=trace)

    safe_print(f"  trace_id: {trace.trace_id}")
    safe_print(f"  stages: {len(trace.stages)}")
    for stage in trace.stages:
        safe_print(f"  - {stage['name']}: {stage}")


def main():
    test_build_and_query()
    test_idf_analysis()
    test_persistence_roundtrip()
    test_incremental_update()
    test_document_removal()
    test_trace_recording()
    print(f"\n{'='*60}")
    print("  全部手动测试完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
