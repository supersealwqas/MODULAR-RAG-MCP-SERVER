"""D3 SparseRetriever 手动测试。

使用真实 BM25Indexer + ChromaStore 进行端到端稀疏检索。
需要先执行 ingest 摄取数据并构建 BM25 索引。

测试覆盖：
1. 基本检索流程（keywords → BM25 → VectorStore → RetrievalResult）
2. top_k 参数传递
3. Trace 记录
4. 空查询边界处理
5. RetrievalResult 序列化
"""

import sys
import os

# 修复 Windows 终端中文编码问题
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.core.settings import load_settings
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.core.trace.trace_context import TraceContext


def test_basic_retrieval(retriever: SparseRetriever, proc: QueryProcessor):
    """测试1: 基本检索流程。"""
    print("\n" + "=" * 60)
    print("测试1: 基本检索流程")
    print("=" * 60)

    queries = [
        "如何配置 Ollama？",
        "BGE-M3 模型的向量维度",
        "什么是 RAG？",
    ]

    for query in queries:
        # 提取关键词
        pq = proc.process(query)
        print(f"\n{'─' * 40}")
        print(f"Query: {query}")
        print(f"Keywords: {pq.keywords}")

        results = retriever.retrieve(pq.keywords, top_k=3)
        print(f"Results: {len(results)} 条")
        for i, r in enumerate(results):
            print(f"  [{i+1}] score={r.score:.4f} chunk_id={r.chunk_id}")
            print(f"      text={r.text[:60]}...")
            print(f"      source={r.metadata.get('source_path', 'N/A')}")

    print("\n✅ 基本检索测试通过")


def test_top_k_parameter(retriever: SparseRetriever, proc: QueryProcessor):
    """测试2: top_k 参数传递。"""
    print("\n" + "=" * 60)
    print("测试2: top_k 参数传递")
    print("=" * 60)

    query = "什么是语言模型？"
    pq = proc.process(query)

    for k in [1, 3, 5]:
        results = retriever.retrieve(pq.keywords, top_k=k)
        print(f"  top_k={k} → 返回 {len(results)} 条结果")
        assert len(results) <= k, f"top_k={k} 但返回了 {len(results)} 条结果"

    print("✅ top_k 参数测试通过")


def test_trace_recording(retriever: SparseRetriever, proc: QueryProcessor):
    """测试3: Trace 记录。"""
    print("\n" + "=" * 60)
    print("测试3: Trace 记录")
    print("=" * 60)

    query = "什么是语言模型？"
    pq = proc.process(query)
    trace = TraceContext()

    results = retriever.retrieve(pq.keywords, top_k=3, trace=trace)

    stages = [s for s in trace.stages if s["name"] == "sparse_retrieval"]
    assert len(stages) == 1, f"应记录1个 sparse_retrieval 阶段，实际: {len(stages)}"

    stage = stages[0]
    print(f"  记录阶段: {stage['name']}")
    print(f"  关键词数量: {stage['keyword_count']}")
    print(f"  结果数量: {stage['result_count']}")
    print(f"  BM25耗时: {stage['bm25_ms']}ms")
    print(f"  Lookup耗时: {stage['lookup_ms']}ms")
    print(f"  总耗时: {stage['elapsed_ms']}ms")

    assert stage["result_count"] == len(results)
    assert "bm25_ms" in stage
    assert "lookup_ms" in stage
    assert "elapsed_ms" in stage

    print("✅ Trace 记录测试通过")


def test_empty_query(retriever: SparseRetriever):
    """测试4: 空查询边界处理。"""
    print("\n" + "=" * 60)
    print("测试4: 空查询边界处理")
    print("=" * 60)

    results = retriever.retrieve([])
    assert results == [], f"空查询应返回空列表，实际: {results}"
    print("  空关键词 → 空列表 ✅")

    print("✅ 空查询边界测试通过")


def test_serialization(retriever: SparseRetriever, proc: QueryProcessor):
    """测试5: RetrievalResult 序列化。"""
    print("\n" + "=" * 60)
    print("测试5: RetrievalResult 序列化")
    print("=" * 60)

    query = "什么是语言模型？"
    pq = proc.process(query)
    results = retriever.retrieve(pq.keywords, top_k=1)

    if not results:
        print("  ⚠️ 无检索结果，跳过序列化测试")
        return

    r = results[0]

    d = r.to_dict()
    assert "chunk_id" in d
    assert "score" in d
    assert "text" in d
    assert "metadata" in d
    print(f"  to_dict: chunk_id={d['chunk_id']}, score={d['score']:.4f}")

    from src.core.types import RetrievalResult
    restored = RetrievalResult.from_dict(d)
    assert restored.chunk_id == r.chunk_id
    assert restored.score == r.score
    print(f"  from_dict: 还原一致 ✅")

    print("✅ 序列化测试通过")


def main():
    """运行所有 D3 测试。"""
    print("=" * 60)
    print("D3 SparseRetriever 完整测试")
    print("=" * 60)

    # 加载配置
    settings = load_settings()
    print(f"BM25 索引目录: data/db/bm25")
    print(f"VectorStore: {settings.vector_store.provider}")
    print(f"默认 top_k: {settings.retrieval.top_k}")

    # 创建 QueryProcessor 和 SparseRetriever
    proc = QueryProcessor(settings)
    retriever = SparseRetriever(settings)

    # 运行所有测试
    test_basic_retrieval(retriever, proc)
    test_top_k_parameter(retriever, proc)
    test_trace_recording(retriever, proc)
    test_empty_query(retriever)
    test_serialization(retriever, proc)

    # 汇总
    print("\n" + "=" * 60)
    print("✅ 所有 D3 测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
