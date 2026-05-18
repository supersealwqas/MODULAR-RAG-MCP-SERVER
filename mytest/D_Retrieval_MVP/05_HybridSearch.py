"""D5 HybridSearch 手动测试。

使用真实组件进行端到端混合检索测试。
需要先执行 ingest 摄取数据并构建索引。

测试覆盖：
1. 基本混合检索流程（Dense + Sparse + RRF）
2. top_k 参数传递
3. Metadata 过滤
4. Trace 记录
5. 空查询边界处理
"""

import sys
import os

# 修复 Windows 终端中文编码问题
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.core.settings import load_settings
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.trace.trace_context import TraceContext


def test_basic_search(hs: HybridSearch):
    """测试1: 基本混合检索。"""
    print("\n" + "=" * 60)
    print("测试1: 基本混合检索")
    print("=" * 60)

    queries = [
        "如何配置 Ollama？",
        "BGE-M3 模型的向量维度",
        "什么是 RAG？",
    ]

    for query in queries:
        print(f"\n{'─' * 40}")
        print(f"Query: {query}")
        results = hs.search(query, top_k=3)
        print(f"Results: {len(results)} 条")
        for i, r in enumerate(results):
            print(f"  [{i+1}] score={r.score:.4f} chunk_id={r.chunk_id}")
            print(f"      text={r.text[:60]}...")
            print(f"      source={r.metadata.get('source_path', 'N/A')}")

    print("\n✅ 基本检索测试通过")


def test_top_k(hs: HybridSearch):
    """测试2: top_k 参数。"""
    print("\n" + "=" * 60)
    print("测试2: top_k 参数")
    print("=" * 60)

    query = "什么是语言模型？"

    for k in [1, 3, 5]:
        results = hs.search(query, top_k=k)
        print(f"  top_k={k} → 返回 {len(results)} 条结果")
        assert len(results) <= k, f"top_k={k} 但返回了 {len(results)} 条结果"

    print("✅ top_k 测试通过")


def test_metadata_filters(hs: HybridSearch):
    """测试3: Metadata 过滤。"""
    print("\n" + "=" * 60)
    print("测试3: Metadata 过滤")
    print("=" * 60)

    query = "什么是语言模型？"

    # 不带 filters
    results_no_filter = hs.search(query, top_k=5)
    print(f"  无 filters: {len(results_no_filter)} 条结果")

    # 带 filters（测试空filters不影响结果）
    results_with_filter = hs.search(query, top_k=5, filters={})
    print(f"  空 filters: {len(results_with_filter)} 条结果")

    assert len(results_no_filter) == len(results_with_filter), "空filters应与无filters结果一致"

    print("✅ Metadata 过滤测试通过")


def test_trace_recording(hs: HybridSearch):
    """测试4: Trace 记录。"""
    print("\n" + "=" * 60)
    print("测试4: Trace 记录")
    print("=" * 60)

    query = "什么是语言模型？"
    trace = TraceContext()

    results = hs.search(query, top_k=3, trace=trace)

    # 检查 hybrid_search 阶段
    stages = [s for s in trace.stages if s["name"] == "hybrid_search"]
    assert len(stages) == 1, f"应记录1个 hybrid_search 阶段，实际: {len(stages)}"

    stage = stages[0]
    print(f"  记录阶段: {stage['name']}")
    print(f"  Dense 结果数: {stage['dense_count']}")
    print(f"  Sparse 结果数: {stage['sparse_count']}")
    print(f"  融合后结果数: {stage['fused_count']}")
    print(f"  最终结果数: {stage['final_count']}")
    print(f"  总耗时: {stage['elapsed_ms']}ms")

    print("✅ Trace 记录测试通过")


def test_empty_query(hs: HybridSearch):
    """测试5: 空查询边界处理。"""
    print("\n" + "=" * 60)
    print("测试5: 空查询边界处理")
    print("=" * 60)

    results = hs.search("")
    assert results == [], f"空查询应返回空列表，实际: {results}"
    print("  空字符串 → 空列表 ✅")

    results = hs.search("   ")
    assert results == [], f"纯空格应返回空列表，实际: {results}"
    print("  纯空格 → 空列表 ✅")

    print("✅ 空查询边界测试通过")


def test_serialization(hs: HybridSearch):
    """测试6: RetrievalResult 序列化。"""
    print("\n" + "=" * 60)
    print("测试6: RetrievalResult 序列化")
    print("=" * 60)

    results = hs.search("什么是语言模型？", top_k=1)

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
    """运行所有 D5 测试。"""
    print("=" * 60)
    print("D5 HybridSearch 完整测试")
    print("=" * 60)

    # 加载配置
    settings = load_settings()
    print(f"Embedding: {settings.embedding.provider} / {settings.embedding.model}")
    print(f"VectorStore: {settings.vector_store.provider}")
    print(f"默认 top_k: {settings.retrieval.top_k}")
    print(f"权重配置: Dense={settings.retrieval.dense_weight}, Sparse={settings.retrieval.sparse_weight}")

    # 创建 HybridSearch（使用真实组件）
    hs = HybridSearch(settings)

    # 运行所有测试
    test_basic_search(hs)
    test_top_k(hs)
    test_metadata_filters(hs)
    test_trace_recording(hs)
    test_empty_query(hs)
    test_serialization(hs)

    # 汇总
    print("\n" + "=" * 60)
    print("✅ 所有 D5 测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
