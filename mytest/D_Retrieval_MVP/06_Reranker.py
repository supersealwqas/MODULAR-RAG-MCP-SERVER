"""D6 Reranker Core 层编排手动测试。

使用真实组件进行端到端重排序测试。
需要先执行 ingest 摄取数据并构建索引。

测试覆盖：
1. 基本重排序流程（NoneReranker 后端）
2. Fallback 机制（模拟后端异常）
3. top_k 参数传递
4. Trace 记录
5. 空输入边界处理
"""

import sys
import os

# 修复 Windows 终端中文编码问题
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.core.settings import load_settings
from src.core.query_engine.reranker import Reranker
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult
from src.libs.reranker.base_reranker import Candidate, RankedCandidate, NoneReranker


def _make_results(ids, scores=None):
    """创建测试用 RetrievalResult 列表。"""
    if scores is None:
        scores = [1.0 - i * 0.1 for i in range(len(ids))]
    return [
        RetrievalResult(
            chunk_id=cid,
            score=score,
            text=f"text_{cid}",
            metadata={"source": f"doc_{cid}", "collection": "test"},
        )
        for cid, score in zip(ids, scores)
    ]


def test_basic_rerank(reranker: Reranker):
    """测试1: 基本重排序（NoneReranker 后端）。"""
    print("\n" + "=" * 60)
    print("测试1: 基本重排序（NoneReranker 后端）")
    print("=" * 60)

    candidates = _make_results(["chunk_A", "chunk_B", "chunk_C"])
    result = reranker.rerank("什么是语言模型？", candidates)

    print(f"  输入候选数: {len(candidates)}")
    print(f"  输出结果数: {len(result['results'])}")
    print(f"  Fallback: {result['fallback']}")
    print(f"  耗时: {result['elapsed_ms']}ms")

    assert result["fallback"] is False
    assert len(result["results"]) == 3

    for i, r in enumerate(result["results"]):
        print(f"    [{i+1}] {r.chunk_id} score={r.score:.4f}")

    print("\n✅ 基本重排序测试通过")


def test_fallback_mechanism():
    """测试2: Fallback 机制（模拟后端异常）。"""
    print("\n" + "=" * 60)
    print("测试2: Fallback 机制（模拟后端异常）")
    print("=" * 60)

    from unittest.mock import MagicMock
    from src.libs.reranker.base_reranker import BaseReranker

    settings = load_settings()
    # 创建一个会抛异常的 mock 后端
    mock_backend = MagicMock(spec=BaseReranker)
    mock_backend.rerank.side_effect = RuntimeError("模拟后端异常")

    reranker = Reranker(settings, backend=mock_backend)
    candidates = _make_results(["chunk_A", "chunk_B", "chunk_C"])
    result = reranker.rerank("什么是语言模型？", candidates)

    print(f"  Fallback: {result['fallback']}")
    print(f"  输出结果数: {len(result['results'])}")

    assert result["fallback"] is True
    assert len(result["results"]) == 3
    # 回退时保持原始顺序
    assert result["results"][0].chunk_id == "chunk_A"
    assert result["results"][1].chunk_id == "chunk_B"
    assert result["results"][2].chunk_id == "chunk_C"

    for i, r in enumerate(result["results"]):
        print(f"    [{i+1}] {r.chunk_id} score={r.score:.4f} (原始顺序)")

    print("\n✅ Fallback 机制测试通过")


def test_top_k(reranker: Reranker):
    """测试3: top_k 参数。"""
    print("\n" + "=" * 60)
    print("测试3: top_k 参数")
    print("=" * 60)

    candidates = _make_results(["A", "B", "C", "D", "E"])

    for k in [1, 3, 5]:
        result = reranker.rerank("test", candidates, top_k=k)
        print(f"  top_k={k} → 返回 {len(result['results'])} 条结果")
        assert len(result["results"]) <= k

    print("\n✅ top_k 测试通过")


def test_trace_recording(reranker: Reranker):
    """测试4: Trace 记录。"""
    print("\n" + "=" * 60)
    print("测试4: Trace 记录")
    print("=" * 60)

    candidates = _make_results(["chunk_A", "chunk_B"])
    trace = TraceContext()

    reranker.rerank("什么是语言模型？", candidates, trace=trace)

    stages = [s for s in trace.stages if s["name"] == "reranker"]
    assert len(stages) == 1

    stage = stages[0]
    print(f"  记录阶段: {stage['name']}")
    print(f"  Fallback: {stage['fallback']}")
    print(f"  输入候选数: {stage['input_count']}")
    print(f"  输出结果数: {stage['output_count']}")
    print(f"  总耗时: {stage['elapsed_ms']}ms")

    print("\n✅ Trace 记录测试通过")


def test_empty_input(reranker: Reranker):
    """测试5: 空输入边界处理。"""
    print("\n" + "=" * 60)
    print("测试5: 空输入边界处理")
    print("=" * 60)

    result = reranker.rerank("test", [])
    assert result["results"] == []
    assert result["fallback"] is False
    print("  空候选列表 → 空结果 ✅")

    print("\n✅ 空输入边界测试通过")


def test_with_real_hybrid_search():
    """测试6: 与 HybridSearch 集成（使用真实组件）。"""
    print("\n" + "=" * 60)
    print("测试6: HybridSearch + Reranker 集成")
    print("=" * 60)

    settings = load_settings()
    hs = HybridSearch(settings)
    reranker = Reranker(settings)

    query = "什么是语言模型？"

    # 先检索
    search_result = hs.search(query, top_k=5)
    print(f"  HybridSearch 返回: {len(search_result)} 条结果")

    if not search_result:
        print("  ⚠️ 无检索结果，跳过集成测试")
        return

    # 再重排序
    result = reranker.rerank(query, search_result, top_k=3)
    print(f"  Reranker 返回: {len(result['results'])} 条结果")
    print(f"  Fallback: {result['fallback']}")

    for i, r in enumerate(result["results"]):
        print(f"    [{i+1}] score={r.score:.4f} chunk_id={r.chunk_id}")
        print(f"        text={r.text[:60]}...")

    print("\n✅ HybridSearch + Reranker 集成测试通过")


def main():
    """运行所有 D6 测试。"""
    print("=" * 60)
    print("D6 Reranker Core 层编排 完整测试")
    print("=" * 60)

    settings = load_settings()
    print(f"Rerank provider: {settings.rerank.provider}")
    print(f"Rerank enabled: {settings.rerank.enabled}")
    print(f"Rerank top_k: {settings.rerank.top_k}")

    reranker = Reranker(settings)

    test_basic_rerank(reranker)
    test_fallback_mechanism()
    test_top_k(reranker)
    test_trace_recording(reranker)
    test_empty_input(reranker)
    test_with_real_hybrid_search()

    print("\n" + "=" * 60)
    print("✅ 所有 D6 测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
