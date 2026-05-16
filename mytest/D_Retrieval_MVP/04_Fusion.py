"""D4 Fusion RRF 手动测试。

测试 RRF (Reciprocal Rank Fusion) 算法，将多个检索结果融合为统一排名。

测试覆盖：
1. 基本 RRF 融合（Dense + Sparse）
2. top_k 截断
3. k 参数影响
4. Trace 记录
5. 确定性输出
"""

import sys
import os

# 修复 Windows 终端中文编码问题
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.core.types import RetrievalResult
from src.core.query_engine.fusion import Fusion
from src.core.trace.trace_context import TraceContext


def _make_results(ids, scores=None):
    """创建测试用 RetrievalResult 列表。"""
    if scores is None:
        scores = [1.0 - i * 0.1 for i in range(len(ids))]
    return [
        RetrievalResult(
            chunk_id=cid,
            score=score,
            text=f"text_{cid}",
            metadata={"source": f"doc_{cid}"},
        )
        for cid, score in zip(ids, scores)
    ]


def test_basic_fusion():
    """测试1: 基本 RRF 融合。"""
    print("\n" + "=" * 60)
    print("测试1: 基本 RRF 融合（Dense + Sparse）")
    print("=" * 60)

    fusion = Fusion(k=60)

    # 模拟 Dense 和 Sparse 检索结果
    dense_results = _make_results(["chunk_A", "chunk_B", "chunk_C"])
    sparse_results = _make_results(["chunk_B", "chunk_A", "chunk_D"])

    results = fusion.fuse([dense_results, sparse_results])

    print(f"\n  Dense 排名: A, B, C")
    print(f"  Sparse 排名: B, A, D")
    print(f"  RRF 融合结果: {len(results)} 条")
    for i, r in enumerate(results):
        print(f"    [{i+1}] {r.chunk_id} score={r.score:.6f}")

    assert len(results) == 4
    # A 和 B 都出现两次，排名靠前
    assert results[0].chunk_id in ["chunk_A", "chunk_B"]
    assert results[1].chunk_id in ["chunk_A", "chunk_B"]

    print("\n✅ 基本融合测试通过")


def test_top_k():
    """测试2: top_k 截断。"""
    print("\n" + "=" * 60)
    print("测试2: top_k 截断")
    print("=" * 60)

    fusion = Fusion(k=60)
    results_list = _make_results(["A", "B", "C", "D", "E"])

    for k in [1, 3, 5]:
        results = fusion.fuse([results_list], top_k=k)
        print(f"  top_k={k} → 返回 {len(results)} 条结果")
        assert len(results) <= k

    print("\n✅ top_k 截断测试通过")


def test_k_parameter():
    """测试3: k 参数影响。"""
    print("\n" + "=" * 60)
    print("测试3: k 参数影响")
    print("=" * 60)

    results_list = _make_results(["A", "B", "C"])

    for k in [10, 60, 100]:
        fusion = Fusion(k=k)
        results = fusion.fuse([results_list])
        gap = results[0].score - results[1].score
        print(f"  k={k}: 分数差距={gap:.6f}")

    # k 越小，排名差距越大
    fusion_small = Fusion(k=10)
    fusion_large = Fusion(k=100)
    gap_small = fusion_small.fuse([results_list])[0].score - fusion_small.fuse([results_list])[1].score
    gap_large = fusion_large.fuse([results_list])[0].score - fusion_large.fuse([results_list])[1].score
    assert gap_small > gap_large

    print("\n✅ k 参数测试通过")


def test_trace_recording():
    """测试4: Trace 记录。"""
    print("\n" + "=" * 60)
    print("测试4: Trace 记录")
    print("=" * 60)

    fusion = Fusion(k=60)
    r1 = _make_results(["A", "B"])
    r2 = _make_results(["B", "A"])
    trace = TraceContext()

    fusion.fuse([r1, r2], trace=trace)

    stages = [s for s in trace.stages if s["name"] == "fusion"]
    assert len(stages) == 1

    stage = stages[0]
    print(f"  记录阶段: {stage['name']}")
    print(f"  方法: {stage['method']}")
    print(f"  k: {stage['k']}")
    print(f"  输入排名数: {stage['input_rankings']}")
    print(f"  输出结果数: {stage['output_count']}")

    print("\n✅ Trace 记录测试通过")


def test_deterministic():
    """测试5: 确定性输出。"""
    print("\n" + "=" * 60)
    print("测试5: 确定性输出")
    print("=" * 60)

    fusion = Fusion(k=60)
    r1 = _make_results(["A", "B", "C"])
    r2 = _make_results(["B", "A", "D"])

    # 多次运行应产生相同结果
    results1 = fusion.fuse([r1, r2])
    results2 = fusion.fuse([r1, r2])

    for r1_item, r2_item in zip(results1, results2):
        assert r1_item.chunk_id == r2_item.chunk_id
        assert r1_item.score == r2_item.score

    print("  多次运行结果一致 ✅")

    # 对称性
    r1 = _make_results(["A", "B"])
    r2 = _make_results(["B", "A"])
    results = fusion.fuse([r1, r2])
    assert results[0].score == results[1].score
    print("  对称排名分数一致 ✅")

    print("\n✅ 确定性输出测试通过")


def test_serialization():
    """测试6: RetrievalResult 序列化。"""
    print("\n" + "=" * 60)
    print("测试6: RetrievalResult 序列化")
    print("=" * 60)

    fusion = Fusion(k=60)
    r1 = _make_results(["A", "B"])
    results = fusion.fuse([r1])

    r = results[0]
    d = r.to_dict()
    assert "chunk_id" in d
    assert "score" in d
    assert "text" in d
    assert "metadata" in d
    print(f"  to_dict: chunk_id={d['chunk_id']}, score={d['score']:.6f}")

    restored = RetrievalResult.from_dict(d)
    assert restored.chunk_id == r.chunk_id
    assert restored.score == r.score
    print(f"  from_dict: 还原一致 ✅")

    print("\n✅ 序列化测试通过")


def main():
    """运行所有 D4 测试。"""
    print("=" * 60)
    print("D4 Fusion RRF 完整测试")
    print("=" * 60)

    test_basic_fusion()
    test_top_k()
    test_k_parameter()
    test_trace_recording()
    test_deterministic()
    test_serialization()

    print("\n" + "=" * 60)
    print("✅ 所有 D4 测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
