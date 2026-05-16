# 04_evaluator.py — 评估器功能测试
import sys
sys.stdout.reconfigure(encoding='utf-8')

from src.libs.evaluator.custom_evaluator import (
    CustomEvaluator,
    compute_hit_rate,
    compute_mrr,
    compute_precision_at_k,
)
from src.libs.evaluator.base_evaluator import EvalCase

print("=" * 60)
print("评估器功能测试")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. 单指标计算
# ─────────────────────────────────────────────
print("\n[1] 单指标计算")
print("-" * 40)

retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
golden = ["doc2", "doc6"]

hit = compute_hit_rate(retrieved, golden, k=5)
print(f"Hit Rate@5: {hit}  (检索到 doc2，命中)")

mrr = compute_mrr(retrieved, golden)
print(f"MRR: {mrr:.4f}  (doc2 排第2，1/2=0.5)")

precision = compute_precision_at_k(retrieved, golden, k=5)
print(f"Precision@5: {precision:.4f}  (5个中命中1个)")

# ─────────────────────────────────────────────
# 2. 边界情况
# ─────────────────────────────────────────────
print("\n[2] 边界情况")
print("-" * 40)

# 空 golden
hit = compute_hit_rate(["doc1"], [], k=5)
print(f"空 golden 的 Hit Rate: {hit}")

# 空 retrieved
hit = compute_hit_rate([], ["doc1"], k=5)
print(f"空 retrieved 的 Hit Rate: {hit}")

# 第一个就命中
mrr = compute_mrr(["doc1", "doc2"], ["doc1"])
print(f"第1个命中 MRR: {mrr}  (1/1=1.0)")

# 完全没命中
mrr = compute_mrr(["doc1", "doc2"], ["doc3"])
print(f"完全没命中 MRR: {mrr}")

# ─────────────────────────────────────────────
# 3. 批量评估
# ─────────────────────────────────────────────
print("\n[3] 批量评估")
print("-" * 40)

evaluator = CustomEvaluator(k=5)

cases = [
    EvalCase(
        query="什么是 RAG？",
        retrieved_ids=["doc_rag_1", "doc_other_1", "doc_other_2"],
        golden_ids=["doc_rag_1"],
    ),
    EvalCase(
        query="Python 装饰器怎么用？",
        retrieved_ids=["doc_other_1", "doc_decorator_1", "doc_other_2"],
        golden_ids=["doc_decorator_1"],
    ),
    EvalCase(
        query="机器学习基础",
        retrieved_ids=["doc_other_1", "doc_other_2", "doc_other_3"],
        golden_ids=["doc_ml_1"],
    ),
    EvalCase(
        query="深度学习框架",
        retrieved_ids=["doc_dl_1", "doc_dl_2", "doc_other_1"],
        golden_ids=["doc_dl_1", "doc_dl_2"],
    ),
]

report = evaluator.evaluate(cases)

print(f"评估用例数: {report.total_cases}")
print()

for i, result in enumerate(report.results, 1):
    print(f"  用例 {i}: {result.query}")
    for metric, value in result.metrics.items():
        print(f"    {metric}: {value:.4f}")

print()
print("汇总指标:")
for metric, value in report.summary.items():
    print(f"  {metric}: {value:.4f}")

# ─────────────────────────────────────────────
# 4. 单条评估便捷方法
# ─────────────────────────────────────────────
print("\n[4] 单条评估便捷方法")
print("-" * 40)

case = EvalCase(
    query="测试查询",
    retrieved_ids=["doc_a", "doc_b", "doc_c"],
    golden_ids=["doc_b"],
)
result = evaluator.evaluate_single(case)
print(f"查询: {result.query}")
for metric, value in result.metrics.items():
    print(f"  {metric}: {value:.4f}")

print("\n" + "=" * 60)
print("评估器测试完成")
print("=" * 60)
