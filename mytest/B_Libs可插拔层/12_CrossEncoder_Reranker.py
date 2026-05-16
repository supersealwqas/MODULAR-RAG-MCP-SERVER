# 14_cross_encoder_reranker.py — 使用本地 bge-reranker-large 模型测试 Cross-Encoder Reranker
import sys
sys.stdout.reconfigure(encoding='utf-8')

import time

from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from src.libs.reranker.base_reranker import Candidate

print("=" * 60)
print("Cross-Encoder Reranker 真实模型测试")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. 创建 Reranker（使用本地 bge-reranker-large）
# ─────────────────────────────────────────────
print("\n[1] 加载 Cross-Encoder 模型")
print("-" * 60)

reranker = CrossEncoderReranker(model="models/bge-reranker-large")
print(f"模型: {reranker._model_name}")

# 触发模型加载
reranker._load_model()
print(f"模型类型: {type(reranker._model).__name__}")

# ─────────────────────────────────────────────
# 2. 构造测试候选文档
# ─────────────────────────────────────────────
print("\n[2] 构造测试数据")
print("-" * 60)

candidates = [
    Candidate(
        id="chunk_001",
        text="Ollama 是一个本地运行大语言模型的工具，支持 Llama、Qwen、Gemma 等模型。"
             "安装后使用 ollama serve 启动服务，默认监听 localhost:11434。",
        score=0.65,
    ),
    Candidate(
        id="chunk_002",
        text="本项目采用 ChromaDB 作为向量数据库，支持持久化存储和元数据过滤。"
             "数据目录默认为 data/db/chroma/。",
        score=0.72,
    ),
    Candidate(
        id="chunk_003",
        text="配置 Ollama：在 settings.yaml 中设置 ollama.model 为模型名称，"
             "ollama.base_url 为服务地址（默认 http://localhost:11434），"
             "然后运行 ollama pull <model> 下载模型。",
        score=0.58,
    ),
    Candidate(
        id="chunk_004",
        text="RAG（检索增强生成）通过检索外部知识库来增强大语言模型的回答能力，"
             "减少幻觉并提升事实准确性。",
        score=0.55,
    ),
    Candidate(
        id="chunk_005",
        text="BGE-M3 是 BAAI 发布的多功能嵌入模型，支持稠密检索、稀疏检索和多向量检索，"
             "在中英文任务上表现优异。模型维度为 1024。",
        score=0.60,
    ),
]

queries = [
    "如何配置 Ollama？",
    "什么是 RAG？",
    "向量数据库用的什么？",
]

print(f"候选文档数: {len(candidates)}")
print(f"测试查询数: {len(queries)}")

# ─────────────────────────────────────────────
# 3. 逐个查询测试 rerank
# ─────────────────────────────────────────────
print("\n[3] Rerank 测试")
print("-" * 60)

for query in queries:
    print(f"\n  查询: {query}")
    print(f"  {'─' * 50}")

    # 原始排序（按检索分数）
    print(f"  原始排序（按检索分数）:")
    sorted_by_score = sorted(candidates, key=lambda c: c.score, reverse=True)
    for i, c in enumerate(sorted_by_score):
        print(f"    [{i+1}] {c.id} score={c.score:.2f}  {c.text[:35]}...")

    # Cross-Encoder Rerank
    t0 = time.time()
    try:
        results = reranker.rerank(query, candidates, top_k=3)
        t1 = time.time()
        print(f"\n  Cross-Encoder Rerank ({(t1-t0)*1000:.0f} ms):")
        for i, r in enumerate(results):
            print(f"    [{i+1}] {r.id} rerank_score={r.rerank_score:.4f}  "
                  f"原分={r.original_score:.2f}  {r.text[:35]}...")
    except RuntimeError as e:
        print(f"  ❌ Rerank 失败: {e}")

# ─────────────────────────────────────────────
# 4. 与 LLM Reranker 对比速度
# ─────────────────────────────────────────────
print("\n[4] 性能对比")
print("-" * 60)

# 预热
reranker.rerank("测试", candidates[:1])

# 计时
times = []
for query in queries:
    t0 = time.time()
    reranker.rerank(query, candidates)
    t1 = time.time()
    times.append((t1 - t0) * 1000)

avg_ms = sum(times) / len(times)
print(f"  Cross-Encoder 平均耗时: {avg_ms:.0f} ms/次")
print(f"  (对比: LLM Reranker 约 17000 ms/次)")

print("\n" + "=" * 60)
print("Cross-Encoder Reranker 测试完成")
print("=" * 60)
