# 15_reranker_compare.py — Cross-Encoder vs LLM Reranker 真实性能对比
import sys
sys.stdout.reconfigure(encoding='utf-8')

import time

from src.core.settings import load_settings
from src.libs.llm.openai_llm import OpenAILLM
from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from src.libs.reranker.llm_reranker import LLMReranker
from src.libs.reranker.base_reranker import Candidate

print("=" * 60)
print("Reranker 性能对比测试")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. 初始化
# ─────────────────────────────────────────────
print("\n[1] 初始化")
print("-" * 60)

settings = load_settings()
llm = OpenAILLM(
    model=settings.llm.model,
    api_key=settings.llm.api_key,
    base_url=settings.llm.base_url,
    temperature=settings.llm.temperature,
    max_tokens=settings.llm.max_tokens,
)

cross_encoder = CrossEncoderReranker(model="models/bge-reranker-large")
llm_reranker = LLMReranker(llm=llm)

# 预热 Cross-Encoder（首次加载模型）
cross_encoder.rerank("预热", [Candidate(id="warmup", text="预热文本", score=0.5)])
print("Cross-Encoder 预热完成")

# ─────────────────────────────────────────────
# 2. 测试数据
# ─────────────────────────────────────────────
candidates = [
    Candidate(id="chunk_001",
              text="Ollama 是一个本地运行大语言模型的工具，支持 Llama、Qwen、Gemma 等模型。",
              score=0.65),
    Candidate(id="chunk_002",
              text="本项目采用 ChromaDB 作为向量数据库，支持持久化存储和元数据过滤。",
              score=0.72),
    Candidate(id="chunk_003",
              text="配置 Ollama：在 settings.yaml 中设置 ollama.model 和 ollama.base_url。",
              score=0.58),
    Candidate(id="chunk_004",
              text="RAG（检索增强生成）通过检索外部知识库来增强大语言模型的回答能力。",
              score=0.55),
    Candidate(id="chunk_005",
              text="BGE-M3 是 BAAI 发布的多功能嵌入模型，支持稠密检索和稀疏检索。",
              score=0.60),
]

queries = [
    "如何配置 Ollama？",
    "什么是 RAG？",
    "向量数据库用的什么？",
]

# ─────────────────────────────────────────────
# 3. 逐查询对比
# ─────────────────────────────────────────────
print(f"\n[2] 逐查询对比（{len(queries)} 个查询 × {len(candidates)} 个候选）")
print("-" * 60)

cross_times = []
llm_times = []

for query in queries:
    print(f"\n  查询: {query}")

    # Cross-Encoder
    t0 = time.time()
    ce_results = cross_encoder.rerank(query, candidates, top_k=3)
    t1 = time.time()
    ce_ms = (t1 - t0) * 1000
    cross_times.append(ce_ms)

    # LLM Reranker
    t0 = time.time()
    try:
        llm_results = llm_reranker.rerank(query, candidates, top_k=3)
        t1 = time.time()
        llm_ms = (t1 - t0) * 1000
        llm_times.append(llm_ms)
        llm_ok = True
    except Exception as e:
        llm_ms = 0
        llm_ok = False
        print(f"  ⚠ LLM Reranker 失败: {e}")

    # 对比输出
    print(f"  {'方法':<18} {'耗时':>8}  {'Top1':<10} {'Top2':<10} {'Top3':<10}")
    print(f"  {'─' * 60}")

    ce_top = " → ".join(f"{r.id}({r.rerank_score:.2f})" for r in ce_results)
    print(f"  {'Cross-Encoder':<18} {ce_ms:>6.0f}ms  {ce_top}")

    if llm_ok:
        llm_top = " → ".join(f"{r.id}({r.rerank_score:.2f})" for r in llm_results)
        print(f"  {'LLM Reranker':<18} {llm_ms:>6.0f}ms  {llm_top}")
    else:
        print(f"  {'LLM Reranker':<18}    失败")

# ─────────────────────────────────────────────
# 4. 汇总
# ─────────────────────────────────────────────
print(f"\n[3] 汇总")
print("-" * 60)

ce_avg = sum(cross_times) / len(cross_times)
print(f"  Cross-Encoder 平均: {ce_avg:.0f} ms/次")

if llm_times:
    llm_avg = sum(llm_times) / len(llm_times)
    print(f"  LLM Reranker 平均:  {llm_avg:.0f} ms/次")
    print(f"  速度倍数: Cross-Encoder 快 {llm_avg / ce_avg:.0f} 倍")
else:
    print(f"  LLM Reranker: 全部失败，无法对比")

print("\n" + "=" * 60)
print("对比测试完成")
print("=" * 60)
