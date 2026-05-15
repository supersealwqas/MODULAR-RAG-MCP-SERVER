# 13_llm_reranker.py — 使用真实大模型测试 LLM Reranker
import sys
sys.stdout.reconfigure(encoding='utf-8')

import time

from src.core.settings import load_settings
from src.libs.llm.openai_llm import OpenAILLM
from src.libs.reranker.llm_reranker import LLMReranker
from src.libs.reranker.base_reranker import Candidate

print("=" * 60)
print("LLM Reranker 真实大模型测试")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. 加载配置，创建 LLM 实例
# ─────────────────────────────────────────────
print("\n[1] 加载 LLM 配置")
print("-" * 60)

settings = load_settings()
llm = OpenAILLM(
    model=settings.llm.model,
    api_key=settings.llm.api_key,
    base_url=settings.llm.base_url,
    temperature=settings.llm.temperature,
    max_tokens=settings.llm.max_tokens,
)
print(f"模型: {settings.llm.model}")
print(f"端点: {settings.llm.base_url}")

# ─────────────────────────────────────────────
# 2. 创建 Reranker
# ─────────────────────────────────────────────
print("\n[2] 创建 LLM Reranker")
print("-" * 60)

reranker = LLMReranker(llm=llm)
print(f"Prompt 文件: {reranker._prompt_path}")

# ─────────────────────────────────────────────
# 3. 构造测试候选文档
# ─────────────────────────────────────────────
print("\n[3] 构造测试数据")
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
# 4. 逐个查询测试 rerank
# ─────────────────────────────────────────────
print("\n[4] Rerank 测试")
print("-" * 60)

for query in queries:
    print(f"\n  查询: {query}")
    print(f"  {'─' * 50}")

    # 原始排序（按检索分数）
    print(f"  原始排序（按检索分数）:")
    sorted_by_score = sorted(candidates, key=lambda c: c.score, reverse=True)
    for i, c in enumerate(sorted_by_score):
        print(f"    [{i+1}] {c.id} score={c.score:.2f}  {c.text[:35]}...")

    # LLM Rerank
    t0 = time.time()
    try:
        results = reranker.rerank(query, candidates, top_k=3)
        t1 = time.time()
        print(f"\n  LLM Rerank 结果 ({(t1-t0)*1000:.0f} ms):")
        for i, r in enumerate(results):
            print(f"    [{i+1}] {r.id} rerank_score={r.rerank_score:.2f}  "
                  f"原分={r.original_score:.2f}  {r.text[:35]}...")
    except RuntimeError as e:
        print(f"  ❌ Rerank 失败: {e}")

# ─────────────────────────────────────────────
# 5. 测试 prompt 内容
# ─────────────────────────────────────────────
print("\n[5] Prompt 模板内容")
print("-" * 60)

template = reranker._load_prompt_template()
# 只显示前 300 字符
print(template[:300])
if len(template) > 300:
    print(f"... (共 {len(template)} 字符)")

print("\n" + "=" * 60)
print("LLM Reranker 测试完成")
print("=" * 60)
