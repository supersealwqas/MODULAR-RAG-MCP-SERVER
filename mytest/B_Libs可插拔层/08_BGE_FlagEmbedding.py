# 10_bge_flag.py — 测试改写后的 BGEEmbedding（基于 FlagEmbedding）
import sys
sys.stdout.reconfigure(encoding='utf-8')

import time
import numpy as np

print("=" * 60)
print("BGEEmbedding (FlagEmbedding) 功能测试")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. 基本加载和稠密向量
# ─────────────────────────────────────────────
print("\n[1] 加载模型 + 稠密向量 (embed)")
print("-" * 40)

from src.libs.embedding.bge_embedding import BGEEmbedding

emb = BGEEmbedding(
    model="bge-m3",
    dimensions=1024,
    model_path="models/bge-m3",
    use_fp16=True,
)

texts = [
    "RAG 是一种结合检索和生成的 AI 技术",
    "检索增强生成（Retrieval-Augmented Generation）",
    "今天天气不错，适合出去散步",
]

t0 = time.perf_counter()
dense_vecs = emb.embed(texts)
elapsed = time.perf_counter() - t0

print(f"模型已加载: {emb._embedding_model is not None}")
print(f"文本数: {len(texts)}")
print(f"向量维度: {len(dense_vecs[0])}")
print(f"编码耗时: {elapsed:.3f}s")
print(f"单条平均: {elapsed / len(texts):.4f}s")

# ─────────────────────────────────────────────
# 2. 稠密 + 稀疏向量
# ─────────────────────────────────────────────
print("\n[2] 稠密 + 稀疏向量 (embed_with_sparse)")
print("-" * 40)

t0 = time.perf_counter()
dense, sparse = emb.embed_with_sparse(texts)
elapsed = time.perf_counter() - t0

print(f"编码耗时: {elapsed:.3f}s")
for i, (t, sp) in enumerate(zip(texts, sparse)):
    short = t[:30] + "..." if len(t) > 30 else t
    nonzero = len(sp)
    weights = list(sp.values())
    print(f"\n  [{i}] {short}")
    print(f"      非零token: {nonzero}  权重范围: [{min(weights):.4f}, {max(weights):.4f}]")

# ─────────────────────────────────────────────
# 3. 稠密 + ColBERT 向量
# ─────────────────────────────────────────────
print("\n[3] 稠密 + ColBERT 向量 (embed_with_colbert)")
print("-" * 40)

t0 = time.perf_counter()
dense, colbert = emb.embed_with_colbert(texts)
elapsed = time.perf_counter() - t0

print(f"编码耗时: {elapsed:.3f}s")
for i, (t, cv) in enumerate(zip(texts, colbert)):
    short = t[:30] + "..." if len(t) > 30 else t
    arr = np.array(cv)
    print(f"  [{i}] shape=({arr.shape[0]:>2}, {arr.shape[1]})  | {short}")

# ─────────────────────────────────────────────
# 4. 全部三种向量
# ─────────────────────────────────────────────
print("\n[4] 全部三种向量 (embed_all)")
print("-" * 40)

t0 = time.perf_counter()
all_vecs = emb.embed_all(texts)
elapsed = time.perf_counter() - t0

print(f"编码耗时: {elapsed:.3f}s")
print(f"返回键: {list(all_vecs.keys())}")
print(f"dense 数量: {len(all_vecs['dense'])}")
print(f"sparse 数量: {len(all_vecs['sparse'])}")
print(f"colbert 数量: {len(all_vecs['colbert'])}")

# ─────────────────────────────────────────────
# 5. 单文本嵌入 (embed_single)
# ─────────────────────────────────────────────
print("\n[5] 单文本嵌入 (embed_single)")
print("-" * 40)

t0 = time.perf_counter()
vec = emb.embed_single("什么是 RAG？")
elapsed = time.perf_counter() - t0

print(f"向量维度: {len(vec)}")
print(f"编码耗时: {elapsed:.4f}s")

# ─────────────────────────────────────────────
# 6. 混合检索示例
# ─────────────────────────────────────────────
print("\n[6] 混合检索示例")
print("-" * 40)

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def sparse_sim(sp_a, sp_b):
    all_tokens = set(sp_a.keys()) | set(sp_b.keys())
    if not all_tokens:
        return 0.0
    dot = sum(sp_a.get(t, 0) * sp_b.get(t, 0) for t in all_tokens)
    norm_a = sum(v ** 2 for v in sp_a.values()) ** 0.5
    norm_b = sum(v ** 2 for v in sp_b.values()) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

query = "什么是 RAG？"
candidates = [
    "RAG（检索增强生成）是一种结合外部知识库和大语言模型的技术",
    "向量数据库用于高效存储和检索嵌入向量",
    "今天天气不错，适合户外活动",
]

# 一次调用获取所有向量
q_all = emb.embed_all([query])
c_all = emb.embed_all(candidates)

print(f"查询: {query}\n")
for i, cand in enumerate(candidates):
    d = cosine_sim(q_all["dense"][0], c_all["dense"][i])
    s = sparse_sim(q_all["sparse"][0], c_all["sparse"][i])
    hybrid = 0.6 * d + 0.4 * s
    print(f"  [{i}] hybrid={hybrid:.4f}  (dense={d:.4f}, sparse={s:.4f})")
    print(f"      {cand}")

# ─────────────────────────────────────────────
# 7. 通过工厂创建
# ─────────────────────────────────────────────
print("\n[7] 通过 EmbeddingFactory 创建")
print("-" * 40)

from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.core.settings import EmbeddingConfig

config = EmbeddingConfig(
    provider="bge",
    model="bge-m3",
    dimensions=1024,
    model_path="models/bge-m3",
)

factory_emb = EmbeddingFactory.create(config)
print(f"类型: {type(factory_emb).__name__}")

# 工厂创建的实例也支持新方法
d, s = factory_emb.embed_with_sparse(["测试"])
print(f"embed_with_sparse 可用: dense_dim={len(d[0])}, sparse_tokens={len(s[0])}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
