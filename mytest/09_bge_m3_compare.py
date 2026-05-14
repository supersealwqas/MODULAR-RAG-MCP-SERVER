# 09_bge_m3_compare.py — 对比 SentenceTransformer vs FlagEmbedding 两种方式
import sys
sys.stdout.reconfigure(encoding='utf-8')

import time
import numpy as np

print("=" * 60)
print("BGE-M3 双后端对比：SentenceTransformer vs FlagEmbedding")
print("=" * 60)

texts = [
    "RAG 是一种结合检索和生成的 AI 技术",
    "检索增强生成（Retrieval-Augmented Generation）",
    "今天天气不错，适合出去散步",
    "向量数据库用于存储高维嵌入向量",
    "BM25 是经典的稀疏检索算法",
]

# ─────────────────────────────────────────────
# 1. SentenceTransformer（当前项目使用）
# ─────────────────────────────────────────────
print("\n[1] SentenceTransformer 后端")
print("-" * 40)

from sentence_transformers import SentenceTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"设备: {device}")

t0 = time.perf_counter()
st_model = SentenceTransformer("models/bge-m3", device=device)
st_load = time.perf_counter() - t0
print(f"模型加载: {st_load:.2f}s")

t0 = time.perf_counter()
st_vecs = st_model.encode(texts)
st_time = time.perf_counter() - t0
print(f"编码耗时: {st_time:.3f}s  ({len(texts)} 条)")
print(f"单条平均: {st_time / len(texts):.4f}s")
print(f"向量维度: {len(st_vecs[0])}")
print(f"L2 范数:  {np.linalg.norm(st_vecs[0]):.4f}")

# ─────────────────────────────────────────────
# 2. FlagEmbedding（支持 sparse + colbert）
# ─────────────────────────────────────────────
print("\n[2] FlagEmbedding 后端")
print("-" * 40)

from FlagEmbedding import BGEM3FlagModel

t0 = time.perf_counter()
flag_model = BGEM3FlagModel("models/bge-m3", use_fp16=True)
flag_load = time.perf_counter() - t0
print(f"模型加载: {flag_load:.2f}s")

t0 = time.perf_counter()
flag_out = flag_model.encode(
    texts,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
flag_time = time.perf_counter() - t0
print(f"编码耗时: {flag_time:.3f}s  ({len(texts)} 条)")
print(f"单条平均: {flag_time / len(texts):.4f}s")

flag_dense = flag_out["dense_vecs"]
flag_sparse = flag_out["lexical_weights"]
flag_colbert = flag_out["colbert_vecs"]

print(f"稠密维度: {len(flag_dense[0])}")
print(f"L2 范数:  {np.linalg.norm(flag_dense[0]):.4f}")

# ─────────────────────────────────────────────
# 3. 稠密向量一致性对比
# ─────────────────────────────────────────────
print("\n[3] 稠密向量一致性对比")
print("-" * 40)

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"{'文本':<30} {'余弦相似度':>10} {'L2差(ST)':>10} {'L2差(Flag)':>10}")
print("-" * 65)

for i, t in enumerate(texts):
    sim = cosine_sim(st_vecs[i], flag_dense[i])
    st_norm = np.linalg.norm(st_vecs[i])
    flag_norm = np.linalg.norm(flag_dense[i])
    short = t[:28] + ".." if len(t) > 28 else t
    print(f"{short:<30} {sim:>10.6f} {st_norm:>10.4f} {flag_norm:>10.4f}")

# ─────────────────────────────────────────────
# 4. 稀疏向量详情（仅 FlagEmbedding）
# ─────────────────────────────────────────────
print("\n[4] 稀疏向量详情（仅 FlagEmbedding 支持）")
print("-" * 40)

tokenizer = flag_model.tokenizer

for i, (t, sp) in enumerate(zip(texts, flag_sparse)):
    short = t[:30] + "..." if len(t) > 30 else t
    nonzero = len(sp)
    weights = list(sp.values())
    print(f"\n  [{i}] {short}")
    print(f"      非零token: {nonzero}  权重: [{min(weights):.4f}, {max(weights):.4f}]")

    top_tokens = sorted(sp.items(), key=lambda x: x[1], reverse=True)[:5]
    keywords = []
    for token_id, weight in top_tokens:
        token = tokenizer.decode([int(token_id)])
        keywords.append(f"{token}({weight:.2f})")
    print(f"      关键词: {', '.join(keywords)}")

# ─────────────────────────────────────────────
# 5. ColBERT 向量详情（仅 FlagEmbedding）
# ─────────────────────────────────────────────
print("\n[5] ColBERT 向量详情（仅 FlagEmbedding 支持）")
print("-" * 40)

for i, (t, cv) in enumerate(zip(texts, flag_colbert)):
    cv_arr = np.array(cv)
    short = t[:30] + "..." if len(t) > 30 else t
    print(f"  [{i}] shape=({cv_arr.shape[0]:>2}, {cv_arr.shape[1]})  | {short}")

# ─────────────────────────────────────────────
# 6. 相似度检索对比：三种向量
# ─────────────────────────────────────────────
print("\n[6] 相似度检索对比：三种向量")
print("-" * 40)

def colbert_sim(vecs_a, vecs_b):
    """ColBERT MaxSim"""
    a, b = np.array(vecs_a), np.array(vecs_b)
    sim_matrix = np.dot(a, b.T)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    sim_matrix = sim_matrix / (a_norm * b_norm.T)
    return sim_matrix.max(axis=1).mean()

def sparse_sim(sp_a, sp_b):
    """稀疏向量余弦相似度"""
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

print(f"查询: {query}\n")

# SentenceTransformer 结果
q_st = st_model.encode([query])
c_st = st_model.encode(candidates)

print("SentenceTransformer（仅 Dense）:")
for i, cand in enumerate(candidates):
    d = cosine_sim(q_st[0], c_st[i])
    print(f"  [{i}] dense={d:.4f}  | {cand[:40]}")

# FlagEmbedding 结果
q_flag = flag_model.encode([query], return_dense=True, return_sparse=True)
c_flag = flag_model.encode(candidates, return_dense=True, return_sparse=True)

print("\nFlagEmbedding（Dense + Sparse 混合）:")
for i, cand in enumerate(candidates):
    d = cosine_sim(q_flag["dense_vecs"][0], c_flag["dense_vecs"][i])
    s = sparse_sim(q_flag["lexical_weights"][0], c_flag["lexical_weights"][i])
    hybrid = 0.6 * d + 0.4 * s
    print(f"  [{i}] hybrid={hybrid:.4f}  (dense={d:.4f}, sparse={s:.4f})  | {cand[:40]}")

# ─────────────────────────────────────────────
# 7. 性能对比汇总
# ─────────────────────────────────────────────
print("\n[7] 性能对比汇总")
print("-" * 40)

print(f"{'指标':<20} {'SentenceTransformer':>20} {'FlagEmbedding':>20}")
print("-" * 65)
print(f"{'模型加载':.<20} {st_load:>18.2f}s {flag_load:>18.2f}s")
print(f"{'编码(5条)':.<20} {st_time:>18.3f}s {flag_time:>18.3f}s")
print(f"{'单条平均':.<20} {st_time/len(texts):>18.4f}s {flag_time/len(texts):>18.4f}s")
print(f"{'输出类型':.<20} {'dense only':>20} {'dense+sparse+colbert':>20}")
print(f"{'向量维度':.<20} {len(st_vecs[0]):>20} {len(flag_dense[0]):>20}")

print("\n[结论]")
print("  - 稠密向量两者一致（余弦相似度≈1.0）")
print("  - FlagEmbedding 额外提供 sparse 和 colbert，适合混合检索")
print("  - SentenceTransformer 更轻量，只需稠密向量时性能更好")

print("\n" + "=" * 60)
print("对比测试完成")
print("=" * 60)
