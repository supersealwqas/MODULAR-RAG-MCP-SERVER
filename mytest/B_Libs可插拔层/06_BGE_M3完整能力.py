# 08_bge_m3_full.py — BGE-M3 三种向量测试（dense / sparse / colbert）
import sys
sys.stdout.reconfigure(encoding='utf-8')

import time
import numpy as np

print("=" * 60)
print("BGE-M3 全功能测试：稠密 / 稀疏 / ColBERT 向量")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. 加载模型
# ─────────────────────────────────────────────
print("\n[1] 加载 BGE-M3 模型（FlagEmbedding）")
print("-" * 40)

from FlagEmbedding import BGEM3FlagModel

t0 = time.perf_counter()
model = BGEM3FlagModel("models/bge-m3", use_fp16=True)
elapsed = time.perf_counter() - t0
print(f"模型加载耗时: {elapsed:.2f}s")

# ─────────────────────────────────────────────
# 2. 编码：获取三种向量
# ─────────────────────────────────────────────
print("\n[2] 编码文本，获取三种向量")
print("-" * 40)

texts = [
    "RAG 是一种结合检索和生成的 AI 技术",
    "检索增强生成（Retrieval-Augmented Generation）",
    "今天天气不错，适合出去散步",
]

t0 = time.perf_counter()
output = model.encode(
    texts,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
elapsed = time.perf_counter() - t0

dense_vecs = output["dense_vecs"]
sparse_vecs = output["lexical_weights"]
colbert_vecs = output["colbert_vecs"]

print(f"编码耗时: {elapsed:.3f}s")
print(f"文本数量: {len(texts)}")

# ─────────────────────────────────────────────
# 3. 稠密向量 (Dense)
# ─────────────────────────────────────────────
print("\n[3] 稠密向量 (Dense)")
print("-" * 40)

for i, (t, v) in enumerate(zip(texts, dense_vecs)):
    v_arr = np.array(v)
    short_text = t[:35] + "..." if len(t) > 35 else t
    print(f"  [{i}] dim={len(v):>4}  L2={np.linalg.norm(v_arr):.4f}  | {short_text}")

# ─────────────────────────────────────────────
# 4. 稀疏向量 (Sparse / Lexical Weights)
# ─────────────────────────────────────────────
print("\n[4] 稀疏向量 (Sparse / Lexical Weights)")
print("-" * 40)

for i, (t, sp) in enumerate(zip(texts, sparse_vecs)):
    short_text = t[:35] + "..." if len(t) > 35 else t
    # sp 是 {token_id: weight} 的字典
    nonzero = len(sp)
    weights = list(sp.values())
    print(f"\n  [{i}] {short_text}")
    print(f"      非零元素数: {nonzero}")
    print(f"      权重范围:   [{min(weights):.4f}, {max(weights):.4f}]")
    print(f"      权重均值:   {np.mean(weights):.4f}")

    # 解码 token（展示权重最高的词）
    tokenizer = model.tokenizer
    top_tokens = sorted(sp.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"      Top-5 关键词:")
    for token_id, weight in top_tokens:
        token = tokenizer.decode([int(token_id)])
        print(f"        {weight:.4f}  →  '{token}'")

# ─────────────────────────────────────────────
# 5. ColBERT 向量（多向量表示）
# ─────────────────────────────────────────────
print("\n[5] ColBERT 向量（多向量表示）")
print("-" * 40)

for i, (t, cv) in enumerate(zip(texts, colbert_vecs)):
    short_text = t[:35] + "..." if len(t) > 35 else t
    cv_arr = np.array(cv)
    print(f"  [{i}] shape=({cv_arr.shape[0]}, {cv_arr.shape[1]})  | {short_text}")
    print(f"      token 数: {cv_arr.shape[0]}  每个 token 维度: {cv_arr.shape[1]}")

# ─────────────────────────────────────────────
# 6. 相似度对比：三种向量
# ─────────────────────────────────────────────
print("\n[6] 相似度对比：三种向量")
print("-" * 40)

def cosine_sim(a, b):
    """余弦相似度"""
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def colbert_sim(vecs_a, vecs_b):
    """ColBERT MaxSim 相似度"""
    a, b = np.array(vecs_a), np.array(vecs_b)
    # 计算所有 token 对的余弦相似度，取每行最大值再求和
    sim_matrix = np.dot(a, b.T)
    # 归一化
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    sim_matrix = sim_matrix / (a_norm * b_norm.T)
    return sim_matrix.max(axis=1).mean()

def sparse_sim(sp_a, sp_b):
    """稀疏向量余弦相似度"""
    # 合并所有 token_id
    all_tokens = set(sp_a.keys()) | set(sp_b.keys())
    if not all_tokens:
        return 0.0
    dot = sum(sp_a.get(t, 0) * sp_b.get(t, 0) for t in all_tokens)
    norm_a = sum(v ** 2 for v in sp_a.values()) ** 0.5
    norm_b = sum(v ** 2 for v in sp_b.values()) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

# 测试文本对
pairs = [
    ("RAG 是什么？", "检索增强生成技术介绍"),
    ("向量数据库怎么用？", "嵌入存储方案对比"),
    ("今天天气好", "明天适合郊游"),
    ("RAG 检索增强生成", "今天午饭吃什么"),
]

print(f"{'文本对':<35} {'Dense':>8} {'Sparse':>8} {'ColBERT':>8}")
print("-" * 65)

for t1, t2 in pairs:
    # 编码
    out = model.encode([t1, t2], return_dense=True, return_sparse=True, return_colbert_vecs=True)

    d_sim = cosine_sim(out["dense_vecs"][0], out["dense_vecs"][1])
    s_sim = sparse_sim(out["lexical_weights"][0], out["lexical_weights"][1])
    c_sim = colbert_sim(out["colbert_vecs"][0], out["colbert_vecs"][1])

    label = f"{t1[:12]} vs {t2[:12]}"
    print(f"{label:<35} {d_sim:>8.4f} {s_sim:>8.4f} {c_sim:>8.4f}")

# ─────────────────────────────────────────────
# 7. 混合检索得分（加权融合）
# ─────────────────────────────────────────────
print("\n[7] 混合检索得分（Dense + Sparse 加权融合）")
print("-" * 40)

query = "什么是 RAG？"
candidates = [
    "RAG（检索增强生成）是一种结合外部知识库和大语言模型的技术",
    "向量数据库用于高效存储和检索嵌入向量",
    "今天天气不错，适合户外活动",
]

q_out = model.encode([query], return_dense=True, return_sparse=True)
c_out = model.encode(candidates, return_dense=True, return_sparse=True)

print(f"查询: {query}\n")
for i, cand in enumerate(candidates):
    d = cosine_sim(q_out["dense_vecs"][0], c_out["dense_vecs"][i])
    s = sparse_sim(q_out["lexical_weights"][0], c_out["lexical_weights"][i])
    # 混合得分：dense 0.6 + sparse 0.4
    hybrid = 0.6 * d + 0.4 * s
    print(f"  [{i}] hybrid={hybrid:.4f}  (dense={d:.4f}, sparse={s:.4f})")
    print(f"      {cand}")

print("\n" + "=" * 60)
print("BGE-M3 全功能测试完成")
print("=" * 60)
