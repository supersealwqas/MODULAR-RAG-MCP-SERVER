# 07_embedding.py — Embedding 模型功能测试
import sys
sys.stdout.reconfigure(encoding='utf-8')

import time

print("=" * 60)
print("Embedding 模型功能测试")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. 直接使用 BGEEmbedding 类
# ─────────────────────────────────────────────
print("\n[1] 直接实例化 BGEEmbedding")
print("-" * 40)

from src.libs.embedding.bge_embedding import BGEEmbedding

emb = BGEEmbedding(
    model="bge-m3",
    dimensions=1024,
    model_path="models/bge-m3",
)

print(f"模型名称: {emb.model}")
print(f"向量维度: {emb.dimensions}")
print(f"模型路径: {emb.model_path}")
print(f"模型已加载: {emb._embedding_model is not None}")

# ─────────────────────────────────────────────
# 2. 单文本嵌入 + 向量详情
# ─────────────────────────────────────────────
print("\n[2] 单文本嵌入 (embed_single)")
print("-" * 40)

import numpy as np

text = "什么是 RAG（检索增强生成）？"
t0 = time.perf_counter()
vec = emb.embed_single(text)
elapsed = time.perf_counter() - t0


# vec2 = emb.embed_single("什么")
# vec3 = emb.embed_single("什么")
# print(vec2[:10])
# print("="*40)
# print(vec3[:10])
# print(len(vec))
# print(f"模型已加载: {emb._embedding_model is not None}")


vec_arr = np.array(vec)
print(f"输入文本:   {text}")
print(f"向量维度:   {len(vec)}")
print(f"向量均值:   {vec_arr.mean():.6f}")
print(f"向量标准差: {vec_arr.std():.6f}")
print(f"向量最大值: {vec_arr.max():.6f}")
print(f"向量最小值: {vec_arr.min():.6f}")
print(f"向量 L2 范数: {np.linalg.norm(vec_arr):.6f}")
print(f"前5个值:   {vec[:5]}")
print(f"耗时:      {elapsed:.3f}s（含首次模型加载）")

# ─────────────────────────────────────────────
# 3. 批量文本嵌入
# ─────────────────────────────────────────────
print("\n[3] 批量文本嵌入 (embed)")
print("-" * 40)

texts = [
    "RAG 是一种结合检索和生成的 AI 技术",
    "向量数据库用于存储高维嵌入向量",
    "今天天气真不错，适合出去散步",
    "BM25 是经典的稀疏检索算法",
    "SentenceTransformer 可以生成文本嵌入",
]

t0 = time.perf_counter()
vectors = emb.embed(texts)
elapsed = time.perf_counter() - t0

print(f"输入文本数: {len(texts)}")
print(f"输出向量数: {len(vectors)}")
print(f"每个向量维度: {len(vectors[0])}")
print(f"总耗时: {elapsed:.3f}s")
print(f"平均每条: {elapsed / len(texts):.3f}s")

# 打印每个向量的统计信息
print("\n各文本向量统计:")
for i, (t, v) in enumerate(zip(texts, vectors)):
    v_arr = np.array(v)
    short_text = t[:30] + "..." if len(t) > 30 else t
    print(f"  [{i}] dim={len(v):>4}  mean={v_arr.mean():>8.4f}  std={v_arr.std():>8.4f}  | {short_text}")

# ─────────────────────────────────────────────
# 4. 余弦相似度计算
# ─────────────────────────────────────────────
print("\n[4] 余弦相似度对比")
print("-" * 40)

def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def sim_label(score):
    """根据分数返回相似度等级"""
    if score >= 0.8:
        return "高度相似"
    elif score >= 0.6:
        return "较相似"
    elif score >= 0.4:
        return "一般"
    elif score >= 0.2:
        return "较不相似"
    else:
        return "完全不同"

# 语义相似的文本对
pairs = [
    ("RAG 是什么？", "检索增强生成技术介绍"),
    ("向量数据库怎么用？", "嵌入存储方案对比"),
    ("我爱吃苹果", "I like apple"),
    ("我爱吃苹果", "我讨厌苹果"),
]

print("语义相似文本对:")
for t1, t2 in pairs:
    v1 = emb.embed_single(t1)
    v2 = emb.embed_single(t2)
    sim = cosine_similarity(v1, v2)
    print(f"  [{sim:.4f}] {sim_label(sim):　<6} '{t1}' vs '{t2}'")

# 语义不相似的文本对
print("\n语义不相似文本对:")
diff_pairs = [
    ("RAG 检索增强生成", "今天午饭吃什么"),
    ("Python 编程语言", "篮球比赛结果"),
]
for t1, t2 in diff_pairs:
    v1 = emb.embed_single(t1)
    v2 = emb.embed_single(t2)
    sim = cosine_similarity(v1, v2)
    print(f"  [{sim:.4f}] {sim_label(sim):　<6} '{t1}' vs '{t2}'")

# ─────────────────────────────────────────────
# 4.1 自定义测试：输入任意两句话
# ─────────────────────────────────────────────
print("\n[4.1] 自定义两句话相似度测试")
print("-" * 40)

custom_pairs = [
    ("机器学习是人工智能的子领域", "深度学习属于机器学习的一种"),
    ("我喜欢吃苹果", "今天股市大涨"),
    ("Python 是编程语言", "蟒蛇是一种爬行动物"),
    ("向量检索提高搜索精度", "Embedding 模型将文本转为向量"),
]

for t1, t2 in custom_pairs:
    v1 = emb.embed_single(t1)
    v2 = emb.embed_single(t2)
    sim = cosine_similarity(v1, v2)
    print(f"  [{sim:.4f}] {sim_label(sim):　<6}")
    print(f"         A: {t1}")
    print(f"         B: {t2}")

# # ─────────────────────────────────────────────
# # 5. 通过工厂创建
# # ─────────────────────────────────────────────
# print("\n[5] 通过 EmbeddingFactory 创建")
# print("-" * 40)

# from src.libs.embedding.embedding_factory import EmbeddingFactory
# from src.core.settings import EmbeddingConfig

# config = EmbeddingConfig(
#     provider="bge",
#     model="bge-m3",
#     dimensions=1024,
#     model_path="models/bge-m3",
# )

# factory_emb = EmbeddingFactory.create(config)
# print(f"工厂创建类型: {type(factory_emb).__name__}")
# print(f"模型名称: {factory_emb.model}")

# # 测试工厂创建的实例能否正常工作
# test_vec = factory_emb.embed_single("测试文本")
# print(f"嵌入成功: 维度={len(test_vec)}")

# # ─────────────────────────────────────────────
# # 6. 异常处理测试
# # ─────────────────────────────────────────────
# print("\n[6] 异常处理测试")
# print("-" * 40)

# # 空输入
# try:
#     emb.embed([])
#     print("空输入: 未抛出异常（异常）")
# except ValueError as e:
#     print(f"空输入: 正确抛出 ValueError — {e}")

# # 错误的模型路径
# try:
#     bad_emb = BGEEmbedding(model_path="models/not_exist")
#     bad_emb.embed(["test"])
#     print("错误路径: 未抛出异常（异常）")
# except FileNotFoundError as e:
#     print(f"错误路径: 正确抛出 FileNotFoundError")
# except Exception as e:
#     print(f"错误路径: 抛出 {type(e).__name__} — {e}")

# # ─────────────────────────────────────────────
# # 7. 已注册提供者列表
# # ─────────────────────────────────────────────
# print("\n[7] 已注册的 Embedding 提供者")
# print("-" * 40)

# providers = EmbeddingFactory.list_providers()
# print(f"提供者列表: {providers}")

# print("\n" + "=" * 60)
# print("Embedding 测试完成")
# print("=" * 60)
