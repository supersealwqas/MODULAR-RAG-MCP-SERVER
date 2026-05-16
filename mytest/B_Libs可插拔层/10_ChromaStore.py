# 12_chroma_store.py — 模拟真实向量化入库 + 语义检索全流程
import sys
sys.stdout.reconfigure(encoding='utf-8')

import time
from pathlib import Path

from src.libs.vector_store.chroma_store import ChromaStore
from src.libs.vector_store.base_vector_store import VectorRecord
from src.libs.embedding.bge_embedding import BGEEmbedding
from src.libs.splitter.recursive_splitter import RecursiveSplitter

print("=" * 60)
print("向量化入库 + 语义检索 全流程测试")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. 准备：加载 Embedding 模型 + Splitter
# ─────────────────────────────────────────────
print("\n[1] 加载模型")
print("-" * 60)

embedding = BGEEmbedding(model="bge-m3", dimensions=1024, model_path="models/bge-m3")
splitter = RecursiveSplitter(chunk_size=1000, chunk_overlap=200)

# 读取 README 作为测试文档
readme_path = Path(__file__).parent.parent / "README.md"
with open(readme_path, "r", encoding="utf-8") as f:
    doc_text = f.read()
print(f"文档: {readme_path.name} ({len(doc_text)} 字符)")

# ─────────────────────────────────────────────
# 2. 切分文档
# ─────────────────────────────────────────────
print("\n[2] 文档切分")
print("-" * 60)

chunks = splitter.split_text(doc_text)
print(f"切分结果: {len(chunks)} 块")
for i, c in enumerate(chunks[:3]):
    print(f"  [{i}] len={len(c)}  {c[:40].replace(chr(10), ' ')}...")
print(f"  ... 共 {len(chunks)} 块")

# ─────────────────────────────────────────────
# 3. 向量化（BGE-M3）
# ─────────────────────────────────────────────
print("\n[3] 向量化（BGE-M3 dense embedding）")
print("-" * 60)

t0 = time.time()
vectors = embedding.embed(chunks)
t1 = time.time()
print(f"编码 {len(chunks)} 个文本块: {(t1-t0)*1000:.0f} ms")
print(f"向量维度: {len(vectors[0])}")

# ─────────────────────────────────────────────
# 4. 构建 VectorRecord 并写入 ChromaStore
# ─────────────────────────────────────────────
print("\n[4] 写入 ChromaStore")
print("-" * 60)

# 使用项目配置的持久化目录
persist_dir = "data/db/chroma"
store = ChromaStore(collection_name="readme_demo", persist_directory=persist_dir)

records = []
for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
    records.append(VectorRecord(
        id=f"readme_{i:04d}",
        vector=vec,
        text=chunk,
        metadata={
            "source": "README.md",
            "chunk_index": i,
            "chunk_len": len(chunk),
        },
    ))

t0 = time.time()
count = store.upsert(records)
t1 = time.time()
print(f"写入 {count} 条记录: {(t1-t0)*1000:.0f} ms")
print(f"持久化目录: {persist_dir}")
print(f"向量维度: {len(vectors[0])}")

# ─────────────────────────────────────────────
# 5. 语义检索测试
# ─────────────────────────────────────────────
print("\n[5] 语义检索测试")
print("-" * 60)

queries = [
    "这个项目是做什么的？",
    "怎么配置 Ollama？",
    "BGE-M3 模型有什么特点？",
    "如何运行测试？",
    "MCP Server 怎么用？",
]

for query in queries:
    t0 = time.time()
    q_vec = embedding.embed_single(query)
    results = store.query(vector=q_vec, top_k=2)
    t1 = time.time()

    print(f"\n  查询: {query}")
    print(f"  耗时: {(t1-t0)*1000:.0f} ms")
    for i, r in enumerate(results):
        preview = r.text[:60].replace("\n", " ")
        print(f"    [{i}] score={r.score:.4f} chunk[{r.metadata['chunk_index']}] {preview}...")

# ─────────────────────────────────────────────
# 6. 元数据过滤检索
# ─────────────────────────────────────────────
print("\n[6] 元数据过滤检索")
print("-" * 60)

q_vec = embedding.embed_single("向量数据库")
results_all = store.query(vector=q_vec, top_k=3)
results_filtered = store.query(
    vector=q_vec, top_k=3,
    filters={"source": "README.md"},
)
print(f"无过滤: {len(results_all)} 条")
print(f"source=README.md: {len(results_filtered)} 条")

# ─────────────────────────────────────────────
# 7. 持久化验证（重新打开）
# ─────────────────────────────────────────────
print("\n[7] 持久化验证（重新打开连接）")
print("-" * 60)

store2 = ChromaStore(collection_name="readme_demo", persist_directory=persist_dir)
print(f"重新打开后记录数: {store2.count()}")

q_vec = embedding.embed_single("项目架构")
results2 = store2.query(vector=q_vec, top_k=1)
print(f"查询「项目架构」: score={results2[0].score:.4f}")
print(f"  {results2[0].text[:80].replace(chr(10), ' ')}...")

print("\n" + "=" * 60)
print("全流程测试完成")
print(f"数据已持久化到: {persist_dir}/")
print("下次启动可直接查询，无需重新入库")
print("=" * 60)
