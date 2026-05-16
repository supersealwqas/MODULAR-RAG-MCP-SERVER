"""D2 DenseRetriever 手动测试。

使用真实 BGE-M3 Embedding + ChromaStore 进行端到端稠密向量检索。
需要先执行 ingest 摄取数据到 ChromaDB。
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.core.settings import load_settings
from src.core.query_engine.dense_retriever import DenseRetriever


def main():
    """手动测试 DenseRetriever。"""
    print("=" * 60)
    print("D2 DenseRetriever 手动测试")
    print("=" * 60)

    # 加载配置
    settings = load_settings()
    print(f"Embedding: {settings.embedding.provider} / {settings.embedding.model}")
    print(f"VectorStore: {settings.vector_store.provider}")

    # 创建 DenseRetriever（依赖注入）
    retriever = DenseRetriever(settings)

    # 测试查询
    queries = [
        "如何配置 Ollama？",
        "BGE-M3 模型的向量维度",
        "什么是 RAG？",
    ]

    for query in queries:
        print(f"\n{'─' * 40}")
        print(f"Query: {query}")
        results = retriever.retrieve(query, top_k=3)
        print(f"Results: {len(results)} 条")
        for i, r in enumerate(results):
            print(f"  [{i+1}] score={r.score:.4f} chunk_id={r.chunk_id}")
            print(f"      text={r.text[:80]}...")
            print(f"      source={r.metadata.get('source_path', 'N/A')}")

    print(f"\n{'=' * 60}")
    print("测试完成")


if __name__ == "__main__":
    main()
