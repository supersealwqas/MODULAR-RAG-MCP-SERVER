"""调试脚本：展示 ChromaDB 中所有 chunk 的完整内容。"""

import chromadb
import io
import json
import os
import sys

# 强制 stdout 使用 UTF-8，避免 Windows GBK 编码错误
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def main():
    db_path = "data/db/chroma"
    if not os.path.exists(db_path):
        print(f"Error: DB path {db_path} not found.")
        return

    client = chromadb.PersistentClient(path=db_path)
    try:
        collection = client.get_collection("default")
    except Exception as e:
        print(f"Error getting collection: {e}")
        return

    results = collection.get(include=["metadatas", "documents"])
    ids = results.get("ids", [])
    metadatas = results.get("metadatas", [])
    documents = results.get("documents", [])

    print(f"集合 'default' 共 {len(ids)} 个 chunk\n")
    print("=" * 70)

    for i in range(len(ids)):
        meta = metadatas[i]
        doc = documents[i]

        # 基本信息
        print(f"\n[Chunk {i + 1}/{len(ids)}]")
        print(f"  ID: {ids[i]}")
        print(f"  chunk_index: {meta.get('chunk_index', 'N/A')}")
        print(f"  来源: {meta.get('source_path', 'N/A')}")
        print(f"  标题: {meta.get('title', 'N/A')}")
        print(f"  标签: {meta.get('tags', 'N/A')}")

        # 图片信息
        images_data = meta.get("images")
        if images_data:
            try:
                images = json.loads(images_data) if isinstance(images_data, str) else images_data
                print(f"  图片数量: {len(images)}")
                for img in images:
                    print(f"    - {img['id']} (page {img['page']}, {img['position']['width']}x{img['position']['height']})")
            except Exception:
                print(f"  图片原始数据: {images_data}")
        else:
            print(f"  图片: 无")

        # 摘要
        summary = meta.get("summary", "")
        if summary:
            print(f"  摘要: {summary[:100]}...")

        # 全部元数据
        print(f"  --- 完整元数据 ---")
        print(f"  {json.dumps(meta, ensure_ascii=False, indent=4)}")

        # 完整文本
        print(f"  --- 完整文本 ({len(doc)} 字符) ---")
        print(doc)
        print(f"  --- 文本结束 ---")
        print("=" * 70)

    # 统计
    image_chunks = sum(1 for m in metadatas if m.get("images"))
    print(f"\n统计: {len(ids)} 个 chunk，其中 {image_chunks} 个包含图片")


if __name__ == "__main__":
    main()
