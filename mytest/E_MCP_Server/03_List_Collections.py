import asyncio
import os
import sys
import io

# 强制使用 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 将 src 加入路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.mcp_server.tools.list_collections import list_collections

async def main():
    print(f"\n{'=' * 60}\n  E4: list_collections 工具测试\n{'=' * 60}")

    print("\n[1] 获取所有集合列表...")
    result = await list_collections()
    
    print(f"  isError: {result.get('isError', False)}")
    
    # 打印 Markdown
    print("\n--- Markdown 响应 ---")
    print(result['content'][0]['text'])
    
    # 打印结构化数据
    sc = result.get("structuredContent", {})
    collections = sc.get("collections", [])
    print(f"\n  结构化数据 (共 {len(collections)} 个):")
    for coll in collections:
        print(f"    - {coll['name']} (文档数: {coll['document_count']})")

    print("\n=== E4 工具测试完成 ===")

if __name__ == "__main__":
    asyncio.run(main())
