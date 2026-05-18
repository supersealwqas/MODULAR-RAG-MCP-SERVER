import asyncio
import os
import sys
import time
import io

# 强制使用 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 将 src 加入路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.mcp_server.tools.query_knowledge_hub import query_knowledge_hub
from src.core.settings import load_settings

async def main():
    print(f"\n{'=' * 60}\n  E3: query_knowledge_hub 工具测试\n{'=' * 60}")

    settings = load_settings()
    
    # 测试常规查询
    query = "什么是 RAG"
    print(f"\n[1] 执行检索查询: '{query}'")
    
    start = time.time()
    result = await query_knowledge_hub(query=query, top_k=3, settings=settings)
    elapsed = time.time() - start
    
    print(f"  耗时: {elapsed:.2f}s")
    print(f"  isError: {result.get('isError', False)}")
    
    # 打印 Markdown 响应
    content = result.get("content", [])
    if content:
        print("\n--- Markdown 响应 ---")
        print(content[0].get("text", "")[:500] + "...")
        print("--- 结束 ---\n")
    
    # 打印结构化引用
    sc = result.get("structuredContent", {})
    citations = sc.get("citations", [])
    print(f"  获取到 {len(citations)} 条结构化引用:")
    for c in citations:
        print(f"    [{c['index']}] {c['source']} (Score: {c['score']})")

    # 测试空结果
    print("\n[2] 测试空结果处理...")
    empty_result = await query_knowledge_hub(query="一个不存在的关键词xyz123", settings=settings)
    print(f"  内容预览: {empty_result['content'][0]['text'][:100]}...")

    print("\n=== E3 工具测试完成 ===")

if __name__ == "__main__":
    asyncio.run(main())
