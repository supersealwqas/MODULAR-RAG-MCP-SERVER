import asyncio
import os
import sys
import io

# 强制使用 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 将 src 加入路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.mcp_server.tools.get_document_summary import get_document_summary

async def main():
    print(f"\n{'=' * 60}\n  E5: get_document_summary 工具测试\n{'=' * 60}")

    # 1. 测试存在文档 (假设 test_doc_001 存在，或者你可以换成库里有的 ID)
    doc_id = "test_doc_001"
    print(f"\n[1] 获取文档摘要: '{doc_id}'")
    result = await get_document_summary(doc_id=doc_id)
    
    print(f"  isError: {result.get('isError', False)}")
    print(f"  Found: {result.get('structuredContent', {}).get('found', False)}")
    
    print("\n--- Markdown 响应 ---")
    print(result['content'][0]['text'])

    # 2. 测试错误处理
    print("\n[2] 测试空 ID 错误处理...")
    err_result = await get_document_summary(doc_id="")
    print(f"  isError: {err_result.get('isError', False)}")
    print(f"  消息: {err_result['content'][0]['text']}")

    print("\n=== E5 工具测试完成 ===")

if __name__ == "__main__":
    asyncio.run(main())
