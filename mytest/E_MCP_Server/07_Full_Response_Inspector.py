import asyncio
import json
import os
import sys
import io

# 强制使用 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 将项目根目录加入路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.mcp_server.tools.query_knowledge_hub import query_knowledge_hub
from src.mcp_server.tools.list_collections import list_collections
from src.mcp_server.tools.get_document_summary import get_document_summary
from src.core.settings import load_settings

def print_mcp_output(title: str, output: dict):
    print(f"\n{'=' * 80}")
    print(f"  功能: {title}")
    print(f"{'=' * 80}")
    
    # 模拟 JSON-RPC 响应包装
    mcp_response = {
        "jsonrpc": "2.0",
        "result": output
    }
    
    print("\n[原始响应字典 (JSON 格式)]")
    print(json.dumps(mcp_response, indent=2, ensure_ascii=False))
    
    print("\n[Markdown 内容预览]")
    content = output.get("content", [])
    for item in content:
        if item.get("type") == "text":
            text = item.get("text", "")
            # 只显示前 500 字符和后 200 字符，避免刷屏
            if len(text) > 700:
                print(text[:500])
                print("\n... (中间内容已省略) ...\n")
                print(text[-200:])
            else:
                print(text)
    
    print("\n" + "-"*80)

async def run_inspector():
    print("\n正在启动 MCP 功能输出深度审查 (进程内模拟)...")
    settings = load_settings()

    # 1. list_collections
    try:
        print("\n[1/3] 正在测试 list_collections...")
        res_list = await list_collections(settings=settings)
        print_mcp_output("list_collections", res_list)
    except Exception as e:
        print(f"\n[错误] list_collections 执行失败: {e}")

    # 2. query_knowledge_hub
    try:
        print("\n[2/3] 正在测试 query_knowledge_hub (混合检索)...")
        res_query = await query_knowledge_hub(
            query="什么是 RAG", 
            top_k=2, 
            settings=settings
        )
        print_mcp_output("query_knowledge_hub", res_query)
    except Exception as e:
        print(f"\n[错误] query_knowledge_hub 执行失败: {e}")

    # 3. get_document_summary
    try:
        print("\n[3/3] 正在测试 get_document_summary...")
        # 尝试找一个存在的 ID 或者测试错误处理
        res_summary = await get_document_summary(
            doc_id="test_doc_001", 
            settings=settings
        )
        print_mcp_output("get_document_summary", res_summary)
    except Exception as e:
        print(f"\n[错误] get_document_summary 执行失败: {e}")

    print("\n审查完成。")

if __name__ == "__main__":
    asyncio.run(run_inspector())
