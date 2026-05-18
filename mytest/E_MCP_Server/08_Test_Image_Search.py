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
from src.core.settings import load_settings

async def test_image_retrieval():
    print(f"\n{'=' * 80}")
    print(f"  实测图片检索功能: 搜文字出图")
    print(f"{'=' * 80}")

    settings = load_settings()
    
    # 针对性查询：我们知道这个词关联了图片
    query = "计算困惑度"
    print(f"\n[1] 执行查询: '{query}'")
    
    try:
        # 调用工具函数
        # 注意：我们之前已经把 query_knowledge_hub 里的 include_images 设为 True 了
        response = await query_knowledge_hub(
            query=query, 
            top_k=1, 
            settings=settings
        )
        
        # 检查响应内容
        content = response.get("content", [])
        print(f"\n[响应分析]")
        print(f"Content 列表长度: {len(content)}")
        
        has_image = False
        for i, item in enumerate(content):
            item_type = item.get("type")
            print(f"  Item {i}: 类型 = {item_type}")
            
            if item_type == "image":
                has_image = True
                print(f"    MIME: {item.get('mimeType')}")
                print(f"    Data (Base64) 长度: {len(item.get('data', ''))} 字符")
                # 预览前 50 位数据
                print(f"    Data 预览: {item.get('data', '')[:50]}...")
            
            if item_type == "text":
                print(f"    文本长度: {len(item.get('text', ''))} 字符")

        if has_image:
            print("\n[🎉 成功] 系统已成功返回关联图片数据！")
        else:
            print("\n[❌ 失败] 未能在响应中找到图片。请检查 Ingestion 阶段是否正确关联了图片路径。")

    except Exception as e:
        print(f"\n[错误] 执行过程发生异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_image_retrieval())
