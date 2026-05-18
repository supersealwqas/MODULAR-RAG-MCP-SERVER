"""
MCP Server 综合测试

整合所有 E_MCP_Server 测试项，验证：
1. 协议层：initialize 握手 + tools/list
2. 工具层：query_knowledge_hub / list_collections / get_document_summary 进程内实际调用
3. 多模态：MultimodalAssembler 图片组装
4. MCP SDK：通过 MCP SDK 客户端连接 Server 并调用工具

用法：
    uv run python mytest/E_MCP_Server/test_all.py           # 全部测试
    uv run python mytest/E_MCP_Server/test_all.py protocol  # 仅协议层
    uv run python mytest/E_MCP_Server/test_all.py tools     # 仅工具层
    uv run python mytest/E_MCP_Server/test_all.py mcp       # 仅 MCP SDK 客户端
"""

import asyncio
import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import time

# 强制 stdout 使用 UTF-8，避免 Windows GBK 编码错误
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ============================================================
# 工具函数
# ============================================================

def section(title: str):
    """打印分节标题。"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_item(name: str):
    """打印测试项名称。"""
    print(f"\n--- {name} ---")


def send_jsonrpc(proc, request: dict, timeout: float = 30.0) -> dict | None:
    """发送 JSON-RPC 请求并读取响应（原始 stdin/stdout 方式）。"""
    proc.stdin.write(json.dumps(request) + "\n")
    proc.stdin.flush()
    start = time.time()
    while time.time() - start < timeout:
        line = proc.stdout.readline()
        if line:
            return json.loads(line.strip())
        time.sleep(0.1)
    return None


# ============================================================
# 1. 协议层测试
# ============================================================

async def test_protocol():
    """测试 MCP 协议层：initialize + tools/list。"""
    section("1. 协议层测试")

    # 1a. 原始 JSON-RPC 方式
    test_item("1a. 原始 JSON-RPC 通信")
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
        cwd="d:\\AI\\my_AI_project\\MODULAR-RAG-MCP-SERVER",
    )

    try:
        # initialize
        resp = send_jsonrpc(proc, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-all", "version": "0.1.0"},
            },
        })
        assert resp is not None, "未收到 initialize 响应"
        info = resp["result"]
        print(f"  server: {info['serverInfo']['name']}")
        print(f"  protocol: {info['protocolVersion']}")
        print(f"  capabilities: {list(info['capabilities'].keys())}")

        # initialized 通知
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        proc.stdin.flush()

        # tools/list
        resp = send_jsonrpc(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert resp is not None, "未收到 tools/list 响应"
        tools = resp["result"]["tools"]
        print(f"  工具数量: {len(tools)}")
        for t in tools:
            print(f"    - {t['name']}: {t.get('description', '')[:50]}")
        print("  [通过]")

        # 1b. MCP SDK 方式
        test_item("1b. MCP SDK 客户端连接")
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "src.mcp_server.server"],
            cwd="d:\\AI\\my_AI_project\\MODULAR-RAG-MCP-SERVER",
        )
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                sdk_tools = await session.list_tools()
                print(f"  SDK 工具数量: {len(sdk_tools.tools)}")
                for t in sdk_tools.tools:
                    print(f"    - {t.name}")
                print("  [通过]")

    finally:
        proc.terminate()
        proc.wait(timeout=5)


# ============================================================
# 2. 工具函数测试（进程内调用）
# ============================================================

async def test_tools():
    """测试 MCP 工具函数 + 直接检索原始内容。"""
    section("2. 工具函数测试（进程内实际调用）")

    from src.core.query_engine.hybrid_search import HybridSearch
    from src.core.query_engine.reranker import Reranker
    from src.core.settings import load_settings
    from src.core.trace.trace_context import TraceContext
    from src.mcp_server.tools.list_collections import list_collections
    from src.mcp_server.tools.get_document_summary import get_document_summary

    # 2a. list_collections
    test_item("2a. list_collections — 列出所有集合")
    start = time.time()
    result = await list_collections()
    elapsed = time.time() - start
    print(f"  耗时: {elapsed:.2f}s")
    print(f"  isError: {result['isError']}")
    sc = result.get("structuredContent", {})
    print(f"  集合数量: {sc.get('total_count', 0)}")
    for coll in sc.get("collections", []):
        print(f"    - {coll['name']}: {coll['document_count']} 条文档")
    print(f"  [通过]")

    # 2b. 直接调用 HybridSearch + Reranker，打印原始检索内容
    test_item("2b. HybridSearch + Reranker — 原始检索结果")
    query = "什么是大语言模型"
    top_k = 5
    print(f"  查询: '{query}'")
    print(f"  top_k: {top_k}")

    settings = load_settings()
    trace = TraceContext()

    # 粗排
    hybrid_search = HybridSearch(settings)
    start = time.time()
    results = hybrid_search.search(query=query, top_k=top_k, trace=trace)
    coarse_elapsed = time.time() - start
    print(f"\n  [粗排] HybridSearch 耗时: {coarse_elapsed:.2f}s，返回 {len(results)} 条")
    for i, r in enumerate(results):
        print(f"    [{i+1}] score={r.score:.4f}  chunk_id={r.chunk_id}")
        print(f"        来源: {r.metadata.get('source_path', 'N/A')}")
        print(f"        文本: {r.text[:120]}...")

    # 精排
    reranker = Reranker(settings)
    start = time.time()
    rerank_result = reranker.rerank(query=query, candidates=results, top_k=top_k, trace=trace)
    results = rerank_result["results"]
    fine_elapsed = time.time() - start
    print(f"\n  [精排] Reranker 耗时: {fine_elapsed:.2f}s，返回 {len(results)} 条")
    for i, r in enumerate(results):
        print(f"    [{i+1}] score={r.score:.4f}  chunk_id={r.chunk_id}")
        print(f"        来源: {r.metadata.get('source_path', 'N/A')}")

    # 打印每条检索块的完整文本
    print(f"\n  {'=' * 56}")
    print(f"  检索结果完整文本")
    print(f"  {'=' * 56}")
    for i, r in enumerate(results):
        print(f"\n  [{i+1}] score={r.score:.4f}")
        print(f"      chunk_id: {r.chunk_id}")
        print(f"      来源: {r.metadata.get('source_path', 'N/A')}")
        print(f"      metadata: {r.metadata}")
        print(f"      --- 文本 ---")
        print(r.text)
        print(f"      --- 文本结束 ---")
    print(f"  [通过]")

    # 2c. get_document_summary
    test_item("2c. get_document_summary — 获取文档摘要")
    doc_id = "test_doc_001"
    print(f"  doc_id: '{doc_id}'")
    start = time.time()
    result = await get_document_summary(doc_id=doc_id)
    elapsed = time.time() - start
    print(f"  耗时: {elapsed:.2f}s")
    print(f"  isError: {result['isError']}")
    sc = result.get("structuredContent", {})
    print(f"  found: {sc.get('found', False)}")
    if sc.get("found"):
        print(f"  title: {sc.get('title')}")
        print(f"  summary: {sc.get('summary')}")
    print(f"  [通过]")

    # 2d. 空 doc_id 错误处理
    test_item("2d. get_document_summary — 空 doc_id 错误处理")
    result = await get_document_summary(doc_id="")
    print(f"  isError: {result['isError']}")
    print(f"  内容: {result['content'][0]['text']}")
    assert result["isError"] is True
    print(f"  [通过]")


# ============================================================
# 3. 多模态组装测试
# ============================================================

def test_multimodal():
    """测试 MultimodalAssembler 图片组装。"""
    section("3. 多模态组装测试")

    from src.core.response.multimodal_assembler import (
        assemble_image_content,
        assemble_multimodal_response,
        _get_mime_type,
    )
    from src.core.types import RetrievalResult

    # 创建测试 PNG
    png_data = (
        b'\x89PNG\r\n\x1a\n'
        b'\x00\x00\x00\rIHDR'
        b'\x00\x00\x00\x01'
        b'\x00\x00\x00\x01'
        b'\x08\x02'
        b'\x00\x00\x00'
        b'\x90wS\xde'
        b'\x00\x00\x00\x0cIDAT'
        b'x\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05'
        b'\x18\xd8N\x00\x00\x00\x00IEND'
        b'\xaeB`\x82'
    )
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(png_data)
        image_path = f.name

    try:
        # 3a. MIME 类型检测
        test_item("3a. MIME 类型检测")
        for ext, expected in [("test.png", "image/png"), ("test.jpg", "image/jpeg"), ("test.gif", "image/gif")]:
            result = _get_mime_type(ext)
            print(f"  {ext} -> {result}")
            assert result == expected
        print(f"  [通过]")

        # 3b. 单图片组装
        test_item("3b. 单图片组装（base64 编码）")
        result = assemble_image_content(image_path)
        print(f"  type: {result['type']}")
        print(f"  mimeType: {result['mimeType']}")
        print(f"  data 长度: {len(result['data'])} 字符")
        decoded = base64.b64decode(result["data"])
        print(f"  解码后大小: {len(decoded)} 字节")
        assert result["type"] == "image"
        print(f"  [通过]")

        # 3c. 多模态响应（无图片）
        test_item("3c. 多模态响应组装（无图片）")
        results = [RetrievalResult(chunk_id="c1", score=0.9, text="测试文本", metadata={})]
        content = assemble_multimodal_response(results, "## Markdown 内容")
        print(f"  content 长度: {len(content)}")
        print(f"  类型: {[c['type'] for c in content]}")
        assert len(content) == 1 and content[0]["type"] == "text"
        print(f"  [通过]")

        # 3d. 多模态响应（有图片）
        test_item("3d. 多模态响应组装（有图片）")
        results = [RetrievalResult(
            chunk_id="c1", score=0.9, text="测试文本",
            metadata={"images": [{"id": "img_001", "path": image_path}]},
        )]
        content = assemble_multimodal_response(results, "## Markdown 内容")
        print(f"  content 长度: {len(content)}")
        print(f"  类型: {[c['type'] for c in content]}")
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image"
        print(f"  [通过]")

        # 3e. JSON 序列化
        test_item("3e. JSON 序列化")
        response = {"content": content, "structuredContent": {"result_count": 1}, "isError": False}
        serialized = json.dumps(response, ensure_ascii=False)
        print(f"  序列化后长度: {len(serialized)} 字符")
        print(f"  [通过]")

    finally:
        if os.path.exists(image_path):
            os.unlink(image_path)


# ============================================================
# 4. MCP SDK 工具调用测试（进程内通过 server 实例）
# ============================================================

async def test_mcp_sdk_inprocess():
    """通过 server 实例的 tool_manager 直接调用工具。"""
    section("4. MCP Server 工具调用（进程内）")

    from src.mcp_server.server import server

    tools = server._tool_manager._tools
    print(f"  注册工具: {list(tools.keys())}")

    # 4a. list_collections
    test_item("4a. list_collections（通过 server tool_manager）")
    start = time.time()
    result = await tools['list_collections'].fn()
    elapsed = time.time() - start
    print(f"  耗时: {elapsed:.2f}s")
    print(f"  结果: {result[:300]}")
    print(f"  [通过]")

    # 4b. get_document_summary
    test_item("4b. get_document_summary（通过 server tool_manager）")
    start = time.time()
    result = await tools['get_document_summary'].fn(doc_id="test_doc")
    elapsed = time.time() - start
    print(f"  耗时: {elapsed:.2f}s")
    print(f"  结果: {result[:300]}")
    print(f"  [通过]")


# ============================================================
# 主入口
# ============================================================

async def main():
    section("MCP Server 综合测试")

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    try:
        if mode in ("all", "protocol"):
            await test_protocol()

        if mode in ("all", "tools"):
            await test_tools()

        if mode in ("all", "multimodal"):
            test_multimodal()

        if mode in ("all", "mcp"):
            await test_mcp_sdk_inprocess()

        section("全部测试通过！")
        return 0

    except Exception as e:
        print(f"\n[失败] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
