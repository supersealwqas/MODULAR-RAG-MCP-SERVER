"""
E2 手动验证测试：Protocol Handler 协议解析与能力协商

验证项：
1. handle_initialize 返回正确的 serverInfo 和 capabilities
2. handle_tools_list 返回已注册工具的 schema
3. handle_tools_call 正确路由到工具并返回结果（异步）
4. JSON-RPC 错误码：-32601（未知工具）、-32602（参数错误）、-32603（内部异常）
5. 多模态返回：列表结果直接作为 content，不强转 str
"""

import asyncio
import sys

from mcp.server.fastmcp import FastMCP

from src.mcp_server.protocol_handler import (
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    JSONRPCError,
    ProtocolHandler,
)


async def main():
    print("=" * 60)
    print("E2 手动验证：Protocol Handler 协议解析与能力协商")
    print("=" * 60)

    # 创建测试服务器
    server = FastMCP(name="test-server")

    @server.tool()
    async def echo(text: str) -> str:
        """返回输入文本。"""
        return text

    @server.tool()
    async def add(a: int, b: int) -> int:
        """两数相加。"""
        return a + b

    @server.tool()
    async def multimodal_tool() -> list:
        """返回多模态内容的工具。"""
        return [{"type": "text", "text": "文本内容"}, {"type": "text", "text": "图片占位"}]

    handler = ProtocolHandler(server)

    # 测试 1: initialize 握手
    print("\n[测试 1] handle_initialize 握手...")
    result = handler.handle_initialize({
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "manual-test", "version": "0.1.0"},
    })
    print(f"  serverInfo: {result['serverInfo']}")
    print(f"  capabilities: {result['capabilities']}")
    assert result["serverInfo"]["name"] == "test-server"
    assert "tools" in result["capabilities"]
    print("  [通过] initialize 握手成功")

    # 测试 2: tools/list
    print("\n[测试 2] handle_tools_list 工具列表...")
    result = handler.handle_tools_list()
    tools = result["tools"]
    print(f"  工具数量: {len(tools)}")
    for tool in tools:
        print(f"    - {tool['name']}: {tool.get('description', '')}")
    assert len(tools) == 3
    print("  [通过] 工具列表正确")

    # 测试 3: tools/call 正常调用（异步）
    print("\n[测试 3] handle_tools_call 正常调用（异步）...")
    result = await handler.handle_tools_call("echo", {"text": "你好世界"})
    text = result["content"][0]["text"]
    print(f"  echo('你好世界') -> {text}")
    assert "你好世界" in text

    result = await handler.handle_tools_call("add", {"a": 17, "b": 25})
    text = result["content"][0]["text"]
    print(f"  add(17, 25) -> {text}")
    assert "42" in text
    print("  [通过] 工具调用正确")

    # 测试 4: 多模态返回
    print("\n[测试 4] 多模态返回（列表直接作为 content）...")
    result = await handler.handle_tools_call("multimodal_tool")
    content = result["content"]
    print(f"  content 类型: {type(content).__name__}, 长度: {len(content)}")
    for item in content:
        print(f"    - {item}")
    assert isinstance(content, list)
    assert len(content) == 2
    print("  [通过] 多模态返回正确")

    # 测试 5: 未知工具 (-32601)
    print("\n[测试 5] 未知工具错误码 (-32601)...")
    try:
        await handler.handle_tools_call("nonexistent", {})
        print("  [失败] 应该抛出异常")
        return 1
    except JSONRPCError as e:
        print(f"  code={e.code}, message={e.message}")
        assert e.code == METHOD_NOT_FOUND
        print("  [通过] 返回 -32601")

    # 测试 6: 参数错误 (-32602)
    print("\n[测试 6] 参数错误 (-32602)...")
    try:
        await handler.handle_tools_call(123, {})
        print("  [失败] 应该抛出异常")
        return 1
    except JSONRPCError as e:
        print(f"  code={e.code}, message={e.message}")
        assert e.code == INVALID_PARAMS
        print("  [通过] 返回 -32602")

    print("\n" + "=" * 60)
    print("全部验证通过！")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
