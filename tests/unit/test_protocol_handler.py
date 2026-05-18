"""
ProtocolHandler 单元测试

覆盖：initialize 握手、tools/list 工具列表、tools/call 路由执行、
     JSON-RPC 错误码（-32601/-32602/-32603）。
"""

import pytest
from mcp.server.fastmcp import FastMCP

from src.mcp_server.protocol_handler import (
    INVALID_PARAMS,
    INTERNAL_ERROR,
    METHOD_NOT_FOUND,
    JSONRPCError,
    ProtocolHandler,
)


@pytest.fixture
def server():
    """创建带测试工具的 FastMCP 实例。"""
    s = FastMCP(name="test-server")

    @s.tool()
    async def echo(text: str) -> str:
        """返回输入文本。"""
        return text

    @s.tool()
    async def add(a: int, b: int) -> int:
        """两数相加。"""
        return a + b

    @s.tool()
    async def fail_tool() -> str:
        """总是抛出异常的工具。"""
        raise RuntimeError("模拟内部错误")

    return s


@pytest.fixture
def handler(server):
    """创建 ProtocolHandler 实例。"""
    return ProtocolHandler(server)


# ==================== initialize 测试 ====================


class TestHandleInitialize:
    """initialize 握手测试。"""

    def test_returns_server_info(self, handler):
        """应返回正确的 serverInfo。"""
        result = handler.handle_initialize({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0.1.0"},
        })
        assert result["serverInfo"]["name"] == "test-server"
        assert result["serverInfo"]["version"] == "0.1.0"

    def test_returns_protocol_version(self, handler):
        """应返回支持的 protocolVersion。"""
        result = handler.handle_initialize({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0.1.0"},
        })
        assert result["protocolVersion"] == "2024-11-05"

    def test_returns_capabilities_with_tools(self, handler):
        """应声明 tools 能力。"""
        result = handler.handle_initialize({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0.1.0"},
        })
        assert "tools" in result["capabilities"]
        assert result["capabilities"]["tools"]["listChanged"] is False

    def test_accepts_empty_params(self, handler):
        """空参数也应返回有效响应。"""
        result = handler.handle_initialize({})
        assert "serverInfo" in result
        assert "capabilities" in result

    def test_rejects_non_dict_params(self, handler):
        """非字典参数应抛出 -32602。"""
        with pytest.raises(JSONRPCError) as exc_info:
            handler.handle_initialize("invalid")
        assert exc_info.value.code == INVALID_PARAMS


# ==================== tools/list 测试 ====================


class TestHandleToolsList:
    """tools/list 测试。"""

    def test_returns_registered_tools(self, handler):
        """应返回已注册的工具列表。"""
        result = handler.handle_tools_list()
        assert "tools" in result
        tools = result["tools"]
        assert len(tools) == 3

    def test_tool_has_name(self, handler):
        """每个工具应包含 name 字段。"""
        result = handler.handle_tools_list()
        names = {t["name"] for t in result["tools"]}
        assert "echo" in names
        assert "add" in names
        assert "fail_tool" in names

    def test_tool_has_description(self, handler):
        """每个工具应包含 description 字段。"""
        result = handler.handle_tools_list()
        for tool in result["tools"]:
            assert "description" in tool
            assert isinstance(tool["description"], str)

    def test_tool_has_input_schema(self, handler):
        """每个工具应包含 inputSchema 字段。"""
        result = handler.handle_tools_list()
        for tool in result["tools"]:
            assert "inputSchema" in tool
            schema = tool["inputSchema"]
            assert "type" in schema

    def test_echo_tool_schema(self, handler):
        """echo 工具的 schema 应包含 text 参数。"""
        result = handler.handle_tools_list()
        echo = next(t for t in result["tools"] if t["name"] == "echo")
        props = echo["inputSchema"].get("properties", {})
        assert "text" in props


# ==================== tools/call 测试（异步） ====================


class TestHandleToolsCall:
    """tools/call 测试（handle_tools_call 是异步方法）。"""

    @pytest.mark.anyio
    async def test_call_echo(self, handler):
        """调用 echo 工具应返回输入文本。"""
        result = await handler.handle_tools_call("echo", {"text": "你好"})
        assert "content" in result
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert "你好" in result["content"][0]["text"]

    @pytest.mark.anyio
    async def test_call_add(self, handler):
        """调用 add 工具应返回计算结果。"""
        result = await handler.handle_tools_call("add", {"a": 3, "b": 5})
        text = result["content"][0]["text"]
        assert "8" in text

    @pytest.mark.anyio
    async def test_call_with_empty_args(self, handler):
        """空参数调用应正常处理。"""
        with pytest.raises(JSONRPCError) as exc_info:
            await handler.handle_tools_call("fail_tool", {})
        assert exc_info.value.code == INTERNAL_ERROR

    @pytest.mark.anyio
    async def test_unknown_tool_returns_32601(self, handler):
        """未知工具应返回 -32601 Method not found。"""
        with pytest.raises(JSONRPCError) as exc_info:
            await handler.handle_tools_call("nonexistent_tool", {})
        assert exc_info.value.code == METHOD_NOT_FOUND
        assert "未知工具" in exc_info.value.message

    @pytest.mark.anyio
    async def test_non_string_name_returns_32602(self, handler):
        """非字符串 name 应返回 -32602。"""
        with pytest.raises(JSONRPCError) as exc_info:
            await handler.handle_tools_call(123, {})
        assert exc_info.value.code == INVALID_PARAMS

    @pytest.mark.anyio
    async def test_non_dict_arguments_returns_32602(self, handler):
        """非字典 arguments 应返回 -32602。"""
        with pytest.raises(JSONRPCError) as exc_info:
            await handler.handle_tools_call("echo", "invalid")
        assert exc_info.value.code == INVALID_PARAMS

    @pytest.mark.anyio
    async def test_tool_execution_error_returns_32603(self, handler):
        """工具执行异常应返回 -32603 Internal error。"""
        with pytest.raises(JSONRPCError) as exc_info:
            await handler.handle_tools_call("fail_tool")
        assert exc_info.value.code == INTERNAL_ERROR
        # 不应泄露堆栈
        assert "Traceback" not in str(exc_info.value.data or "")

    @pytest.mark.anyio
    async def test_returns_list_content(self, handler):
        """工具返回列表时应直接作为 content。"""

        @handler._server.tool()
        async def list_tool() -> list:
            """返回列表的工具。"""
            return [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]

        result = await handler.handle_tools_call("list_tool")
        assert result["content"] == [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]

    @pytest.mark.anyio
    async def test_returns_none_as_empty(self, handler):
        """工具返回 None 时应返回空文本。"""

        @handler._server.tool()
        async def none_tool() -> None:
            """返回 None 的工具。"""
            return None

        result = await handler.handle_tools_call("none_tool")
        assert result["content"][0]["text"] == ""

    @pytest.mark.anyio
    async def test_is_error_flag_false_on_success(self, handler):
        """成功调用时 isError 应为 False。"""
        result = await handler.handle_tools_call("echo", {"text": "ok"})
        assert result.get("isError") is False

    @pytest.mark.anyio
    async def test_content_is_native_types(self, handler):
        """返回的 content 应全部是原生 Python 类型（非 Pydantic 对象），可被 json.dumps 序列化。"""
        import json

        result = await handler.handle_tools_call("echo", {"text": "test"})
        # 确保整个响应可被 json.dumps 序列化，不触发 TypeError
        serialized = json.dumps(result, ensure_ascii=False)
        assert "test" in serialized

    @pytest.mark.anyio
    async def test_list_content_serializable(self, handler):
        """列表类型的 content 也应可被 json.dumps 序列化。"""
        import json

        @handler._server.tool()
        async def multi_tool() -> list:
            """返回多条内容。"""
            return [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]

        result = await handler.handle_tools_call("multi_tool")
        serialized = json.dumps(result, ensure_ascii=False)
        assert '"a"' in serialized
        assert '"b"' in serialized


# ==================== JSONRPCError 测试 ====================


class TestJSONRPCError:
    """JSONRPCError 异常测试。"""

    def test_to_dict(self):
        """to_dict 应返回标准 JSON-RPC error 结构。"""
        err = JSONRPCError(-32601, "Method not found")
        d = err.to_dict()
        assert d["code"] == -32601
        assert d["message"] == "Method not found"
        assert "data" not in d

    def test_to_dict_with_data(self):
        """带 data 的 to_dict 应包含 data 字段。"""
        err = JSONRPCError(-32603, "Internal error", data="detail")
        d = err.to_dict()
        assert d["data"] == "detail"
