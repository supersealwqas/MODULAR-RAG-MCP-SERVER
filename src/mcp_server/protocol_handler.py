"""
MCP 协议处理器

封装 JSON-RPC 2.0 协议解析，处理 initialize、tools/list、tools/call 三类核心方法。
错误码遵循 JSON-RPC 2.0 规范：
  -32600  Invalid Request
  -32601  Method not found
  -32602  Invalid params
  -32603  Internal error
"""

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# JSON-RPC 2.0 标准错误码
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class JSONRPCError(Exception):
    """JSON-RPC 2.0 协议错误。"""

    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(message)

    def to_dict(self) -> dict:
        """转换为 JSON-RPC error 对象。"""
        err = {"code": self.code, "message": self.message}
        if self.data is not None:
            err["data"] = self.data
        return err


class ProtocolHandler:
    """MCP 协议处理器，封装 JSON-RPC 2.0 核心方法。"""

    def __init__(self, server: FastMCP):
        """初始化协议处理器。

        Args:
            server: FastMCP 服务器实例，提供工具注册与执行能力。
        """
        self._server = server

    def handle_initialize(self, params: dict) -> dict:
        """处理 initialize 请求，返回 server capabilities。

        Args:
            params: 客户端发送的 initialize 参数（protocolVersion, capabilities, clientInfo）。

        Returns:
            包含 protocolVersion、capabilities、serverInfo 的响应字典。

        Raises:
            JSONRPCError: 参数缺失或格式错误时抛出 -32602。
        """
        if not isinstance(params, dict):
            raise JSONRPCError(INVALID_PARAMS, "params 必须是字典类型")

        client_info = params.get("clientInfo", {})
        logger.info(
            "initialize 握手: client=%s/%s, protocol=%s",
            client_info.get("name", "unknown"),
            client_info.get("version", "?"),
            params.get("protocolVersion", "?"),
        )

        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {"listChanged": False},
            },
            "serverInfo": {
                "name": self._server.name or "modular-rag-mcp-server",
                "version": "0.1.0",
            },
        }

    def handle_tools_list(self) -> dict:
        """处理 tools/list 请求，返回已注册工具的 schema。

        Returns:
            包含 tools 列表的字典，每个工具含 name、description、inputSchema。
        """
        # 注意：使用 _tool_manager 私有属性，若 FastMCP 版本升级导致报错，
        # 请替换为 self._server.list_tools() 等公开 API
        tools = self._server._tool_manager.list_tools()
        tool_schemas = []
        for tool in tools:
            # 兼容官方 SDK 的 inputSchema 与 FastMCP 的 parameters 两种属性名
            schema = (
                getattr(tool, "inputSchema", None)
                or getattr(tool, "parameters", None)
                or {"type": "object", "properties": {}}
            )
            tool_schemas.append({
                "name": tool.name,
                "description": tool.description or "",
                "inputSchema": schema,
            })
        logger.info("tools/list 返回 %d 个工具", len(tool_schemas))
        return {"tools": tool_schemas}

    async def handle_tools_call(self, name: str, arguments: dict | None = None) -> dict:
        """处理 tools/call 请求，路由到具体 tool 执行。

        注意：这是异步方法，必须被 await 调用。

        Args:
            name: 工具名称。
            arguments: 工具参数字典，可为空。

        Returns:
            包含 content 列表的 MCP 响应字典。

        Raises:
            JSONRPCError:
                - -32602: name 非字符串或 arguments 非字典。
                - -32601: 工具不存在。
                - -32603: 工具执行内部异常。
        """
        if not isinstance(name, str):
            raise JSONRPCError(INVALID_PARAMS, "tool name 必须是字符串")

        if arguments is not None and not isinstance(arguments, dict):
            raise JSONRPCError(INVALID_PARAMS, "tool arguments 必须是字典或 null")

        logger.info("tools/call: name=%s, args=%s", name, arguments)

        try:
            result = await self._server._tool_manager.call_tool(name, arguments or {})
        except Exception as e:
            error_msg = str(e).lower()
            if "unknown tool" in error_msg or "not found" in error_msg:
                raise JSONRPCError(METHOD_NOT_FOUND, f"未知工具: {name}")
            logger.error("工具 %s 执行失败: %s", name, e, exc_info=True)
            raise JSONRPCError(INTERNAL_ERROR, f"工具执行异常: {e}")

        # 提取 MCP 协议的 isError 标志（业务级错误，非系统异常）
        is_error = getattr(result, "isError", False)

        # 获取原始 content 列表
        if isinstance(result, list):
            raw_content = result
        elif hasattr(result, "content"):
            raw_content = result.content
        else:
            text = str(result) if result is not None else ""
            raw_content = [{"type": "text", "text": text}]

        # Pydantic 序列化防御：将 Pydantic 模型对象转为原生字典，
        # 防止外层 json.dumps() 触发 TypeError: Object of type TextContent is not JSON serializable
        serialized_content = []
        for item in raw_content:
            if hasattr(item, "model_dump"):  # Pydantic v2
                serialized_content.append(item.model_dump(exclude_none=True))
            elif hasattr(item, "dict"):  # Pydantic v1
                serialized_content.append(item.dict(exclude_none=True))
            else:
                serialized_content.append(item)

        return {"content": serialized_content, "isError": is_error}
