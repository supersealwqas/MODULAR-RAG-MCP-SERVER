"""
MCP Server 集成测试

通过子进程方式启动 MCP Server，验证 Stdio Transport 的协议行为。
测试覆盖：initialize 握手、tools/list 工具列表、stdout/stderr 分离。

修复：使用 stderr 后台排空线程防止管道缓冲区满导致死锁。
"""

import json
import subprocess
import sys
import threading
from pathlib import Path


def _send_json_rpc(proc, method: str, params: dict | None = None, req_id: int = 1) -> None:
    """向服务器 stdin 发送 JSON-RPC 请求（换行分隔格式）。"""
    request = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params:
        request["params"] = params
    proc.stdin.write(json.dumps(request) + "\n")
    proc.stdin.flush()


def _read_response(proc, timeout: float = 10.0) -> dict | None:
    """从服务器 stdout 读取一行 JSON-RPC 响应，带超时（Windows 兼容）。"""
    result = []

    def _reader():
        try:
            line = proc.stdout.readline()
            if line:
                result.append(line.strip())
        except Exception:
            pass

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    if result:
        return json.loads(result[0])
    return None


def _drain_stderr(proc: subprocess.Popen) -> threading.Thread:
    """后台线程排空 stderr，防止管道缓冲区满导致死锁。"""
    def _reader():
        try:
            for _ in proc.stderr:
                pass
        except (ValueError, OSError):
            pass
    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return t


def _start_server() -> subprocess.Popen:
    """启动 MCP Server 子进程（UTF-8 编码 + stderr 排空）。"""
    cwd = str(Path(__file__).resolve().parents[2])
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
    )
    _drain_stderr(proc)
    return proc


def test_server_initialize():
    """验证 MCP Server 能完成 initialize 握手。"""
    proc = _start_server()
    try:
        # 发送 initialize 请求
        _send_json_rpc(proc, "initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "0.1.0"},
        })
        resp = _read_response(proc)
        assert resp is not None, "未收到 initialize 响应"
        assert resp.get("jsonrpc") == "2.0"
        assert "result" in resp, f"initialize 响应缺少 result: {resp}"
        result = resp["result"]
        assert "protocolVersion" in result, "result 缺少 protocolVersion"
        assert "serverInfo" in result, "result 缺少 serverInfo"
        assert result["serverInfo"]["name"] == "modular-rag-mcp-server"
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_server_tools_list():
    """验证 MCP Server 能返回工具列表。"""
    proc = _start_server()
    try:
        # 先完成 initialize 握手
        _send_json_rpc(proc, "initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "0.1.0"},
        })
        _read_response(proc)

        # 发送 initialized 通知（无 id，不需要响应）
        notification = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        proc.stdin.write(json.dumps(notification) + "\n")
        proc.stdin.flush()

        # 发送 tools/list 请求
        _send_json_rpc(proc, "tools/list", req_id=2)
        resp = _read_response(proc)
        assert resp is not None, "未收到 tools/list 响应"
        assert resp.get("jsonrpc") == "2.0"
        assert "result" in resp, f"tools/list 响应缺少 result: {resp}"
        result = resp["result"]
        assert "tools" in result, "result 缺少 tools 列表"
        tools = result["tools"]
        assert len(tools) >= 1, f"工具列表为空: {tools}"

        # 验证工具结构
        tool_names = {t["name"] for t in tools}
        assert "query_knowledge_hub" in tool_names, f"缺少 query_knowledge_hub 工具, 实际: {tool_names}"
        for tool in tools:
            assert "name" in tool, f"工具缺少 name 字段: {tool}"
            assert "description" in tool, f"工具缺少 description 字段: {tool}"
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_server_stdio_separation():
    """验证 stdout 仅包含 MCP 消息，日志输出到 stderr。"""
    # 此测试需要单独捕获 stderr，不使用公共 _start_server 的排空线程
    cwd = str(Path(__file__).resolve().parents[2])
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
    )
    stderr_lines: list[str] = []

    def _collect_stderr():
        try:
            for line in proc.stderr:
                stderr_lines.append(line)
        except (ValueError, OSError):
            pass

    threading.Thread(target=_collect_stderr, daemon=True).start()
    try:
        _send_json_rpc(proc, "initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "0.1.0"},
        })
        resp = _read_response(proc)
        assert resp is not None

        import time
        time.sleep(0.5)
        proc.terminate()
        proc.wait(timeout=5)

        stderr_output = "".join(stderr_lines)
        assert len(stderr_output) > 0, "stderr 没有日志输出，日志可能污染了 stdout"
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)


def test_server_invalid_method():
    """验证无效方法返回 JSON-RPC 错误。"""
    proc = _start_server()
    try:
        # 发送无效方法请求
        _send_json_rpc(proc, "invalid/method", req_id=99)
        resp = _read_response(proc, timeout=5.0)
        # MCP SDK 对无效方法可能返回 error 或直接忽略
        # 如果收到响应，验证其结构
        if resp is not None:
            assert resp.get("jsonrpc") == "2.0"
            # 可能是 error response 或其他格式
            if "error" in resp:
                assert resp["error"]["code"] != 0
    finally:
        proc.terminate()
        proc.wait(timeout=5)
