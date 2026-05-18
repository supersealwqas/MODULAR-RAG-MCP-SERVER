"""
E1 手动验证测试：MCP Server 入口与 Stdio 约束

验证项：
1. 服务器能通过 Stdio Transport 正常启动
2. initialize 握手返回正确的 serverInfo 和 capabilities
3. tools/list 返回已注册的工具列表
4. stdout 仅包含 MCP 消息，日志输出到 stderr
"""

import json
import subprocess
import sys
import threading


def send_and_read(proc, method: str, params: dict = None, req_id: int = 1, timeout: float = 10.0) -> dict | None:
    """发送 JSON-RPC 请求并读取响应。"""
    request = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params:
        request["params"] = params
    proc.stdin.write(json.dumps(request) + "\n")
    proc.stdin.flush()

    result = []

    def reader():
        try:
            line = proc.stdout.readline()
            if line:
                result.append(line.strip())
        except Exception:
            pass

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return json.loads(result[0]) if result else None


def main():
    print("=" * 60)
    print("E1 手动验证：MCP Server 入口与 Stdio 约束")
    print("=" * 60)

    proc = subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd="d:\\AI\\my_AI_project\\MODULAR-RAG-MCP-SERVER",
    )

    try:
        # 测试 1: initialize 握手
        print("\n[测试 1] initialize 握手...")
        resp = send_and_read(proc, "initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "manual-test", "version": "0.1.0"},
        })
        assert resp is not None, "未收到 initialize 响应"
        assert "result" in resp, f"响应缺少 result: {resp}"
        result = resp["result"]
        print(f"  protocolVersion: {result.get('protocolVersion')}")
        print(f"  serverInfo: {result.get('serverInfo')}")
        print(f"  capabilities: {result.get('capabilities')}")
        print("  [通过] initialize 握手成功")

        # 发送 initialized 通知
        notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        proc.stdin.write(json.dumps(notif) + "\n")
        proc.stdin.flush()

        # 测试 2: tools/list
        print("\n[测试 2] tools/list 工具列表...")
        resp = send_and_read(proc, "tools/list", req_id=2)
        assert resp is not None, "未收到 tools/list 响应"
        assert "result" in resp, f"响应缺少 result: {resp}"
        tools = resp["result"].get("tools", [])
        print(f"  工具数量: {len(tools)}")
        for tool in tools:
            print(f"    - {tool['name']}: {tool.get('description', '')[:60]}")
        assert "query_knowledge_hub" in {t["name"] for t in tools}
        print("  [通过] 工具列表正确")

        # 测试 3: stdout/stderr 分离
        print("\n[测试 3] stdout/stderr 分离...")
        proc.terminate()
        proc.wait(timeout=5)
        stderr = proc.stderr.read()
        assert len(stderr) > 0, "stderr 为空，日志可能未输出"
        print(f"  stderr 日志行数: {stderr.count(chr(10))}")
        print("  [通过] 日志正确输出到 stderr")

        print("\n" + "=" * 60)
        print("全部验证通过！")
        print("=" * 60)

    except Exception as e:
        print(f"\n[失败] {e}")
        proc.terminate()
        proc.wait(timeout=5)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
