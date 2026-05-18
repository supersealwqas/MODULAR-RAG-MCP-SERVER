import asyncio
import json
import os
import subprocess
import sys
import time
import io

# 强制使用 UTF-8 避免 Windows 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def section(title: str):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")

def send_jsonrpc(proc, request: dict, timeout: float = 10.0) -> dict | None:
    proc.stdin.write(json.dumps(request) + "\n")
    proc.stdin.flush()
    start = time.time()
    while time.time() - start < timeout:
        line = proc.stdout.readline()
        if line:
            return json.loads(line.strip())
        time.sleep(0.1)
    return None

async def main():
    section("E1/E2: MCP Server 协议与 Stdio 测试")

    # 启动 Server 进程
    print("[1] 启动 MCP Server 子进程...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
        cwd=os.getcwd(),
    )

    try:
        # 1. Initialize
        print("\n[2] 发送 initialize 请求...")
        init_req = {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "manual-test", "version": "0.1.0"},
            },
        }
        resp = send_jsonrpc(proc, init_req)
        if resp and "result" in resp:
            print(f"  成功响应: {resp['result']['serverInfo']['name']} v{resp['result']['serverInfo']['version']}")
        else:
            print(f"  失败响应: {resp}")
            return

        # 2. Initialized Notification
        print("\n[3] 发送 notifications/initialized...")
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        proc.stdin.flush()

        # 3. Tools List
        print("\n[4] 发送 tools/list 请求...")
        list_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
        resp = send_jsonrpc(proc, list_req)
        if resp and "result" in resp:
            tools = resp["result"]["tools"]
            print(f"  发现 {len(tools)} 个工具:")
            for t in tools:
                print(f"    - {t['name']}: {t.get('description', '')[:50]}...")
        else:
            print(f"  失败响应: {resp}")

        print("\n=== E1/E2 协议测试完成 ===")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except:
            proc.kill()

if __name__ == "__main__":
    asyncio.run(main())
