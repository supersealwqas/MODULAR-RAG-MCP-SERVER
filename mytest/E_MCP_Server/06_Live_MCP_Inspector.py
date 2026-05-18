import asyncio
import json
import os
import subprocess
import sys
import time
import io

# 强制使用 UTF-8 避免 Windows 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

def print_json(label: str, data: dict):
    """美化打印 JSON。"""
    print(f"\n>>> {label} <<<", flush=True)
    print(json.dumps(data, indent=2, ensure_ascii=False), flush=True)

def send_and_receive(proc, method: str, params: dict = None, request_id: int = 1):
    """发送请求并接收原始响应。"""
    request = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params or {}
    }
    
    print_json(f"发送请求: {method}", request)
    
    proc.stdin.write(json.dumps(request) + "\n")
    proc.stdin.flush()
    
    # 读取响应
    start_time = time.time()
    timeout = 60 # 增加超时时间到 60 秒
    while time.time() - start_time < timeout:
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.1)
            continue
        try:
            line_str = line.strip()
            if not line_str:
                continue
            response = json.loads(line_str)
            # 过滤掉通知消息，只看对应的 id 响应
            if response.get("id") == request_id:
                print_json(f"收到原始响应: {method}", response)
                return response
        except json.JSONDecodeError:
            continue
    print(f"\n[TIMEOUT] 等待 {method} 响应超时", flush=True)
    return None

async def run_inspector():
    print(f"\n{'=' * 80}")
    print(f"  MCP Server 全功能输出审查工具 (Live Inspector)")
    print(f"{'=' * 80}", flush=True)

    # 启动 Server，将 stderr 重定向到系统 stderr，避免缓冲区满导致阻塞
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr, # 直接输出到终端
        text=True, encoding="utf-8", errors="replace",
        cwd=os.getcwd(),
    )

    try:
        # 1. 协议握手
        print("\n--- 步骤 1: 协议初始化 ---")
        send_and_receive(proc, "initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "live-inspector", "version": "1.0.0"}
        }, request_id=101)
        
        # 发送 initialized 通知（无响应）
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        proc.stdin.flush()

        # 2. 获取工具列表
        print("\n--- 步骤 2: 工具能力发现 ---")
        send_and_receive(proc, "tools/list", {}, request_id=102)

        # 3. 调用 list_collections
        print("\n--- 步骤 3: 调用 list_collections 工具 ---")
        send_and_receive(proc, "tools/call", {
            "name": "list_collections",
            "arguments": {}
        }, request_id=103)

        # 4. 调用 query_knowledge_hub
        print("\n--- 步骤 4: 调用 query_knowledge_hub 工具 (混合检索) ---")
        print("提示: 检索可能需要几秒钟...")
        send_and_receive(proc, "tools/call", {
            "name": "query_knowledge_hub",
            "arguments": {
                "query": "什么是 RAG",
                "top_k": 2
            }
        }, request_id=104)

        # 5. 调用 get_document_summary (测试不存在的情况)
        print("\n--- 步骤 5: 调用 get_document_summary 工具 ---")
        send_and_receive(proc, "tools/call", {
            "name": "get_document_summary",
            "arguments": {
                "doc_id": "non_existent_doc"
            }
        }, request_id=105)

        print(f"\n{'=' * 80}")
        print("  所有功能审查完成！")
        print(f"{'=' * 80}")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except:
            proc.kill()

if __name__ == "__main__":
    asyncio.run(run_inspector())
