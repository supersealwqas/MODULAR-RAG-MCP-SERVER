import asyncio
import json
import os
import subprocess
import sys
import time
import io

# 强制使用 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def section(title: str):
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}", flush=True)

def send_jsonrpc(proc, request: dict, timeout: float = 30.0) -> dict | None:
    """发送请求并读取响应，忽略通知。"""
    req_json = json.dumps(request)
    # print(f"\n[发送] {req_json}", flush=True)
    proc.stdin.write(req_json + "\n")
    proc.stdin.flush()
    
    start = time.time()
    while time.time() - start < timeout:
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.1)
            continue
        
        line = line.strip()
        if not line:
            continue
            
        try:
            resp = json.loads(line)
            # 如果是响应（带有 id），则返回
            if "id" in resp and resp["id"] == request.get("id"):
                return resp
            # 如果是日志/通知，跳过或记录
        except json.JSONDecodeError:
            continue
            
    return None

async def main():
    section("MCP Server 完整输出内容校验 (JSON-RPC 协议层)")

    # 1. 启动服务
    print("[步骤 1] 启动 MCP Server 子进程...", flush=True)
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, # 忽略日志输出到控制台，防止干扰
        text=True, encoding="utf-8", errors="replace",
        cwd=os.getcwd(),
    )

    try:
        # 2. Initialize
        print("\n[步骤 2] 协议握手 (initialize)...", flush=True)
        init_resp = send_jsonrpc(proc, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "validator", "version": "1.0.0"},
            },
        })
        print(f"原始响应内容:\n{json.dumps(init_resp, indent=2, ensure_ascii=False)}", flush=True)

        # 3. Initialized
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        proc.stdin.flush()

        # 4. list_collections
        print("\n[步骤 3] 测试 list_collections 输出...", flush=True)
        list_coll_resp = send_jsonrpc(proc, {
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "list_collections", "arguments": {}}
        })
        print(f"原始响应内容:\n{json.dumps(list_coll_resp, indent=2, ensure_ascii=False)}", flush=True)

        # 5. query_knowledge_hub
        print("\n[步骤 4] 测试 query_knowledge_hub 输出 (耗时较长)...", flush=True)
        query_resp = send_jsonrpc(proc, {
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {
                "name": "query_knowledge_hub",
                "arguments": {"query": "什么是 RAG", "top_k": 2}
            }
        }, timeout=60.0) # 给 60 秒超时
        print(f"原始响应内容:\n{json.dumps(query_resp, indent=2, ensure_ascii=False)}", flush=True)

        # 6. get_document_summary
        print("\n[步骤 5] 测试 get_document_summary 输出...", flush=True)
        summary_resp = send_jsonrpc(proc, {
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "get_document_summary", "arguments": {"doc_id": "test_doc_001"}}
        })
        print(f"原始响应内容:\n{json.dumps(summary_resp, indent=2, ensure_ascii=False)}", flush=True)

        print("\n" + "="*70)
        print("  所有 MCP 功能原始输出内容已捕获并显示完成")
        print("="*70, flush=True)

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except:
            proc.kill()

if __name__ == "__main__":
    asyncio.run(main())
