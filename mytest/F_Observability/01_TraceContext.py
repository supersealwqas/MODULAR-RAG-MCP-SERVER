import sys
import os
import time
import json

# 将 src 目录添加到路径以便导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from core.trace.trace_context import TraceContext

def test_trace_context_manual():
    print("=== 开始测试 TraceContext (F1 阶段能力) ===")
    
    # 1. 初始化
    print("\n[1] 初始化 TraceContext...")
    trace = TraceContext(trace_type="ingestion")
    print(f"Trace ID: {trace.trace_id}")
    print(f"Trace Type: {trace.trace_type}")
    
    # 2. 记录阶段
    print("\n[2] 模拟记录两个阶段 (load 和 split)...")
    
    # 模拟加载阶段
    time.sleep(0.05)  # 模拟一些耗时
    trace.record_stage("load", method="markitdown", elapsed_ms=50.5, details={"file_size": 1024})
    
    # 模拟切分阶段
    time.sleep(0.03)
    trace.record_stage("split", method="recursive", elapsed_ms=30.2, details={"chunks": 10})
    
    # 3. 结束追踪
    print("\n[3] 调用 finish()...")
    trace.finish()
    
    # 4. 耗时统计
    print("\n[4] 耗时统计检查:")
    total_time = trace.elapsed_ms()
    load_time = trace.elapsed_ms("load")
    split_time = trace.elapsed_ms("split")
    
    print(f"总耗时: {total_time:.2f} ms")
    print(f"加载阶段耗时 (从记录中读取): {load_time} ms")
    print(f"切分阶段耗时 (从记录中读取): {split_time} ms")
    
    # 5. 序列化检查
    print("\n[5] 序列化结果 (to_dict):")
    trace_dict = trace.to_dict()
    print(json.dumps(trace_dict, indent=2, ensure_ascii=False))
    
    # 验证关键字段
    assert "trace_id" in trace_dict
    assert trace_dict["trace_type"] == "ingestion"
    assert len(trace_dict["stages"]) == 2
    assert trace_dict["total_elapsed_ms"] >= 80  # 至少大于等于两个阶段耗时之和的模拟值
    
    print("\n=== F1 阶段测试通过！ ===")

if __name__ == "__main__":
    try:
        test_trace_context_manual()
    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        sys.exit(1)
