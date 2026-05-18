import time
import pytest
from src.core.trace.trace_context import TraceContext
from src.core.trace.trace_collector import TraceCollector

def test_trace_context_initialization():
    """测试追踪上下文初始化。"""
    trace = TraceContext(trace_type="ingestion")
    assert len(trace.trace_id) == 16
    assert trace.trace_type == "ingestion"
    assert trace.stages == []
    assert trace._finished_at is None

def test_trace_context_record_stage():
    """测试记录阶段。"""
    trace = TraceContext()
    trace.record_stage("load", method="markitdown", details={"size": 1024})
    
    assert len(trace.stages) == 1
    assert trace.stages[0]["name"] == "load"
    assert trace.stages[0]["method"] == "markitdown"
    assert trace.stages[0]["details"]["size"] == 1024
    assert "timestamp" in trace.stages[0]

def test_trace_context_finish():
    """测试结束追踪。"""
    trace = TraceContext()
    assert trace._finished_at is None
    
    trace.finish()
    assert trace._finished_at is not None
    assert trace._finished_at >= trace._started_at

def test_trace_context_elapsed_ms():
    """测试耗时统计。"""
    trace = TraceContext()
    time.sleep(0.01)  # 模拟操作
    
    # 未结束时的总耗时
    elapsed = trace.elapsed_ms()
    assert elapsed >= 10
    
    # 记录阶段耗时
    trace.record_stage("split", elapsed_ms=5.5)
    assert trace.elapsed_ms("split") == 5.5
    assert trace.elapsed_ms("non_existent") == 0.0
    
    # 结束后的总耗时
    trace.finish()
    final_elapsed = trace.elapsed_ms()
    assert final_elapsed >= elapsed

def test_trace_context_to_dict():
    """测试序列化。"""
    trace = TraceContext(trace_type="query")
    trace.record_stage("dense_retrieval", elapsed_ms=20.0)
    trace.finish()
    
    data = trace.to_dict()
    assert data["trace_id"] == trace.trace_id
    assert data["trace_type"] == "query"
    assert data["total_elapsed_ms"] == trace.elapsed_ms()
    assert len(data["stages"]) == 1
    assert data["stages"][0]["name"] == "dense_retrieval"
    assert "finished_at" in data

def test_trace_collector_collect():
    """测试收集器。"""
    collector = TraceCollector()
    trace = TraceContext()
    trace.record_stage("test")
    
    collector.collect(trace)
    
    # 验证 collect 会自动触发 finish
    assert trace._finished_at is not None
