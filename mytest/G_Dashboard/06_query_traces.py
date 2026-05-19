"""手动测试 G6: Dashboard Query 追踪页面。

用法:
    uv run python mytest/G_Dashboard/06_query_traces.py

验证项:
    1. query_traces 页面可导入
    2. app.py 中 page_query_traces 已接入真实页面
    3. 辅助函数正确性
    4. TraceService 查询过滤功能
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_page_import():
    """测试页面可导入。"""
    print("=" * 50)
    print("测试 1: query_traces 页面可导入")
    print("=" * 50)

    from src.observability.dashboard.pages.query_traces import (
        render_query_traces,
    )
    print(f"  render_query_traces: {render_query_traces}")
    assert callable(render_query_traces)

    print("\n✅ 页面导入测试通过\n")


def test_app_wiring():
    """测试 app.py 已接入。"""
    print("=" * 50)
    print("测试 2: app.py 页面接入")
    print("=" * 50)

    import inspect
    from src.observability.dashboard.app import page_query_traces

    source = inspect.getsource(page_query_traces)
    has_real = "render_query_traces" in source
    has_placeholder = "_placeholder_page" in source

    print(f"  使用 render_query_traces: {has_real}")
    print(f"  使用 _placeholder_page: {has_placeholder}")

    assert has_real, "未接入真实页面"
    assert not has_placeholder, "仍使用占位页面"

    print("\n✅ app.py 接入测试通过\n")


def test_helpers():
    """测试辅助函数。"""
    print("=" * 50)
    print("测试 3: 辅助函数")
    print("=" * 50)

    from src.observability.dashboard.pages.query_traces import (
        _stage_label,
        _extract_query_text,
        _filter_traces,
    )
    from src.observability.dashboard.services.trace_service import TraceRecord

    # 阶段标签
    labels = {
        "query_processing": "查询处理",
        "dense_retrieval": "Dense 检索",
        "sparse_retrieval": "Sparse 检索",
        "fusion": "RRF 融合",
        "hybrid_search": "混合检索",
        "reranker": "Rerank 重排",
    }
    for eng, chn in labels.items():
        assert _stage_label(eng) == chn
        print(f"  {eng} -> {chn}")

    # 查询文本提取
    record = TraceRecord.from_dict({
        "trace_id": "test",
        "trace_type": "query",
        "started_at": 100.0,
        "total_elapsed_ms": 50.0,
        "stages": [{
            "name": "query_processing",
            "original_query": "测试查询",
        }],
    })
    assert _extract_query_text(record) == "测试查询"
    print(f"  查询文本提取: OK")

    # 过滤
    traces = [
        TraceRecord.from_dict({
            "trace_id": "aaa",
            "trace_type": "query",
            "started_at": 1.0,
            "total_elapsed_ms": 10.0,
            "stages": [{"name": "query_processing", "original_query": "Ollama 配置"}],
        }),
        TraceRecord.from_dict({
            "trace_id": "bbb",
            "trace_type": "query",
            "started_at": 2.0,
            "total_elapsed_ms": 20.0,
            "stages": [{"name": "query_processing", "original_query": "RAG 概念"}],
        }),
    ]
    filtered = _filter_traces(traces, "Ollama")
    assert len(filtered) == 1
    print(f"  过滤 'Ollama': {len(filtered)} 条")

    print("\n✅ 辅助函数测试通过\n")


def test_trace_service_query():
    """测试 TraceService 查询功能。"""
    print("=" * 50)
    print("测试 4: TraceService 查询功能")
    print("=" * 50)

    from src.observability.dashboard.services.trace_service import TraceService

    # 写入临时 traces.jsonl
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        f.write(json.dumps({
            "trace_id": "query001aaabbb",
            "trace_type": "query",
            "started_at": 1700000000.0,
            "finished_at": 1700000001.0,
            "total_elapsed_ms": 1000.0,
            "stages": [
                {"name": "query_processing", "elapsed_ms": 10.0, "method": "jieba",
                 "original_query": "如何配置 Ollama？", "keywords": ["配置", "Ollama"]},
                {"name": "dense_retrieval", "elapsed_ms": 200.0, "method": "vector_search",
                 "result_count": 10},
                {"name": "sparse_retrieval", "elapsed_ms": 150.0, "method": "bm25",
                 "result_count": 8},
                {"name": "fusion", "elapsed_ms": 5.0, "method": "rrf", "k": 60,
                 "result_count": 10},
                {"name": "reranker", "elapsed_ms": 500.0, "method": "cross_encoder",
                 "input_count": 10, "output_count": 5, "fallback": False},
            ],
        }) + "\n")
        # ingestion 记录（应被过滤）
        f.write(json.dumps({
            "trace_id": "ingest001aaa",
            "trace_type": "ingestion",
            "started_at": 1700000010.0,
            "total_elapsed_ms": 5000.0,
            "stages": [],
        }) + "\n")
        path = f.name

    try:
        service = TraceService(traces_file=path)
        query_traces = service.get_query_traces()
        assert len(query_traces) == 1

        trace = query_traces[0]
        assert trace.trace_id == "query001aaabbb"
        print(f"  query 记录数: {len(query_traces)}")

        # 阶段解析
        assert len(trace.stages) == 5
        print(f"  阶段数: {len(trace.stages)}")

        # stage_map
        smap = trace.stage_map
        assert "query_processing" in smap
        assert "dense_retrieval" in smap
        assert "sparse_retrieval" in smap
        assert "fusion" in smap
        assert "reranker" in smap
        print(f"  stage_map keys: {list(smap.keys())}")

        # reranker 详情
        reranker = smap["reranker"]
        assert reranker.extra["input_count"] == 10
        assert reranker.extra["output_count"] == 5
        assert reranker.extra["fallback"] is False
        print(f"  reranker: input={reranker.extra['input_count']}, output={reranker.extra['output_count']}")

    finally:
        os.unlink(path)

    print("\n✅ TraceService 查询功能测试通过\n")


if __name__ == "__main__":
    print("🧪 G6 Query 追踪页面 — 手动测试\n")
    test_page_import()
    test_app_wiring()
    test_helpers()
    test_trace_service_query()
    print("=" * 50)
    print("所有手动测试完成！")
    print("启动 Dashboard: uv run python scripts/start_dashboard.py")
    print("=" * 50)
