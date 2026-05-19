"""手动测试 G5: Dashboard Ingestion 追踪页面。

用法:
    uv run python mytest/G_Dashboard/05_ingestion_traces.py

验证项:
    1. TraceService 可导入并正常工作
    2. ingestion_traces 页面可导入
    3. app.py 中 page_ingestion_traces 已接入真实页面
    4. TraceRecord / TraceStage 数据类正确性
    5. 辅助函数正确性
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_trace_service_import():
    """测试 TraceService 可导入。"""
    print("=" * 50)
    print("测试 1: TraceService 可导入")
    print("=" * 50)

    from src.observability.dashboard.services.trace_service import (
        TraceService,
        TraceRecord,
        TraceStage,
    )
    print(f"  TraceService: {TraceService}")
    print(f"  TraceRecord: {TraceRecord}")
    print(f"  TraceStage: {TraceStage}")
    assert callable(TraceService)

    print("\n✅ TraceService 导入测试通过\n")


def test_page_import():
    """测试页面可导入。"""
    print("=" * 50)
    print("测试 2: ingestion_traces 页面可导入")
    print("=" * 50)

    from src.observability.dashboard.pages.ingestion_traces import (
        render_ingestion_traces,
    )
    print(f"  render_ingestion_traces: {render_ingestion_traces}")
    assert callable(render_ingestion_traces)

    print("\n✅ 页面导入测试通过\n")


def test_app_wiring():
    """测试 app.py 已接入。"""
    print("=" * 50)
    print("测试 3: app.py 页面接入")
    print("=" * 50)

    import inspect
    from src.observability.dashboard.app import page_ingestion_traces

    source = inspect.getsource(page_ingestion_traces)
    has_real = "render_ingestion_traces" in source
    has_placeholder = "_placeholder_page" in source

    print(f"  使用 render_ingestion_traces: {has_real}")
    print(f"  使用 _placeholder_page: {has_placeholder}")

    assert has_real, "未接入真实页面"
    assert not has_placeholder, "仍使用占位页面"

    print("\n✅ app.py 接入测试通过\n")


def test_trace_service_functional():
    """测试 TraceService 功能。"""
    print("=" * 50)
    print("测试 4: TraceService 功能")
    print("=" * 50)

    from src.observability.dashboard.services.trace_service import TraceService

    # 写入临时 traces.jsonl
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        f.write(json.dumps({
            "trace_id": "test001aaabbbccc",
            "trace_type": "ingestion",
            "started_at": 1700000000.0,
            "finished_at": 1700000005.0,
            "total_elapsed_ms": 5000.0,
            "stages": [
                {"name": "load", "elapsed_ms": 100.0, "method": "PdfLoader"},
                {"name": "split", "elapsed_ms": 200.0, "method": "DocumentChunker"},
                {"name": "upsert", "elapsed_ms": 300.0, "method": "VectorUpserter"},
            ],
        }) + "\n")
        f.write(json.dumps({
            "trace_id": "query002aaabbbcc",
            "trace_type": "query",
            "started_at": 1700000010.0,
            "total_elapsed_ms": 200.0,
            "stages": [],
        }) + "\n")
        path = f.name

    try:
        service = TraceService(traces_file=path)

        # ingestion 过滤
        ingestion = service.get_ingestion_traces()
        assert len(ingestion) == 1
        assert ingestion[0].trace_id == "test001aaabbbccc"
        print(f"  ingestion 记录数: {len(ingestion)}")

        # query 过滤
        query = service.get_query_traces()
        assert len(query) == 1
        print(f"  query 记录数: {len(query)}")

        # 阶段解析
        stages = ingestion[0].stages
        assert len(stages) == 3
        assert stages[0].name == "load"
        assert stages[0].elapsed_ms == 100.0
        print(f"  阶段数: {len(stages)}")

        # stage_map
        smap = ingestion[0].stage_map
        assert "load" in smap
        assert "split" in smap
        print(f"  stage_map keys: {list(smap.keys())}")

        # 统计
        stats = service.get_stats()
        assert stats["total"] == 2
        assert stats["ingestion"] == 1
        assert stats["query"] == 1
        print(f"  统计: {stats}")

    finally:
        os.unlink(path)

    print("\n✅ TraceService 功能测试通过\n")


def test_helpers():
    """测试辅助函数。"""
    print("=" * 50)
    print("测试 5: 辅助函数")
    print("=" * 50)

    from src.observability.dashboard.pages.ingestion_traces import _stage_label

    labels = {
        "integrity": "完整性检查",
        "load": "加载文件",
        "split": "文本切分",
        "transform": "增强处理",
        "embed": "向量编码",
        "upsert": "存储写入",
    }
    for eng, chn in labels.items():
        assert _stage_label(eng) == chn
        print(f"  {eng} -> {chn}")

    # 未知阶段
    assert _stage_label("unknown") == "unknown"
    print(f"  unknown -> unknown")

    print("\n✅ 辅助函数测试通过\n")


if __name__ == "__main__":
    print("🧪 G5 Ingestion 追踪页面 — 手动测试\n")
    test_trace_service_import()
    test_page_import()
    test_app_wiring()
    test_trace_service_functional()
    test_helpers()
    print("=" * 50)
    print("所有手动测试完成！")
    print("启动 Dashboard: uv run python scripts/start_dashboard.py")
    print("=" * 50)
