"""Ingestion 追踪页面单元测试。

测试内容：
- TraceService 解析与查询
- 页面可导入
- app.py 接入
- 辅助函数
"""

import json
import os
import tempfile

import pytest

from src.observability.dashboard.services.trace_service import (
    TraceRecord,
    TraceService,
    TraceStage,
)


# ============================================================
# TraceStage 测试
# ============================================================


class TestTraceStage:
    """TraceStage 数据类测试。"""

    def test_from_dict_basic(self):
        """基本字段解析。"""
        data = {
            "name": "load",
            "elapsed_ms": 123.4,
            "method": "PdfLoader",
            "timestamp": 1000.0,
        }
        stage = TraceStage.from_dict(data)
        assert stage.name == "load"
        assert stage.elapsed_ms == 123.4
        assert stage.method == "PdfLoader"

    def test_from_dict_extra_fields(self):
        """额外字段存入 extra。"""
        data = {
            "name": "load",
            "elapsed_ms": 50.0,
            "text_length": 1000,
            "image_count": 2,
        }
        stage = TraceStage.from_dict(data)
        assert stage.extra["text_length"] == 1000
        assert stage.extra["image_count"] == 2

    def test_from_dict_missing_fields(self):
        """缺失字段使用默认值。"""
        data = {"name": "split"}
        stage = TraceStage.from_dict(data)
        assert stage.name == "split"
        assert stage.elapsed_ms == 0.0
        assert stage.method == ""


# ============================================================
# TraceRecord 测试
# ============================================================


class TestTraceRecord:
    """TraceRecord 数据类测试。"""

    def test_from_dict(self):
        """完整字典解析。"""
        data = {
            "trace_id": "abc12345def67890",
            "trace_type": "ingestion",
            "started_at": 1700000000.0,
            "finished_at": 1700000005.0,
            "total_elapsed_ms": 5000.0,
            "stages": [
                {"name": "load", "elapsed_ms": 100.0},
                {"name": "split", "elapsed_ms": 200.0},
            ],
        }
        record = TraceRecord.from_dict(data)
        assert record.trace_id == "abc12345def67890"
        assert record.trace_type == "ingestion"
        assert record.total_elapsed_ms == 5000.0
        assert len(record.stages) == 2
        assert record.stages[0].name == "load"

    def test_started_datetime(self):
        """started_at 转换为 datetime。"""
        data = {
            "trace_id": "test",
            "trace_type": "ingestion",
            "started_at": 1700000000.0,
            "total_elapsed_ms": 100.0,
            "stages": [],
        }
        record = TraceRecord.from_dict(data)
        dt = record.started_datetime
        assert dt.year == 2023

    def test_stage_map(self):
        """stage_map 按名称索引。"""
        data = {
            "trace_id": "test",
            "trace_type": "ingestion",
            "started_at": 1700000000.0,
            "total_elapsed_ms": 100.0,
            "stages": [
                {"name": "load", "elapsed_ms": 50.0},
                {"name": "split", "elapsed_ms": 50.0},
            ],
        }
        record = TraceRecord.from_dict(data)
        smap = record.stage_map
        assert "load" in smap
        assert "split" in smap
        assert smap["load"].elapsed_ms == 50.0

    def test_empty_stages(self):
        """无阶段数据。"""
        data = {
            "trace_id": "empty",
            "trace_type": "ingestion",
            "started_at": 1700000000.0,
            "total_elapsed_ms": 0.0,
        }
        record = TraceRecord.from_dict(data)
        assert record.stages == []
        assert record.stage_map == {}


# ============================================================
# TraceService 测试
# ============================================================


def _write_traces(path: str, traces: list) -> None:
    """写入临时 traces.jsonl。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


class TestTraceServiceLoad:
    """TraceService 加载测试。"""

    def test_file_not_exists(self):
        """文件不存在返回空列表。"""
        service = TraceService(traces_file="/nonexistent/traces.jsonl")
        assert service.get_ingestion_traces() == []
        assert service.get_stats()["total"] == 0

    def test_empty_file(self):
        """空文件返回空列表。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write("")
            path = f.name
        try:
            service = TraceService(traces_file=path)
            assert service.get_ingestion_traces() == []
        finally:
            os.unlink(path)

    def test_parse_ingestion_traces(self):
        """解析 ingestion 类型记录。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            # ingestion 记录
            f.write(json.dumps({
                "trace_id": "aaa111",
                "trace_type": "ingestion",
                "started_at": 1700000000.0,
                "finished_at": 1700000005.0,
                "total_elapsed_ms": 5000.0,
                "stages": [{"name": "load", "elapsed_ms": 100.0}],
            }) + "\n")
            # query 记录（应被过滤）
            f.write(json.dumps({
                "trace_id": "bbb222",
                "trace_type": "query",
                "started_at": 1700000010.0,
                "total_elapsed_ms": 200.0,
                "stages": [],
            }) + "\n")
            # 另一条 ingestion
            f.write(json.dumps({
                "trace_id": "ccc333",
                "trace_type": "ingestion",
                "started_at": 1700000020.0,
                "total_elapsed_ms": 3000.0,
                "stages": [],
            }) + "\n")
            path = f.name
        try:
            service = TraceService(traces_file=path)
            ingestion = service.get_ingestion_traces()
            assert len(ingestion) == 2
            # 按时间倒序
            assert ingestion[0].trace_id == "ccc333"
            assert ingestion[1].trace_id == "aaa111"
        finally:
            os.unlink(path)

    def test_parse_malformed_lines(self):
        """格式错误的行被跳过。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write("not json\n")
            f.write(json.dumps({
                "trace_id": "good",
                "trace_type": "ingestion",
                "started_at": 1700000000.0,
                "total_elapsed_ms": 100.0,
                "stages": [],
            }) + "\n")
            f.write('{"incomplete": true\n')
            path = f.name
        try:
            service = TraceService(traces_file=path)
            traces = service.get_ingestion_traces()
            assert len(traces) == 1
            assert traces[0].trace_id == "good"
        finally:
            os.unlink(path)


class TestTraceServiceQuery:
    """TraceService 查询测试。"""

    def _make_service(self, traces: list) -> TraceService:
        """创建带数据的临时服务。"""
        self._tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        for t in traces:
            self._tmp.write(json.dumps(t, ensure_ascii=False) + "\n")
        self._tmp.close()
        return TraceService(traces_file=self._tmp.name)

    def teardown_method(self):
        """清理临时文件。"""
        if hasattr(self, "_tmp") and os.path.exists(self._tmp.name):
            os.unlink(self._tmp.name)

    def test_get_query_traces(self):
        """获取 query 类型记录。"""
        service = self._make_service([
            {"trace_id": "q1", "trace_type": "query", "started_at": 100.0,
             "total_elapsed_ms": 50.0, "stages": []},
            {"trace_id": "i1", "trace_type": "ingestion", "started_at": 200.0,
             "total_elapsed_ms": 100.0, "stages": []},
        ])
        query = service.get_query_traces()
        assert len(query) == 1
        assert query[0].trace_id == "q1"

    def test_get_trace_by_id(self):
        """按 ID 查找。"""
        service = self._make_service([
            {"trace_id": "findme", "trace_type": "ingestion",
             "started_at": 100.0, "total_elapsed_ms": 50.0, "stages": []},
        ])
        found = service.get_trace_by_id("findme")
        assert found is not None
        assert found.trace_id == "findme"

        not_found = service.get_trace_by_id("nope")
        assert not_found is None

    def test_get_stats(self):
        """统计信息。"""
        service = self._make_service([
            {"trace_id": "a", "trace_type": "ingestion", "started_at": 1.0,
             "total_elapsed_ms": 10.0, "stages": []},
            {"trace_id": "b", "trace_type": "query", "started_at": 2.0,
             "total_elapsed_ms": 20.0, "stages": []},
            {"trace_id": "c", "trace_type": "ingestion", "started_at": 3.0,
             "total_elapsed_ms": 30.0, "stages": []},
        ])
        stats = service.get_stats()
        assert stats["total"] == 3
        assert stats["ingestion"] == 2
        assert stats["query"] == 1

    def test_cache_invalidation(self):
        """缓存失效后重新加载。"""
        service = self._make_service([
            {"trace_id": "old", "trace_type": "ingestion", "started_at": 1.0,
             "total_elapsed_ms": 10.0, "stages": []},
        ])
        # 首次加载
        assert len(service.get_ingestion_traces()) == 1

        # 追加数据
        with open(self._tmp.name, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "trace_id": "new", "trace_type": "ingestion",
                "started_at": 2.0, "total_elapsed_ms": 20.0, "stages": [],
            }) + "\n")

        # 有缓存时还是旧数据
        assert len(service.get_ingestion_traces()) == 1

        # 清除缓存后拿到新数据
        service.invalidate_cache()
        assert len(service.get_ingestion_traces()) == 2


# ============================================================
# 页面导入与接入测试
# ============================================================


class TestPageImports:
    """页面导入与 app.py 接入测试。"""

    def test_page_importable(self):
        """ingestion_traces 页面可导入。"""
        from src.observability.dashboard.pages.ingestion_traces import (
            render_ingestion_traces,
        )
        assert callable(render_ingestion_traces)

    def test_trace_service_importable(self):
        """TraceService 可导入。"""
        from src.observability.dashboard.services.trace_service import TraceService
        assert callable(TraceService)

    def test_app_wiring(self):
        """app.py 中 page_ingestion_traces 已接入真实页面。"""
        import inspect
        from src.observability.dashboard.app import page_ingestion_traces

        source = inspect.getsource(page_ingestion_traces)
        assert "render_ingestion_traces" in source
        assert "_placeholder_page" not in source


# ============================================================
# 辅助函数测试
# ============================================================


class TestHelperFunctions:
    """辅助函数测试。"""

    def test_stage_label_known(self):
        """已知阶段返回中文标签。"""
        from src.observability.dashboard.pages.ingestion_traces import _stage_label

        assert _stage_label("integrity") == "完整性检查"
        assert _stage_label("load") == "加载文件"
        assert _stage_label("split") == "文本切分"
        assert _stage_label("transform") == "增强处理"
        assert _stage_label("embed") == "向量编码"
        assert _stage_label("upsert") == "存储写入"

    def test_stage_label_unknown(self):
        """未知阶段原样返回。"""
        from src.observability.dashboard.pages.ingestion_traces import _stage_label

        assert _stage_label("custom_stage") == "custom_stage"
