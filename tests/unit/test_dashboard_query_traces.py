"""Query 追踪页面单元测试。

测试内容：
- 页面可导入
- app.py 接入
- 辅助函数（stage_label、extract_query_text、filter_traces）
"""

import pytest


class TestPageImports:
    """页面导入与接入测试。"""

    def test_page_importable(self):
        """query_traces 页面可导入。"""
        from src.observability.dashboard.pages.query_traces import (
            render_query_traces,
        )
        assert callable(render_query_traces)

    def test_app_wiring(self):
        """app.py 中 page_query_traces 已接入真实页面。"""
        import inspect
        from src.observability.dashboard.app import page_query_traces

        source = inspect.getsource(page_query_traces)
        assert "render_query_traces" in source
        assert "_placeholder_page" not in source


class TestStageLabel:
    """阶段标签测试。"""

    def test_known_stages(self):
        """已知阶段返回中文标签。"""
        from src.observability.dashboard.pages.query_traces import _stage_label

        assert _stage_label("query_processing") == "查询处理"
        assert _stage_label("dense_retrieval") == "Dense 检索"
        assert _stage_label("sparse_retrieval") == "Sparse 检索"
        assert _stage_label("fusion") == "RRF 融合"
        assert _stage_label("hybrid_search") == "混合检索"
        assert _stage_label("reranker") == "Rerank 重排"

    def test_unknown_stage(self):
        """未知阶段原样返回。"""
        from src.observability.dashboard.pages.query_traces import _stage_label

        assert _stage_label("unknown") == "unknown"


class TestExtractQueryText:
    """查询文本提取测试。"""

    def test_with_query_processing(self):
        """有 query_processing 阶段时提取查询文本。"""
        from src.observability.dashboard.pages.query_traces import _extract_query_text
        from src.observability.dashboard.services.trace_service import TraceRecord

        record = TraceRecord.from_dict({
            "trace_id": "test",
            "trace_type": "query",
            "started_at": 100.0,
            "total_elapsed_ms": 50.0,
            "stages": [
                {
                    "name": "query_processing",
                    "original_query": "如何配置 Ollama？",
                    "keywords": ["配置", "Ollama"],
                },
            ],
        })
        assert _extract_query_text(record) == "如何配置 Ollama？"

    def test_without_query_processing(self):
        """无 query_processing 阶段时返回空。"""
        from src.observability.dashboard.pages.query_traces import _extract_query_text
        from src.observability.dashboard.services.trace_service import TraceRecord

        record = TraceRecord.from_dict({
            "trace_id": "test",
            "trace_type": "query",
            "started_at": 100.0,
            "total_elapsed_ms": 50.0,
            "stages": [],
        })
        assert _extract_query_text(record) == ""


class TestFilterTraces:
    """过滤测试。"""

    def _make_traces(self):
        from src.observability.dashboard.services.trace_service import TraceRecord
        return [
            TraceRecord.from_dict({
                "trace_id": "aaa111bbb",
                "trace_type": "query",
                "started_at": 100.0,
                "total_elapsed_ms": 50.0,
                "stages": [{
                    "name": "query_processing",
                    "original_query": "如何配置 Ollama？",
                }],
            }),
            TraceRecord.from_dict({
                "trace_id": "ccc222ddd",
                "trace_type": "query",
                "started_at": 200.0,
                "total_elapsed_ms": 30.0,
                "stages": [{
                    "name": "query_processing",
                    "original_query": "什么是 RAG？",
                }],
            }),
        ]

    def test_filter_by_query(self):
        """按查询文本过滤。"""
        from src.observability.dashboard.pages.query_traces import _filter_traces

        traces = self._make_traces()
        result = _filter_traces(traces, "Ollama")
        assert len(result) == 1
        assert result[0].trace_id == "aaa111bbb"

    def test_filter_by_trace_id(self):
        """按 trace_id 过滤。"""
        from src.observability.dashboard.pages.query_traces import _filter_traces

        traces = self._make_traces()
        result = _filter_traces(traces, "ccc222")
        assert len(result) == 1
        assert result[0].trace_id == "ccc222ddd"

    def test_filter_no_match(self):
        """无匹配返回空。"""
        from src.observability.dashboard.pages.query_traces import _filter_traces

        traces = self._make_traces()
        result = _filter_traces(traces, "不存在的关键词")
        assert len(result) == 0

    def test_filter_case_insensitive(self):
        """大小写不敏感。"""
        from src.observability.dashboard.pages.query_traces import _filter_traces

        traces = self._make_traces()
        result = _filter_traces(traces, "ollama")
        assert len(result) == 1
