"""Ingestion 管理页面单元测试。

测试内容：
- 页面可导入
- app.py 中 page_ingestion_manager 已接入真实页面
- 辅助函数正确性
"""

import pytest
from unittest.mock import MagicMock

from src.ingestion.document_manager import DocumentInfo


class TestIngestionManagerImports:
    """页面导入测试。"""

    def test_page_importable(self):
        """ingestion_manager 页面可导入。"""
        from src.observability.dashboard.pages.ingestion_manager import (
            render_ingestion_manager,
        )
        assert callable(render_ingestion_manager)

    def test_app_wiring(self):
        """app.py 中 page_ingestion_manager 已接入真实页面。"""
        import inspect
        from src.observability.dashboard.app import page_ingestion_manager

        source = inspect.getsource(page_ingestion_manager)
        assert "render_ingestion_manager" in source
        assert "_placeholder_page" not in source


class TestHelperFunctions:
    """辅助函数测试。"""

    def test_stage_label_known(self):
        """已知阶段返回中文标签。"""
        from src.observability.dashboard.pages.ingestion_manager import _stage_label

        assert _stage_label("integrity") == "完整性检查"
        assert _stage_label("load") == "加载文件"
        assert _stage_label("split") == "文本切分"
        assert _stage_label("transform") == "增强处理"
        assert _stage_label("encode") == "向量编码"
        assert _stage_label("store") == "存储写入"

    def test_stage_label_unknown(self):
        """未知阶段原样返回。"""
        from src.observability.dashboard.pages.ingestion_manager import _stage_label

        assert _stage_label("custom_stage") == "custom_stage"

    def test_format_size_bytes(self):
        """字节级格式化。"""
        from src.observability.dashboard.pages.ingestion_manager import _format_size

        assert _format_size(500) == "500 B"

    def test_format_size_kb(self):
        """KB 级格式化。"""
        from src.observability.dashboard.pages.ingestion_manager import _format_size

        assert "KB" in _format_size(2048)

    def test_format_size_mb(self):
        """MB 级格式化。"""
        from src.observability.dashboard.pages.ingestion_manager import _format_size

        assert "MB" in _format_size(2 * 1024 * 1024)

    def test_format_size_none(self):
        """None 返回未知。"""
        from src.observability.dashboard.pages.ingestion_manager import _format_size

        assert _format_size(None) == "未知"

    def test_shorten_name_short(self):
        """短文件名不变。"""
        from src.observability.dashboard.pages.ingestion_manager import _shorten_name

        assert _shorten_name("/path/to/doc.pdf") == "doc.pdf"

    def test_shorten_name_long(self):
        """长文件名被截断。"""
        from src.observability.dashboard.pages.ingestion_manager import _shorten_name

        long_name = "/path/to/very_long_filename_that_exceeds_limit.pdf"
        result = _shorten_name(long_name, max_len=20)
        assert len(result) <= 20
        assert "..." in result
