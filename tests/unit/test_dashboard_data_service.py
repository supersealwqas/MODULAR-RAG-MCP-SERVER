"""Dashboard DataService 与数据浏览器页面单元测试。

测试内容：
- DataService 封装 DocumentManager 的查询接口
- data_browser 页面辅助函数
- app.py 页面注册与导入
"""

import pytest
from unittest.mock import MagicMock, patch

from src.ingestion.document_manager import (
    CollectionStats,
    DocumentDetail,
    DocumentInfo,
)


# ============================================================
# DataService 测试
# ============================================================


class TestDataService:
    """DataService 测试。"""

    @patch("src.observability.dashboard.services.data_service.load_settings")
    @patch("src.observability.dashboard.services.data_service.SQLiteIntegrityChecker")
    @patch("src.observability.dashboard.services.data_service.ImageStorage")
    @patch("src.observability.dashboard.services.data_service.BM25Indexer")
    @patch("src.observability.dashboard.services.data_service.ChromaStore")
    def test_list_documents(
        self, mock_chroma, mock_bm25, mock_image, mock_integrity, mock_settings
    ):
        """测试 list_documents 调用 DocumentManager。"""
        from src.observability.dashboard.services.data_service import DataService

        service = DataService()

        # Mock DocumentManager
        mock_dm = MagicMock()
        mock_dm.list_documents.return_value = [
            DocumentInfo(
                source_path="/doc.pdf",
                file_hash="abc",
                collection="default",
                chunk_count=5,
                image_count=2,
                processed_at="2026-05-19",
            )
        ]
        service._document_manager = mock_dm

        result = service.list_documents()

        assert len(result) == 1
        assert result[0].source_path == "/doc.pdf"
        mock_dm.list_documents.assert_called_once_with(collection=None)

    @patch("src.observability.dashboard.services.data_service.load_settings")
    @patch("src.observability.dashboard.services.data_service.SQLiteIntegrityChecker")
    @patch("src.observability.dashboard.services.data_service.ImageStorage")
    @patch("src.observability.dashboard.services.data_service.BM25Indexer")
    @patch("src.observability.dashboard.services.data_service.ChromaStore")
    def test_get_document_detail(
        self, mock_chroma, mock_bm25, mock_image, mock_integrity, mock_settings
    ):
        """测试 get_document_detail 调用 DocumentManager。"""
        from src.observability.dashboard.services.data_service import DataService

        service = DataService()
        mock_dm = MagicMock()
        mock_dm.get_document_detail.return_value = DocumentDetail(
            info=DocumentInfo(
                source_path="/doc.pdf",
                file_hash="abc",
                collection="default",
                chunk_count=2,
                image_count=0,
                processed_at="2026-05-19",
            ),
            chunks=[{"id": "c1", "text": "hello", "metadata": {}}],
            images=[],
        )
        service._document_manager = mock_dm

        result = service.get_document_detail("/doc.pdf")

        assert result is not None
        assert len(result.chunks) == 1
        mock_dm.get_document_detail.assert_called_once_with("/doc.pdf")

    @patch("src.observability.dashboard.services.data_service.load_settings")
    @patch("src.observability.dashboard.services.data_service.SQLiteIntegrityChecker")
    @patch("src.observability.dashboard.services.data_service.ImageStorage")
    @patch("src.observability.dashboard.services.data_service.BM25Indexer")
    @patch("src.observability.dashboard.services.data_service.ChromaStore")
    def test_get_collection_stats(
        self, mock_chroma, mock_bm25, mock_image, mock_integrity, mock_settings
    ):
        """测试 get_collection_stats 调用 DocumentManager。"""
        from src.observability.dashboard.services.data_service import DataService

        service = DataService()
        mock_dm = MagicMock()
        mock_dm.get_collection_stats.return_value = CollectionStats(
            collection="all",
            document_count=3,
            chunk_count=15,
            image_count=5,
        )
        service._document_manager = mock_dm

        result = service.get_collection_stats()

        assert result.document_count == 3
        assert result.chunk_count == 15

    @patch("src.observability.dashboard.services.data_service.load_settings")
    @patch("src.observability.dashboard.services.data_service.SQLiteIntegrityChecker")
    @patch("src.observability.dashboard.services.data_service.ImageStorage")
    @patch("src.observability.dashboard.services.data_service.BM25Indexer")
    @patch("src.observability.dashboard.services.data_service.ChromaStore")
    def test_list_collections(
        self, mock_chroma, mock_bm25, mock_image, mock_integrity, mock_settings
    ):
        """测试 list_collections 去重返回集合名。"""
        from src.observability.dashboard.services.data_service import DataService

        service = DataService()
        mock_dm = MagicMock()
        mock_dm.list_documents.return_value = [
            DocumentInfo(
                source_path="/a.pdf", file_hash="a", collection="test",
                chunk_count=1, image_count=0, processed_at="2026-05-19",
            ),
            DocumentInfo(
                source_path="/b.pdf", file_hash="b", collection="default",
                chunk_count=1, image_count=0, processed_at="2026-05-19",
            ),
            DocumentInfo(
                source_path="/c.pdf", file_hash="c", collection="test",
                chunk_count=1, image_count=0, processed_at="2026-05-19",
            ),
        ]
        service._document_manager = mock_dm

        result = service.list_collections()

        assert result == ["default", "test"]  # sorted


# ============================================================
# data_browser 辅助函数测试
# ============================================================


class TestDataBrowserHelpers:
    """data_browser 页面辅助函数测试。"""

    def test_shorten_path_short(self):
        """短路径不变。"""
        from src.observability.dashboard.pages.data_browser import _shorten_path

        assert _shorten_path("/short/path.pdf") == "/short/path.pdf"

    def test_shorten_path_long(self):
        """长路径被缩短。"""
        from src.observability.dashboard.pages.data_browser import _shorten_path

        long_path = "/very/long/path/to/some/deep/nested/directory/document.pdf"
        result = _shorten_path(long_path, max_len=40)
        assert "..." in result
        assert "document.pdf" in result

    def test_format_file_size_bytes(self):
        """字节级文件大小。"""
        from src.observability.dashboard.pages.data_browser import _format_file_size

        assert _format_file_size(500) == "500 B"

    def test_format_file_size_kb(self):
        """KB 级文件大小。"""
        from src.observability.dashboard.pages.data_browser import _format_file_size

        result = _format_file_size(2048)
        assert "KB" in result
        assert "2.0" in result

    def test_format_file_size_mb(self):
        """MB 级文件大小。"""
        from src.observability.dashboard.pages.data_browser import _format_file_size

        result = _format_file_size(2 * 1024 * 1024)
        assert "MB" in result
        assert "2.0" in result


# ============================================================
# 页面导入测试
# ============================================================


class TestPageImports:
    """页面与服务可导入测试。"""

    def test_data_browser_page_importable(self):
        """data_browser 页面可导入。"""
        from src.observability.dashboard.pages.data_browser import render_data_browser
        assert callable(render_data_browser)

    def test_data_service_importable(self):
        """DataService 可导入。"""
        from src.observability.dashboard.services.data_service import DataService
        assert callable(DataService)

    def test_app_importable(self):
        """app.py 可导入（无语法错误）。"""
        from src.observability.dashboard.app import main
        assert callable(main)
