"""手动测试 G3: Dashboard 数据浏览器页面。

用法:
    uv run python mytest/G_Dashboard/03_data_browser.py

验证项:
    1. DataService 可导入并可实例化
    2. data_browser 页面可导入
    3. render_data_browser 函数签名正确
    4. app.py 中 page_data_browser 已接入真实页面
    5. 辅助函数正确性
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_data_service_import():
    """测试 DataService 可导入。"""
    print("=" * 50)
    print("测试 1: DataService 可导入")
    print("=" * 50)

    from src.observability.dashboard.services.data_service import DataService

    service = DataService()
    print(f"  DataService 实例: {service}")
    print(f"  list_documents 方法: {service.list_documents}")
    print(f"  get_document_detail 方法: {service.get_document_detail}")
    print(f"  get_collection_stats 方法: {service.get_collection_stats}")
    print(f"  list_collections 方法: {service.list_collections}")

    print("\n✅ DataService 导入测试通过\n")


def test_data_browser_import():
    """测试 data_browser 页面可导入。"""
    print("=" * 50)
    print("测试 2: data_browser 页面可导入")
    print("=" * 50)

    from src.observability.dashboard.pages.data_browser import render_data_browser

    print(f"  render_data_browser: {render_data_browser}")
    assert callable(render_data_browser)

    print("\n✅ data_browser 页面导入测试通过\n")


def test_app_wiring():
    """测试 app.py 中 data_browser 已接入。"""
    print("=" * 50)
    print("测试 3: app.py 页面接入")
    print("=" * 50)

    import inspect
    from src.observability.dashboard.app import page_data_browser

    source = inspect.getsource(page_data_browser)
    has_real_import = "render_data_browser" in source
    has_placeholder = "_placeholder_page" in source

    print(f"  使用 render_data_browser: {has_real_import}")
    print(f"  使用 _placeholder_page: {has_placeholder}")

    assert has_real_import, "page_data_browser 未接入真实页面"
    assert not has_placeholder, "page_data_browser 仍使用占位页面"

    print("\n✅ app.py 页面接入测试通过\n")


def test_helper_functions():
    """测试辅助函数。"""
    print("=" * 50)
    print("测试 4: 辅助函数")
    print("=" * 50)

    from src.observability.dashboard.pages.data_browser import (
        _shorten_path,
        _format_file_size,
    )

    # 路径缩短
    assert _shorten_path("/short.pdf") == "/short.pdf"
    long = "/very/long/path/to/some/deep/nested/document.pdf"
    shortened = _shorten_path(long, max_len=40)
    print(f"  路径缩短: {long} -> {shortened}")
    assert "..." in shortened

    # 文件大小
    assert _format_file_size(500) == "500 B"
    assert "KB" in _format_file_size(2048)
    assert "MB" in _format_file_size(2 * 1024 * 1024)
    print(f"  500 B -> {_format_file_size(500)}")
    print(f"  2048 B -> {_format_file_size(2048)}")
    print(f"  2MB -> {_format_file_size(2 * 1024 * 1024)}")

    print("\n✅ 辅助函数测试通过\n")


def test_mock_workflow():
    """测试 mock 完整工作流。"""
    print("=" * 50)
    print("测试 5: Mock DataService 工作流")
    print("=" * 50)

    from unittest.mock import MagicMock
    from src.observability.dashboard.services.data_service import DataService
    from src.ingestion.document_manager import (
        DocumentInfo,
        DocumentDetail,
        CollectionStats,
    )

    service = DataService()
    mock_dm = MagicMock()
    service._document_manager = mock_dm

    # list_documents
    mock_dm.list_documents.return_value = [
        DocumentInfo(
            source_path="/data/test.pdf",
            file_hash="abc",
            collection="default",
            chunk_count=5,
            image_count=2,
            processed_at="2026-05-19",
            file_size=1024,
        ),
    ]
    docs = service.list_documents()
    print(f"  list_documents: {len(docs)} 个文档")
    assert len(docs) == 1

    # get_document_detail
    mock_dm.get_document_detail.return_value = DocumentDetail(
        info=docs[0],
        chunks=[{"id": "c1", "text": "hello", "metadata": {}}],
        images=[{"image_id": "img1"}],
    )
    detail = service.get_document_detail("/data/test.pdf")
    print(f"  get_document_detail: {len(detail.chunks)} chunks, {len(detail.images)} images")
    assert detail is not None

    # get_collection_stats
    mock_dm.get_collection_stats.return_value = CollectionStats(
        collection="all", document_count=1, chunk_count=5, image_count=2,
    )
    stats = service.get_collection_stats()
    print(f"  get_collection_stats: {stats.document_count} docs, {stats.chunk_count} chunks")
    assert stats.document_count == 1

    # list_collections
    collections = service.list_collections()
    print(f"  list_collections: {collections}")
    assert "default" in collections

    print("\n✅ Mock DataService 工作流测试通过\n")


if __name__ == "__main__":
    print("🧪 G3 数据浏览器页面 — 手动测试\n")
    test_data_service_import()
    test_data_browser_import()
    test_app_wiring()
    test_helper_functions()
    test_mock_workflow()
    print("=" * 50)
    print("所有手动测试完成！")
    print("启动 Dashboard: uv run python scripts/start_dashboard.py")
    print("=" * 50)
