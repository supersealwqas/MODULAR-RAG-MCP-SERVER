"""
list_collections Tool 单元测试

覆盖：正常返回、空集合、错误处理。
使用 mock 隔离 ChromaDB 依赖。
"""

import pytest
from unittest.mock import MagicMock, patch

from src.mcp_server.tools.list_collections import list_collections


@pytest.fixture
def mock_settings():
    """创建 mock Settings。"""
    settings = MagicMock()
    settings.vector_store.persist_directory = "/tmp/test_chroma"
    return settings


@pytest.fixture
def mock_chroma_client():
    """创建 mock ChromaDB client。"""
    client = MagicMock()
    return client


class TestListCollections:
    """list_collections 测试。"""

    @pytest.mark.anyio
    async def test_returns_content(self, mock_settings, mock_chroma_client):
        """应返回包含 content 的响应。"""
        # 设置 mock
        mock_coll = MagicMock()
        mock_coll.name = "default"
        mock_coll.count.return_value = 42
        mock_chroma_client.list_collections.return_value = [mock_coll]

        with patch(
            "src.mcp_server.tools.list_collections._get_chroma_client",
            return_value=mock_chroma_client,
        ):
            result = await list_collections(settings=mock_settings)

        assert "content" in result
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"

    @pytest.mark.anyio
    async def test_returns_collections(self, mock_settings, mock_chroma_client):
        """应返回正确的集合信息。"""
        mock_coll1 = MagicMock()
        mock_coll1.name = "docs"
        mock_coll1.count.return_value = 10

        mock_coll2 = MagicMock()
        mock_coll2.name = "knowledge"
        mock_coll2.count.return_value = 25

        mock_chroma_client.list_collections.return_value = [mock_coll1, mock_coll2]

        with patch(
            "src.mcp_server.tools.list_collections._get_chroma_client",
            return_value=mock_chroma_client,
        ):
            result = await list_collections(settings=mock_settings)

        sc = result["structuredContent"]
        assert sc["total_count"] == 2
        assert len(sc["collections"]) == 2
        # 按名称排序
        assert sc["collections"][0]["name"] == "docs"
        assert sc["collections"][0]["document_count"] == 10
        assert sc["collections"][1]["name"] == "knowledge"
        assert sc["collections"][1]["document_count"] == 25

    @pytest.mark.anyio
    async def test_is_error_false(self, mock_settings, mock_chroma_client):
        """成功时 isError 应为 False。"""
        mock_chroma_client.list_collections.return_value = []

        with patch(
            "src.mcp_server.tools.list_collections._get_chroma_client",
            return_value=mock_chroma_client,
        ):
            result = await list_collections(settings=mock_settings)

        assert result["isError"] is False

    @pytest.mark.anyio
    async def test_empty_collections(self, mock_settings, mock_chroma_client):
        """无集合时应返回友好提示。"""
        mock_chroma_client.list_collections.return_value = []

        with patch(
            "src.mcp_server.tools.list_collections._get_chroma_client",
            return_value=mock_chroma_client,
        ):
            result = await list_collections(settings=mock_settings)

        text = result["content"][0]["text"]
        assert "当前没有可用的集合" in text
        assert result["structuredContent"]["total_count"] == 0

    @pytest.mark.anyio
    async def test_markdown_format(self, mock_settings, mock_chroma_client):
        """Markdown 应包含表格格式。"""
        mock_coll = MagicMock()
        mock_coll.name = "test"
        mock_coll.count.return_value = 5
        mock_chroma_client.list_collections.return_value = [mock_coll]

        with patch(
            "src.mcp_server.tools.list_collections._get_chroma_client",
            return_value=mock_chroma_client,
        ):
            result = await list_collections(settings=mock_settings)

        text = result["content"][0]["text"]
        assert "## 知识库集合" in text
        assert "| 集合名称 | 文档数量 |" in text
        assert "| test | 5 |" in text

    @pytest.mark.anyio
    async def test_serializable(self, mock_settings, mock_chroma_client):
        """响应应可被 json.dumps 序列化。"""
        import json

        mock_coll = MagicMock()
        mock_coll.name = "default"
        mock_coll.count.return_value = 10
        mock_chroma_client.list_collections.return_value = [mock_coll]

        with patch(
            "src.mcp_server.tools.list_collections._get_chroma_client",
            return_value=mock_chroma_client,
        ):
            result = await list_collections(settings=mock_settings)

        # 不应抛出 TypeError
        serialized = json.dumps(result, ensure_ascii=False)
        assert "default" in serialized

    @pytest.mark.anyio
    async def test_chroma_error_returns_error_response(self, mock_settings, mock_chroma_client):
        """ChromaDB 异常时应返回错误响应。"""
        mock_chroma_client.list_collections.side_effect = RuntimeError("连接失败")

        with patch(
            "src.mcp_server.tools.list_collections._get_chroma_client",
            return_value=mock_chroma_client,
        ):
            result = await list_collections(settings=mock_settings)

        assert result["isError"] is True
        assert "连接失败" in result["content"][0]["text"]
