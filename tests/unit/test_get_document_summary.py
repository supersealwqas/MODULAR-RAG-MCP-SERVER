"""
get_document_summary Tool 单元测试

覆盖：正常返回、文档不存在、空 doc_id、错误处理。
使用 mock 隔离 ChromaDB 依赖。
"""

import pytest
from unittest.mock import MagicMock, patch

from src.mcp_server.tools.get_document_summary import get_document_summary


@pytest.fixture
def mock_settings():
    """创建 mock Settings。"""
    settings = MagicMock()
    settings.vector_store.persist_directory = "/tmp/test_chroma"
    return settings


@pytest.fixture
def mock_chroma_store():
    """创建 mock ChromaStore。"""
    store = MagicMock()
    return store


class TestGetDocumentSummary:
    """get_document_summary 测试。"""

    @pytest.mark.anyio
    async def test_returns_content(self, mock_settings, mock_chroma_store):
        """应返回包含 content 的响应。"""
        mock_chroma_store.get_by_ids.return_value = [{
            "id": "chunk_001",
            "text": "测试文本",
            "metadata": {
                "title": "测试文档",
                "summary": "这是一个测试文档的摘要",
                "tags": ["test", "demo"],
                "source_path": "/path/to/doc.pdf",
                "page": 1,
            },
        }]

        with patch(
            "src.mcp_server.tools.get_document_summary._get_chroma_store",
            return_value=mock_chroma_store,
        ):
            result = await get_document_summary("chunk_001", settings=mock_settings)

        assert "content" in result
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"

    @pytest.mark.anyio
    async def test_returns_structured_content(self, mock_settings, mock_chroma_store):
        """应返回正确的结构化内容。"""
        mock_chroma_store.get_by_ids.return_value = [{
            "id": "chunk_001",
            "text": "测试文本",
            "metadata": {
                "title": "测试文档",
                "summary": "这是一个测试文档的摘要",
                "tags": ["test", "demo"],
                "source_path": "/path/to/doc.pdf",
                "page": 1,
            },
        }]

        with patch(
            "src.mcp_server.tools.get_document_summary._get_chroma_store",
            return_value=mock_chroma_store,
        ):
            result = await get_document_summary("chunk_001", settings=mock_settings)

        sc = result["structuredContent"]
        assert sc["doc_id"] == "chunk_001"
        assert sc["found"] is True
        assert sc["title"] == "测试文档"
        assert sc["summary"] == "这是一个测试文档的摘要"
        assert sc["tags"] == ["test", "demo"]
        assert sc["source_path"] == "/path/to/doc.pdf"
        assert sc["page"] == 1

    @pytest.mark.anyio
    async def test_is_error_false(self, mock_settings, mock_chroma_store):
        """成功时 isError 应为 False。"""
        mock_chroma_store.get_by_ids.return_value = [{
            "id": "chunk_001",
            "text": "测试文本",
            "metadata": {"source_path": "/path/to/doc.pdf"},
        }]

        with patch(
            "src.mcp_server.tools.get_document_summary._get_chroma_store",
            return_value=mock_chroma_store,
        ):
            result = await get_document_summary("chunk_001", settings=mock_settings)

        assert result["isError"] is False

    @pytest.mark.anyio
    async def test_document_not_found(self, mock_settings, mock_chroma_store):
        """文档不存在时应返回 found=False。"""
        mock_chroma_store.get_by_ids.return_value = []

        with patch(
            "src.mcp_server.tools.get_document_summary._get_chroma_store",
            return_value=mock_chroma_store,
        ):
            result = await get_document_summary("nonexistent", settings=mock_settings)

        assert result["isError"] is False
        sc = result["structuredContent"]
        assert sc["found"] is False
        assert "未找到文档" in result["content"][0]["text"]

    @pytest.mark.anyio
    async def test_empty_doc_id(self, mock_settings, mock_chroma_store):
        """空 doc_id 应返回错误。"""
        result = await get_document_summary("", settings=mock_settings)
        assert result["isError"] is True
        assert "doc_id 不能为空" in result["content"][0]["text"]

    @pytest.mark.anyio
    async def test_whitespace_doc_id(self, mock_settings, mock_chroma_store):
        """空白 doc_id 应返回错误。"""
        result = await get_document_summary("   ", settings=mock_settings)
        assert result["isError"] is True

    @pytest.mark.anyio
    async def test_default_values(self, mock_settings, mock_chroma_store):
        """metadata 缺少字段时应使用默认值。"""
        mock_chroma_store.get_by_ids.return_value = [{
            "id": "chunk_001",
            "text": "测试文本",
            "metadata": {},  # 空 metadata
        }]

        with patch(
            "src.mcp_server.tools.get_document_summary._get_chroma_store",
            return_value=mock_chroma_store,
        ):
            result = await get_document_summary("chunk_001", settings=mock_settings)

        sc = result["structuredContent"]
        assert sc["found"] is True
        assert sc["title"] == "未知标题"
        assert sc["summary"] == "暂无摘要"
        assert sc["tags"] == []

    @pytest.mark.anyio
    async def test_markdown_format(self, mock_settings, mock_chroma_store):
        """Markdown 应包含必要信息。"""
        mock_chroma_store.get_by_ids.return_value = [{
            "id": "chunk_001",
            "text": "测试文本",
            "metadata": {
                "title": "测试文档",
                "summary": "测试摘要",
                "tags": ["ai", "llm"],
                "source_path": "/docs/guide.pdf",
            },
        }]

        with patch(
            "src.mcp_server.tools.get_document_summary._get_chroma_store",
            return_value=mock_chroma_store,
        ):
            result = await get_document_summary("chunk_001", settings=mock_settings)

        text = result["content"][0]["text"]
        assert "## 文档摘要" in text
        assert "chunk_001" in text
        assert "测试文档" in text
        assert "测试摘要" in text
        assert "ai, llm" in text

    @pytest.mark.anyio
    async def test_serializable(self, mock_settings, mock_chroma_store):
        """响应应可被 json.dumps 序列化。"""
        import json

        mock_chroma_store.get_by_ids.return_value = [{
            "id": "chunk_001",
            "text": "测试文本",
            "metadata": {
                "title": "测试文档",
                "summary": "测试摘要",
                "tags": ["test"],
            },
        }]

        with patch(
            "src.mcp_server.tools.get_document_summary._get_chroma_store",
            return_value=mock_chroma_store,
        ):
            result = await get_document_summary("chunk_001", settings=mock_settings)

        # 不应抛出 TypeError
        serialized = json.dumps(result, ensure_ascii=False)
        assert "chunk_001" in serialized

    @pytest.mark.anyio
    async def test_chroma_error_returns_error_response(self, mock_settings, mock_chroma_store):
        """ChromaDB 异常时应返回错误响应。"""
        mock_chroma_store.get_by_ids.side_effect = RuntimeError("连接失败")

        with patch(
            "src.mcp_server.tools.get_document_summary._get_chroma_store",
            return_value=mock_chroma_store,
        ):
            result = await get_document_summary("chunk_001", settings=mock_settings)

        assert result["isError"] is True
        assert "连接失败" in result["content"][0]["text"]
