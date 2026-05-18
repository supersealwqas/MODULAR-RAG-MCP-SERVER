"""
ResponseBuilder 与 CitationGenerator 单元测试

覆盖：引用生成、响应构建、空结果处理、边界情况。
"""

import pytest

from src.core.response.citation_generator import Citation, CitationGenerator
from src.core.response.response_builder import ResponseBuilder
from src.core.types import RetrievalResult


# ==================== 测试数据 ====================


def make_retrieval_result(
    chunk_id: str = "chunk_001",
    score: float = 0.95,
    text: str = "这是一段测试文本。",
    source_path: str = "/path/to/doc.pdf",
    page: int = 1,
) -> RetrievalResult:
    """创建测试用 RetrievalResult。"""
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text,
        metadata={"source_path": source_path, "page": page},
    )


def make_results(n: int = 3) -> list[RetrievalResult]:
    """创建 n 条测试结果。"""
    return [
        make_retrieval_result(
            chunk_id=f"chunk_{i:03d}",
            score=1.0 - i * 0.1,
            text=f"第 {i+1} 段测试文本内容。",
            source_path=f"/path/to/doc_{i+1}.pdf",
            page=i + 1,
        )
        for i in range(n)
    ]


# ==================== CitationGenerator 测试 ====================


class TestCitationGenerator:
    """CitationGenerator 测试。"""

    def test_generate_returns_citations(self):
        """应生成正确数量的引用。"""
        results = make_results(3)
        citations = CitationGenerator.generate(results)
        assert len(citations) == 3

    def test_citation_index_starts_from_1(self):
        """引用序号应从 1 开始。"""
        results = make_results(2)
        citations = CitationGenerator.generate(results)
        assert citations[0].index == 1
        assert citations[1].index == 2

    def test_citation_source_from_metadata(self):
        """应从 metadata 中提取 source_path。"""
        result = make_retrieval_result(source_path="/test/guide.pdf")
        citations = CitationGenerator.generate([result])
        assert citations[0].source == "/test/guide.pdf"

    def test_citation_source_default_unknown(self):
        """无 source_path 时应显示 '未知来源'。"""
        result = RetrievalResult(chunk_id="c1", score=0.9, text="text", metadata={})
        citations = CitationGenerator.generate([result])
        assert citations[0].source == "未知来源"

    def test_citation_page_from_metadata(self):
        """应从 metadata 中提取 page。"""
        result = make_retrieval_result(page=5)
        citations = CitationGenerator.generate([result])
        assert citations[0].page == 5

    def test_citation_page_default_zero(self):
        """无 page 时应默认为 0。"""
        result = RetrievalResult(chunk_id="c1", score=0.9, text="text", metadata={})
        citations = CitationGenerator.generate([result])
        assert citations[0].page == 0

    def test_citation_chunk_id(self):
        """应正确设置 chunk_id。"""
        result = make_retrieval_result(chunk_id="my_chunk_123")
        citations = CitationGenerator.generate([result])
        assert citations[0].chunk_id == "my_chunk_123"

    def test_citation_score(self):
        """应正确设置 score。"""
        result = make_retrieval_result(score=0.8765)
        citations = CitationGenerator.generate([result])
        assert citations[0].score == 0.8765

    def test_text_preview_truncation(self):
        """长文本应被截断到 MAX_PREVIEW_LENGTH。"""
        long_text = "A" * 300
        result = make_retrieval_result(text=long_text)
        citations = CitationGenerator.generate([result])
        assert len(citations[0].text_preview) <= CitationGenerator.MAX_PREVIEW_LENGTH + 3  # +3 for "..."
        assert citations[0].text_preview.endswith("...")

    def test_text_preview_no_truncation_for_short(self):
        """短文本不应被截断。"""
        short_text = "短文本"
        result = make_retrieval_result(text=short_text)
        citations = CitationGenerator.generate([result])
        assert citations[0].text_preview == short_text
        assert not citations[0].text_preview.endswith("...")

    def test_to_dict_structure(self):
        """to_dict 应返回正确的字典结构。"""
        result = make_retrieval_result()
        citations = CitationGenerator.generate([result])
        d = citations[0].to_dict()
        assert "index" in d
        assert "source" in d
        assert "page" in d
        assert "chunk_id" in d
        assert "score" in d
        assert "text_preview" in d

    def test_empty_results(self):
        """空结果应返回空列表。"""
        citations = CitationGenerator.generate([])
        assert citations == []


# ==================== ResponseBuilder 测试 ====================


class TestResponseBuilder:
    """ResponseBuilder 测试。"""

    def test_build_returns_content(self):
        """应返回包含 content 的响应。"""
        results = make_results(2)
        response = ResponseBuilder.build(results, "测试查询")
        assert "content" in response
        assert len(response["content"]) == 1
        assert response["content"][0]["type"] == "text"

    def test_build_markdown_contains_query(self):
        """Markdown 应包含原始查询。"""
        results = make_results(1)
        response = ResponseBuilder.build(results, "如何配置 Ollama？")
        text = response["content"][0]["text"]
        assert "如何配置 Ollama？" in text

    def test_build_markdown_contains_citation_numbers(self):
        """Markdown 应包含引用标注 [1]、[2] 等。"""
        results = make_results(3)
        response = ResponseBuilder.build(results, "测试")
        text = response["content"][0]["text"]
        assert "[1]" in text
        assert "[2]" in text
        assert "[3]" in text

    def test_build_structured_content(self):
        """应返回 structuredContent 字段。"""
        results = make_results(2)
        response = ResponseBuilder.build(results, "测试")
        assert "structuredContent" in response
        sc = response["structuredContent"]
        assert sc["query"] == "测试"
        assert sc["result_count"] == 2
        assert len(sc["citations"]) == 2

    def test_build_citations_structure(self):
        """citations 应包含必要字段。"""
        results = make_results(1)
        response = ResponseBuilder.build(results, "测试")
        citation = response["structuredContent"]["citations"][0]
        assert "index" in citation
        assert "source" in citation
        assert "page" in citation
        assert "chunk_id" in citation
        assert "score" in citation

    def test_build_is_error_false(self):
        """成功响应时 isError 应为 False。"""
        results = make_results(1)
        response = ResponseBuilder.build(results, "测试")
        assert response["isError"] is False

    def test_build_empty_returns_friendly_message(self):
        """空结果应返回友好提示。"""
        response = ResponseBuilder.build_empty("不存在的查询")
        text = response["content"][0]["text"]
        assert "未找到相关文档" in text
        assert "不存在的查询" in text

    def test_build_empty_structured_content(self):
        """空结果的 structuredContent 应为空。"""
        response = ResponseBuilder.build_empty("测试")
        sc = response["structuredContent"]
        assert sc["result_count"] == 0
        assert sc["citations"] == []

    def test_build_empty_is_error_false(self):
        """空结果的 isError 应为 False（不是错误，只是无结果）。"""
        response = ResponseBuilder.build_empty("测试")
        assert response["isError"] is False

    def test_build_markdown_contains_source_info(self):
        """Markdown 应包含来源文件信息。"""
        results = make_retrieval_result(source_path="/docs/guide.pdf", page=3)
        response = ResponseBuilder.build([results], "测试")
        text = response["content"][0]["text"]
        assert "/docs/guide.pdf" in text

    def test_build_serializable(self):
        """响应应可被 json.dumps 序列化。"""
        import json
        results = make_results(2)
        response = ResponseBuilder.build(results, "测试")
        # 不应抛出 TypeError
        serialized = json.dumps(response, ensure_ascii=False)
        assert "测试" in serialized
