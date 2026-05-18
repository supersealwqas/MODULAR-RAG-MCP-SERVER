"""响应构建器模块。

将检索结果转换为 MCP Tool 响应格式，包含：
- Markdown 格式的可读文本（含引用标注 [1]、[2] 等）
- structuredContent.citations 结构化引用列表
- 多模态返回（Text + Image）
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.core.response.citation_generator import Citation, CitationGenerator
from src.core.response.multimodal_assembler import assemble_multimodal_response
from src.core.types import RetrievalResult


class ResponseBuilder:
    """MCP 响应构建器。

    将检索结果转换为 MCP Tool 标准响应格式。
    """

    @classmethod
    def build(
        cls,
        results: List[RetrievalResult],
        query: str,
        include_images: bool = False,
    ) -> Dict[str, Any]:
        """构建 MCP Tool 响应。

        参数:
            results: 检索结果列表（已排序）
            query: 原始查询文本
            include_images: 是否包含图片（多模态返回）

        返回:
            MCP Tool 响应字典，包含 content 和 structuredContent
        """
        # 生成引用列表
        citations = CitationGenerator.generate(results)

        # 构建 Markdown 文本
        markdown_text = cls._build_markdown(query, results, citations)

        # 构建 structuredContent
        structured_content = {
            "query": query,
            "result_count": len(results),
            "citations": [c.to_dict() for c in citations],
        }

        # 构建 content（支持多模态）
        if include_images:
            content = assemble_multimodal_response(results, markdown_text)
        else:
            content = [{"type": "text", "text": markdown_text}]

        return {
            "content": content,
            "structuredContent": structured_content,
            "isError": False,
        }

    @classmethod
    def build_empty(cls, query: str) -> Dict[str, Any]:
        """构建空结果响应。

        参数:
            query: 原始查询文本

        返回:
            友好的"未找到结果"响应
        """
        markdown_text = (
            f"## 查询结果\n\n"
            f"**查询**: {query}\n\n"
            f"未找到相关文档。请确认：\n"
            f"1. 已运行 `ingest.py` 摄取数据\n"
            f"2. 查询关键词与文档内容相关\n"
            f"3. 集合名称正确（如指定了 collection）"
        )

        return {
            "content": [{"type": "text", "text": markdown_text}],
            "structuredContent": {
                "query": query,
                "result_count": 0,
                "citations": [],
            },
            "isError": False,
        }

    @classmethod
    def _build_markdown(
        cls,
        query: str,
        results: List[RetrievalResult],
        citations: List[Citation],
    ) -> str:
        """构建 Markdown 格式的响应文本。

        参数:
            query: 原始查询
            results: 检索结果
            citations: 引用列表

        返回:
            Markdown 文本
        """
        lines = [
            f"## 查询结果",
            f"",
            f"**查询**: {query}",
            f"**命中**: {len(results)} 条结果",
            f"",
            f"---",
            f"",
        ]

        for i, (result, citation) in enumerate(zip(results, citations), start=1):
            # 引用标注
            lines.append(f"### [{i}] 来源: {citation.source}")
            if citation.page:
                lines.append(f"**页码**: {citation.page}")
            lines.append(f"**相关度**: {result.score:.4f}")
            lines.append(f"")
            lines.append(result.text)
            lines.append(f"")
            lines.append(f"---")
            lines.append(f"")

        # 尾部引用汇总
        if citations:
            lines.append("## 引用列表")
            lines.append("")
            for c in citations:
                page_info = f" (页码 {c.page})" if c.page else ""
                lines.append(f"[{c.index}] {c.source}{page_info} - chunk_id: {c.chunk_id}")

        return "\n".join(lines)
