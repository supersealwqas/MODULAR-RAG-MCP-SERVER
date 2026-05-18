"""引用生成器模块。

从 RetrievalResult 列表生成结构化引用信息（Citation），
用于 MCP Tool 响应中的 structuredContent.citations 字段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.core.types import RetrievalResult


@dataclass
class Citation:
    """单条引用信息。

    属性:
        index: 引用序号（从 1 开始）
        source: 来源文件路径
        page: 页码（可选）
        chunk_id: Chunk 唯一标识
        score: 相关性分数
        text_preview: 文本预览（截取前 200 字符）
        metadata: 附加元数据
    """
    index: int
    source: str
    chunk_id: str
    score: float
    page: int = 0
    text_preview: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        return {
            "index": self.index,
            "source": self.source,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
            "text_preview": self.text_preview,
        }


class CitationGenerator:
    """引用生成器。

    从检索结果列表生成结构化引用信息。
    """

    # 文本预览最大长度
    MAX_PREVIEW_LENGTH = 200

    @classmethod
    def generate(cls, results: List[RetrievalResult]) -> List[Citation]:
        """从检索结果生成引用列表。

        参数:
            results: 检索结果列表

        返回:
            Citation 列表，按原始顺序排列，index 从 1 开始
        """
        citations = []
        for i, result in enumerate(results, start=1):
            source = result.metadata.get("source_path", "未知来源")
            page = result.metadata.get("page", 0)

            # 截取文本预览
            text_preview = result.text[:cls.MAX_PREVIEW_LENGTH]
            if len(result.text) > cls.MAX_PREVIEW_LENGTH:
                text_preview += "..."

            citations.append(Citation(
                index=i,
                source=source,
                page=page,
                chunk_id=result.chunk_id,
                score=result.score,
                text_preview=text_preview,
                metadata=result.metadata,
            ))

        return citations
