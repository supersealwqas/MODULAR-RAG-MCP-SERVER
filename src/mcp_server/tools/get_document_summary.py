"""get_document_summary MCP Tool 模块。

按 doc_id 返回文档的摘要信息（title/summary/tags）。
从 ChromaDB 的 metadata 中提取，如果不存在则提供默认值。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.core.settings import Settings

logger = logging.getLogger(__name__)

# 全局 ChromaStore 实例（延迟初始化）
_chroma_store = None


def _get_chroma_store(settings: Settings):
    """获取 ChromaStore 实例（延迟创建）。"""
    global _chroma_store
    if _chroma_store is not None:
        return _chroma_store

    from src.libs.vector_store.chroma_store import ChromaStore
    _chroma_store = ChromaStore(
        collection_name="default",
        persist_directory=settings.vector_store.persist_directory,
    )
    return _chroma_store


async def get_document_summary(
    doc_id: str,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """获取指定文档的摘要信息。

    参数:
        doc_id: 文档 ID（ChromaDB 中的记录 ID）
        settings: Settings 实例（可选，不传则自动加载）

    返回:
        MCP Tool 响应字典，包含文档摘要信息
    """
    logger.info("get_document_summary 被调用: doc_id=%s", doc_id)

    # 参数校验
    if not doc_id or not doc_id.strip():
        return {
            "content": [{"type": "text", "text": "错误：doc_id 不能为空"}],
            "structuredContent": {"error": "doc_id 不能为空"},
            "isError": True,
        }

    # 加载配置
    if settings is None:
        from src.core.settings import load_settings
        settings = load_settings()

    # 从 ChromaDB 获取文档
    try:
        store = _get_chroma_store(settings)
        records = store.get_by_ids([doc_id.strip()])

        if not records:
            return {
                "content": [{"type": "text", "text": f"未找到文档：{doc_id}"}],
                "structuredContent": {
                    "doc_id": doc_id,
                    "found": False,
                    "error": "文档不存在",
                },
                "isError": False,
            }

        record = records[0]
        metadata = record.get("metadata", {})

        # 提取摘要信息
        title = metadata.get("title", metadata.get("source_path", "未知标题"))
        summary = metadata.get("summary", "暂无摘要")
        tags = metadata.get("tags", [])
        source_path = metadata.get("source_path", "未知来源")
        page = metadata.get("page", 0)
        chunk_count = metadata.get("chunk_count", 1)

        # 构建 Markdown 响应
        markdown = f"""## 文档摘要

**文档 ID**: `{doc_id}`

**标题**: {title}

**来源**: {source_path}

**摘要**: {summary}

**标签**: {', '.join(tags) if tags else '无'}

**页码**: {page}

**分块数**: {chunk_count}
"""

        logger.info("返回文档 %s 的摘要信息", doc_id)

        return {
            "content": [{"type": "text", "text": markdown}],
            "structuredContent": {
                "doc_id": doc_id,
                "found": True,
                "title": title,
                "summary": summary,
                "tags": tags,
                "source_path": source_path,
                "page": page,
                "chunk_count": chunk_count,
                "metadata": metadata,
            },
            "isError": False,
        }

    except Exception as e:
        logger.error("获取文档摘要失败: %s", e, exc_info=True)
        return {
            "content": [{"type": "text", "text": f"获取文档摘要失败: {e}"}],
            "structuredContent": {"doc_id": doc_id, "error": str(e)},
            "isError": True,
        }
