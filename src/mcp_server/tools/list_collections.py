"""list_collections MCP Tool 模块。

列出所有可用的知识库集合及其统计信息。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.core.settings import Settings

logger = logging.getLogger(__name__)

# 全局 ChromaDB client（延迟初始化）
_chroma_client = None


def _get_chroma_client(settings: Settings):
    """获取 ChromaDB client 实例（延迟创建）。"""
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client

    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
    except ImportError:
        raise ImportError("请安装 chromadb 库: uv pip install chromadb")

    persist_dir = settings.vector_store.persist_directory
    _chroma_client = chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    return _chroma_client


async def list_collections(
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """列出所有可用的知识库集合。

    参数:
        settings: Settings 实例（可选，不传则自动加载）

    返回:
        MCP Tool 响应字典，包含集合列表和统计信息
    """
    logger.info("list_collections 被调用")

    # 加载配置
    if settings is None:
        from src.core.settings import load_settings
        settings = load_settings()

    try:
        client = _get_chroma_client(settings)
        collections = client.list_collections()

        # 获取每个集合的统计信息
        collection_info = []
        for coll in collections:
            name = coll.name if hasattr(coll, "name") else str(coll)
            try:
                count = coll.count() if hasattr(coll, "count") else 0
            except Exception:
                count = 0
            collection_info.append({
                "name": name,
                "document_count": count,
            })

        # 按名称排序
        collection_info.sort(key=lambda x: x["name"])

    except Exception as e:
        logger.error("列出集合失败: %s", e, exc_info=True)
        return {
            "content": [{"type": "text", "text": f"列出集合失败: {e}"}],
            "structuredContent": {"collections": [], "error": str(e)},
            "isError": True,
        }

    # 构建 Markdown 响应
    if not collection_info:
        markdown = (
            "## 知识库集合\n\n"
            "当前没有可用的集合。请先运行 `ingest.py` 摄取数据：\n\n"
            "```bash\n"
            "python scripts/ingest.py --path data/documents/ --collection default\n"
            "```"
        )
    else:
        lines = [
            "## 知识库集合",
            "",
            f"共 {len(collection_info)} 个集合：",
            "",
            "| 集合名称 | 文档数量 |",
            "|---------|---------|",
        ]
        for info in collection_info:
            lines.append(f"| {info['name']} | {info['document_count']} |")

        markdown = "\n".join(lines)

    logger.info("返回 %d 个集合", len(collection_info))

    return {
        "content": [{"type": "text", "text": markdown}],
        "structuredContent": {
            "collections": collection_info,
            "total_count": len(collection_info),
        },
        "isError": False,
    }
