"""
MCP Server 入口模块

遵循 MCP 规范，通过 Stdio Transport 与客户端通信。
核心约束：stdout 仅输出 MCP 消息，日志统一输出至 stderr。

关键设计约束：
1. 所有重量级依赖必须在模块级别预导入。在 FastMCP 的 anyio 事件循环中
   执行 import 语句（尤其是 chromadb、FlagEmbedding 等涉及 C 扩展或
   网络初始化的库）会导致死锁。
2. 所有耗时同步操作（BGE 模型加载、HybridSearch 检索等）必须通过
   asyncio.to_thread 放入线程池执行，避免阻塞事件循环导致 stdin/stdout
   I/O 死锁。
"""

import asyncio
import io
import json
import logging
import sys

# 关键修复：预导入重量级 C 扩展库
# 必须在此模块最外层执行导入，否则在 FastMCP 的 anyio 事件循环启动后，
# 若由 asyncio.to_thread 中的后台线程触发延迟导入（如 BGEEmbedding 内部），
# 会由于 Windows 上的 Python GIL 与 C 扩展内部锁的交互导致死锁。
import torch
try:
    import FlagEmbedding
except ImportError:
    pass

import chromadb
from chromadb.config import Settings as ChromaSettings
from mcp.server.fastmcp import FastMCP

from src.core.settings import load_settings as _load_settings
from src.core.settings import Settings as _Settings
from src.mcp_server.tools.query_knowledge_hub import _search_sync as _query_knowledge_hub_sync

# 配置日志输出到 stderr（UTF-8 编码），避免污染 stdout 的 MCP 消息通道
_stderr_handler = logging.StreamHandler(
    io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
)
_stderr_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
)
logging.basicConfig(level=logging.INFO, handlers=[_stderr_handler])
logger = logging.getLogger(__name__)

# 创建 MCP Server 实例
server = FastMCP(
    name="modular-rag-mcp-server",
)

# 预加载配置（避免在工具函数中重复读取 YAML）
_settings: _Settings | None = None


def _get_settings() -> _Settings:
    """获取全局配置（延迟加载一次）。"""
    global _settings
    if _settings is None:
        _settings = _load_settings()
    return _settings


def _get_chroma_client(collection_name: str = ""):
    """获取 ChromaDB 客户端和集合。"""
    settings = _get_settings()
    persist_dir = settings.vector_store.persist_directory
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    if collection_name:
        return client, client.get_or_create_collection(collection_name)
    return client, None


# ============================================================
# 同步业务逻辑（在线程池中执行，不阻塞事件循环）
# ============================================================


def _do_query(query: str, top_k: int, collection: str) -> list:
    """执行知识库查询（同步，供 asyncio.to_thread 调用）。"""
    logger.info("query_knowledge_hub 被调用: query=%s, top_k=%d, collection=%s", query, top_k, collection)

    filters = {}
    if collection:
        filters["collection"] = collection

    settings = _get_settings()
    result = _query_knowledge_hub_sync(
        query=query,
        top_k=top_k,
        filters=filters,
        settings=settings,
    )
    return result.get("content", [])


def _do_list_collections() -> list:
    """列出集合（同步，供 asyncio.to_thread 调用）。"""
    logger.info("list_collections 被调用")

    try:
        client, _ = _get_chroma_client()
        collections = client.list_collections()

        collection_info = []
        for coll in collections:
            name = coll.name if hasattr(coll, "name") else str(coll)
            try:
                count = coll.count() if hasattr(coll, "count") else 0
            except Exception:
                count = 0
            collection_info.append({"name": name, "document_count": count})

        collection_info.sort(key=lambda x: x["name"])

    except Exception as e:
        logger.error("列出集合失败: %s", e, exc_info=True)
        return [{"type": "text", "text": f"列出集合失败: {e}"}]

    if not collection_info:
        markdown = (
            "## 知识库集合\n\n"
            "当前没有可用的集合。请先运行 `ingest.py` 摄取数据。"
        )
    else:
        lines = [
            "## 知识库集合", "",
            f"共 {len(collection_info)} 个集合：", "",
            "| 集合名称 | 文档数量 |",
            "|---------|---------|",
        ]
        for info in collection_info:
            lines.append(f"| {info['name']} | {info['document_count']} |")
        markdown = "\n".join(lines)

    return [{"type": "text", "text": markdown}]


def _do_get_document_summary(doc_id: str) -> list:
    """获取文档摘要（同步，供 asyncio.to_thread 调用）。"""
    logger.info("get_document_summary 被调用: doc_id=%s", doc_id)

    try:
        client, _ = _get_chroma_client()

        for coll in client.list_collections():
            try:
                results = coll.get(where={"doc_id": doc_id}, limit=1)
                if results and results["ids"]:
                    metadata = results["metadatas"][0] if results["metadatas"] else {}
                    markdown = (
                        f"## 文档摘要: {doc_id}\n\n"
                        f"**来源**: {metadata.get('source_path', '未知')}\n"
                        f"**集合**: {coll.name}\n"
                        f"**标题**: {metadata.get('title', '无')}\n"
                        f"**摘要**: {metadata.get('summary', '无')}\n"
                        f"**标签**: {metadata.get('tags', '无')}\n"
                    )
                    return [{"type": "text", "text": markdown}]
            except Exception:
                continue

        return [{"type": "text", "text": f"未找到文档: {doc_id}"}]

    except Exception as e:
        logger.error("获取文档摘要失败: %s", e, exc_info=True)
        return [{"type": "text", "text": f"获取文档摘要失败: {e}"}]


# ============================================================
# MCP Tool 注册（async，通过 to_thread 调度同步逻辑）
# ============================================================


@server.tool()
async def query_knowledge_hub(query: str, top_k: int = 10, collection: str = ""):
    """查询知识库，返回最相关的文档片段。

    参数:
        query: 查询文本
        top_k: 返回结果数量（默认 10）
        collection: 限定检索集合（可选）
    """
    return await asyncio.to_thread(_do_query, query, top_k, collection)


@server.tool()
async def list_collections():
    """列出所有可用的知识库集合。"""
    return await asyncio.to_thread(_do_list_collections)


@server.tool()
async def get_document_summary(doc_id: str):
    """获取指定文档的摘要信息。

    参数:
        doc_id: 文档 ID
    """
    return await asyncio.to_thread(_do_get_document_summary, doc_id)


def run() -> None:
    """启动 MCP Server（Stdio Transport）。"""
    logger.info("MCP Server 启动中 (Stdio Transport)...")
    server.run(transport="stdio")


if __name__ == "__main__":
    run()
