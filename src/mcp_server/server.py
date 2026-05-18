"""
MCP Server 入口模块

遵循 MCP 规范，通过 Stdio Transport 与客户端通信。
核心约束：stdout 仅输出 MCP 消息，日志统一输出至 stderr。
"""

import io
import json
import logging
import sys

from mcp.server.fastmcp import FastMCP

from src.mcp_server.tools.query_knowledge_hub import query_knowledge_hub as _query_knowledge_hub
from src.mcp_server.tools.list_collections import list_collections as _list_collections
from src.mcp_server.tools.get_document_summary import get_document_summary as _get_document_summary

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


@server.tool()
async def query_knowledge_hub(query: str, top_k: int = 10, collection: str = "") -> str:
    """查询知识库，返回最相关的文档片段。

    参数:
        query: 查询文本
        top_k: 返回结果数量（默认 10）
        collection: 限定检索集合（可选）
    """
    logger.info("query_knowledge_hub 被调用: query=%s, top_k=%d, collection=%s", query, top_k, collection)

    # 调用实际实现
    result = await _query_knowledge_hub(
        query=query,
        top_k=top_k,
        collection=collection if collection else None,
    )

    # 返回 Markdown 文本（content[0].text）
    if result.get("content") and len(result["content"]) > 0:
        return result["content"][0]["text"]

    return "查询未返回结果"


@server.tool()
async def list_collections() -> str:
    """列出所有可用的知识库集合。"""
    logger.info("list_collections 被调用")

    # 调用实际实现
    result = await _list_collections()

    # 返回 Markdown 文本（content[0].text）
    if result.get("content") and len(result["content"]) > 0:
        return result["content"][0]["text"]

    return "获取集合列表失败"


@server.tool()
async def get_document_summary(doc_id: str) -> str:
    """获取指定文档的摘要信息。

    参数:
        doc_id: 文档 ID
    """
    logger.info("get_document_summary 被调用: doc_id=%s", doc_id)

    # 调用实际实现
    result = await _get_document_summary(doc_id=doc_id)

    # 返回 Markdown 文本（content[0].text）
    if result.get("content") and len(result["content"]) > 0:
        return result["content"][0]["text"]

    return "获取文档摘要失败"


def run() -> None:
    """启动 MCP Server（Stdio Transport）。"""
    logger.info("MCP Server 启动中 (Stdio Transport)...")
    server.run(transport="stdio")


if __name__ == "__main__":
    run()
