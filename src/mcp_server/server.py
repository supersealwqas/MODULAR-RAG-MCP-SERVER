"""
MCP Server 入口模块

遵循 MCP 规范，通过 Stdio Transport 与客户端通信。
核心约束：stdout 仅输出 MCP 消息，日志统一输出至 stderr。
"""

import io
import logging
import sys

from mcp.server.fastmcp import FastMCP

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
async def query_knowledge_hub(query: str, top_k: int = 10) -> str:
    """查询知识库，返回最相关的文档片段。"""
    logger.info("query_knowledge_hub 被调用: query=%s, top_k=%d", query, top_k)
    # TODO: E3 阶段将接入 HybridSearch + Reranker
    return f"查询结果占位符: query='{query}', top_k={top_k}"


@server.tool()
async def list_collections() -> str:
    """列出所有可用的知识库集合。"""
    logger.info("list_collections 被调用")
    # TODO: E4 阶段将实现集合列表
    return "集合列表占位符"


@server.tool()
async def get_document_summary(doc_id: str) -> str:
    """获取指定文档的摘要信息。"""
    logger.info("get_document_summary 被调用: doc_id=%s", doc_id)
    # TODO: E5 阶段将实现文档摘要
    return f"文档摘要占位符: doc_id='{doc_id}'"


def run() -> None:
    """启动 MCP Server（Stdio Transport）。"""
    logger.info("MCP Server 启动中 (Stdio Transport)...")
    server.run(transport="stdio")


if __name__ == "__main__":
    run()
