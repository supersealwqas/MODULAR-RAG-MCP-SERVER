"""MCP Server 入口模块。

启动时加载配置并初始化日志系统。
"""
from __future__ import annotations

import sys


def main() -> int:
    """主入口函数。

    加载配置文件，初始化日志，输出启动信息。

    返回:
        退出码，0 表示成功，1 表示失败
    """
    try:
        from src.core.settings import load_settings
        from src.observability.logger import get_logger

        settings = load_settings()
        logger = get_logger(__name__, settings.observability.log_level)
        logger.info(
            "配置加载成功: LLM=%s/%s, Embedding=%s/%s",
            settings.llm.provider,
            settings.llm.model,
            settings.embedding.provider,
            settings.embedding.model,
        )
    except Exception as e:
        print(f"启动失败: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
