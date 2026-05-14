"""日志工具模块。

提供统一的日志配置和获取接口。
"""

from __future__ import annotations

import logging
import sys


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """获取配置好的日志实例。

    日志输出到 stderr，格式包含时间戳、模块名、级别和消息。

    参数:
        name: 日志名称，通常为模块名（如 __name__）
        level: 日志级别字符串，可选值: DEBUG、INFO、WARNING、ERROR、CRITICAL

    返回:
        配置好的 logging.Logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 避免重复添加 handler
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)

    return logger
