"""日志工具模块。

提供统一的日志配置、获取接口以及结构化追踪日志的持久化功能。
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """JSON 格式化器，将日志记录转换为 JSON 字符串。
    
    用于生成结构化日志，便于后续解析和在 Dashboard 中展示。
    """

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录为 JSON 格式。
        
        参数:
            record: 日志记录对象
            
        返回:
            JSON 格式的日志字符串
        """
        # 如果消息已经是字典类型，直接使用
        if isinstance(record.msg, dict):
            log_data = record.msg
        else:
            # 否则构建标准日志结构
            log_data = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "name": record.name,
                "level": record.levelname,
                "message": record.getMessage(),
            }
            
            # 添加异常信息（如果有）
            if record.exc_info:
                log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """获取配置好的通用日志实例（输出到 stderr）。

    参数:
        name: 日志名称，通常为模块名
        level: 日志级别字符串

    返回:
        配置好的 logging.Logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

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


def get_trace_logger() -> logging.Logger:
    """获取专门用于追踪数据的 JSON Lines 日志实例。

    该日志会将数据追加写入 logs/traces.jsonl，每行一个完整的 JSON 对象。

    返回:
        专门用于 Trace 持久化的 Logger 实例
    """
    logger = logging.getLogger("trace_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False # 避免追踪数据污染 stderr

    if not logger.handlers:
        # 确保日志目录存在
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, "traces.jsonl")
        
        # 使用 FileHandler 进行持久化，强制 UTF-8 编码
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    return logger


def write_trace(trace_data: Dict[str, Any]) -> None:
    """将追踪数据字典写入持久化存储（JSON Lines）。

    这是 Phase F2 的核心接口，供 TraceCollector 调用。

    参数:
        trace_data: 序列化后的追踪数据字典
    """
    try:
        logger = get_trace_logger()
        # 将整个字典作为消息发送给 JSONFormatter
        logger.info(trace_data)
    except Exception as e:
        # 记录到 stderr 的通用日志中
        base_logger = get_logger(__name__)
        base_logger.error("写入 Trace 数据失败: %s", e)
