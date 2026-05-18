"""TraceCollector 模块。

负责收集 TraceContext 并触发持久化。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.trace.trace_context import TraceContext


class TraceCollector:
    """追踪收集器，负责将完成的追踪数据移交给持久化层。"""

    def collect(self, trace: TraceContext) -> None:
        """收集追踪数据。

        参数:
            trace: 已完成的追踪上下文
        """
        # 确保 trace 已完成
        trace.finish()
        
        # 获取序列化数据
        trace_data = trace.to_dict()
        
        # 触发持久化（Phase F2 会实现具体的写入逻辑）
        # 目前我们先通过日志系统或简单的打印来模拟，
        # 等待 F2 实现 write_trace 后再进行对接。
        try:
            from src.observability.logger import get_logger
            logger = get_logger(__name__)
            logger.debug("Trace collected: %s", trace_data["trace_id"])
            
            # 尝试调用 F2 预期的持久化函数（如果已存在）
            try:
                from src.observability.logger import write_trace
                write_trace(trace_data)
            except ImportError:
                # F2 尚未实现持久化函数
                pass
        except ImportError:
            # 基础日志系统不可用
            pass
