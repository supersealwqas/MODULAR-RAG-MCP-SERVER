"""TraceCollector 模块。

负责收集 TraceContext 并触发持久化。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.trace.trace_context import TraceContext


class TraceCollector:
    """追踪收集器，负责将完成的追踪数据移交给持久化层。
    
    它是连接追踪数据生成（TraceContext）与持久化存储（JSON Lines）的桥梁。
    """

    def collect(self, trace: TraceContext) -> None:
        """收集追踪数据并持久化。

        参数:
            trace: 已完成的追踪上下文实例
        """
        # 1. 确保 trace 已结束并完成耗时统计
        trace.finish()
        
        # 2. 获取序列化字典数据
        trace_data = trace.to_dict()
        
        # 3. 触发持久化
        try:
            # 引入通用日志以便记录操作日志
            from src.observability.logger import get_logger, write_trace
            
            logger = get_logger(__name__)
            logger.debug(f"正在收集追踪数据: {trace_data['trace_id']} ({trace_data['trace_type']})")
            
            # 调用 Phase F2 实现的 JSON Lines 写入接口
            write_trace(trace_data)
            
        except Exception as e:
            # 收集过程不应阻塞主业务流程
            print(f"[警告] Trace 收集失败: {e}")
