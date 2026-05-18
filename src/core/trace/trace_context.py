"""TraceContext 最小实现模块。

提供 trace_id 生成和阶段数据记录功能。
Phase F 会增强此模块（finish、耗时统计、trace_type、to_dict）。
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional


class TraceContext:
    """追踪上下文，记录一次操作的各阶段数据。

    属性:
        trace_id: 全局唯一追踪 ID
        trace_type: 追踪类型 ("query" 或 "ingestion")
        stages: 阶段数据列表
    """

    def __init__(self, trace_type: str = "query") -> None:
        """初始化追踪上下文。

        参数:
            trace_type: 追踪类型（"query" 或 "ingestion"）
        """
        self.trace_id: str = uuid.uuid4().hex[:16]
        self.trace_type: str = trace_type
        self.stages: List[Dict[str, Any]] = []
        self._started_at: float = time.time()
        self._finished_at: Optional[float] = None

    def record_stage(self, name: str, **kwargs: Any) -> None:
        """记录一个阶段的数据。

        参数:
            name: 阶段名称（如 "load"、"split"、"refine"）
            **kwargs: 阶段相关数据（method、elapsed_ms、details 等）
        """
        stage = {"name": name, "timestamp": time.time()}
        stage.update(kwargs)
        self.stages.append(stage)

    def finish(self) -> None:
        """结束追踪并记录结束时间。"""
        self._finished_at = time.time()

    def elapsed_ms(self, stage_name: Optional[str] = None) -> float:
        """获取指定阶段或总的耗时（毫秒）。

        参数:
            stage_name: 阶段名称。如果为 None，则返回从开始到结束的总耗时。
        """
        if stage_name:
            # 查找阶段记录
            for stage in self.stages:
                if stage.get("name") == stage_name and "elapsed_ms" in stage:
                    return float(stage["elapsed_ms"])
            return 0.0

        # 计算总耗时
        end_time = self._finished_at or time.time()
        return (end_time - self._started_at) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self._started_at,
            "finished_at": self._finished_at,
            "total_elapsed_ms": self.elapsed_ms(),
            "stages": self.stages,
        }
