"""追踪日志服务。

读取 traces.jsonl，解析为结构化 Trace 列表，供 Dashboard 追踪页面使用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TraceStage:
    """单个阶段的数据。"""

    name: str
    elapsed_ms: float = 0.0
    method: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceStage":
        """从字典构造。"""
        name = data.get("name", "unknown")
        elapsed_ms = data.get("elapsed_ms", 0.0)
        method = data.get("method", "")
        # 其余字段放入 extra
        extra = {
            k: v
            for k, v in data.items()
            if k not in ("name", "elapsed_ms", "method", "timestamp")
        }
        return cls(name=name, elapsed_ms=elapsed_ms, method=method, extra=extra)


@dataclass
class TraceRecord:
    """一条完整的追踪记录。"""

    trace_id: str
    trace_type: str
    started_at: float
    finished_at: Optional[float]
    total_elapsed_ms: float
    stages: List[TraceStage]
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceRecord":
        """从字典构造。"""
        stages = [
            TraceStage.from_dict(s) for s in data.get("stages", [])
        ]
        return cls(
            trace_id=data.get("trace_id", ""),
            trace_type=data.get("trace_type", "unknown"),
            started_at=data.get("started_at", 0.0),
            finished_at=data.get("finished_at"),
            total_elapsed_ms=data.get("total_elapsed_ms", 0.0),
            stages=stages,
            raw=data,
        )

    @property
    def started_datetime(self) -> datetime:
        """开始时间的 datetime 对象。"""
        return datetime.fromtimestamp(self.started_at)

    @property
    def stage_map(self) -> Dict[str, TraceStage]:
        """阶段名称到阶段数据的映射。"""
        return {s.name: s for s in self.stages}


class TraceService:
    """追踪日志服务。

    读取 traces.jsonl 文件，解析并提供查询接口。
    """

    def __init__(self, traces_file: str = "logs/traces.jsonl") -> None:
        """初始化。

        参数:
            traces_file: traces.jsonl 文件路径
        """
        self._traces_file = traces_file
        self._cache: Optional[List[TraceRecord]] = None

    def _load_traces(self) -> List[TraceRecord]:
        """加载所有 trace 记录（带缓存）。"""
        if self._cache is not None:
            return self._cache

        traces_path = Path(self._traces_file)
        if not traces_path.exists():
            self._cache = []
            return self._cache

        records = []
        try:
            lines = traces_path.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(TraceRecord.from_dict(data))
                except (json.JSONDecodeError, KeyError):
                    continue
        except Exception:
            self._cache = []
            return self._cache

        self._cache = records
        return self._cache

    def get_ingestion_traces(self) -> List[TraceRecord]:
        """获取所有 ingestion 类型的追踪记录（按时间倒序）。"""
        traces = self._load_traces()
        ingestion = [t for t in traces if t.trace_type == "ingestion"]
        ingestion.sort(key=lambda t: t.started_at, reverse=True)
        return ingestion

    def get_query_traces(self) -> List[TraceRecord]:
        """获取所有 query 类型的追踪记录（按时间倒序）。"""
        traces = self._load_traces()
        query = [t for t in traces if t.trace_type == "query"]
        query.sort(key=lambda t: t.started_at, reverse=True)
        return query

    def get_trace_by_id(self, trace_id: str) -> Optional[TraceRecord]:
        """按 trace_id 查找记录。"""
        traces = self._load_traces()
        for t in traces:
            if t.trace_id == trace_id:
                return t
        return None

    def get_stats(self) -> Dict[str, int]:
        """获取追踪统计。"""
        traces = self._load_traces()
        return {
            "total": len(traces),
            "ingestion": sum(1 for t in traces if t.trace_type == "ingestion"),
            "query": sum(1 for t in traces if t.trace_type == "query"),
        }

    def invalidate_cache(self) -> None:
        """清除缓存，下次访问重新读取文件。"""
        self._cache = None
