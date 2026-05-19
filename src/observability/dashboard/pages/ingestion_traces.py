"""Ingestion 追踪页面。

展示摄取历史列表、阶段耗时瀑布图。
"""

from __future__ import annotations

from datetime import datetime
from typing import List

import streamlit as st

from src.observability.dashboard.services.trace_service import (
    TraceRecord,
    TraceService,
)


def render_ingestion_traces(trace_service: TraceService) -> None:
    """渲染 Ingestion 追踪页面。

    参数:
        trace_service: TraceService 实例
    """
    st.title("📊 Ingestion 追踪")
    st.markdown("查看摄取历史记录与各阶段耗时分布。")
    st.markdown("---")

    traces = trace_service.get_ingestion_traces()

    if not traces:
        st.info("📭 暂无摄取追踪记录，请先执行一次 ingest 操作。")
        return

    # 概览指标
    _render_summary(traces)

    st.markdown("---")

    # 追踪记录列表
    _render_trace_list(traces)


def _render_summary(traces: List[TraceRecord]) -> None:
    """渲染概览指标。

    参数:
        traces: ingestion 追踪记录列表
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("总记录数", len(traces))

    with col2:
        avg_ms = sum(t.total_elapsed_ms for t in traces) / len(traces)
        st.metric("平均耗时", f"{avg_ms:.0f}ms")

    with col3:
        if traces:
            latest = traces[0].started_datetime.strftime("%m-%d %H:%M")
            st.metric("最近摄取", latest)


def _render_trace_list(traces: List[TraceRecord]) -> None:
    """渲染追踪记录列表。

    参数:
        traces: ingestion 追踪记录列表（已按时间倒序）
    """
    st.subheader("📋 摄取历史")

    for i, trace in enumerate(traces):
        trace_id_short = trace.trace_id[:8]
        time_str = trace.started_datetime.strftime("%Y-%m-%d %H:%M:%S")
        total_ms = trace.total_elapsed_ms
        stage_count = len(trace.stages)

        # 每条记录一个可展开卡片
        with st.expander(
            f"🔄 `{trace_id_short}...` | {time_str} | "
            f"耗时 {total_ms:.0f}ms | {stage_count} 个阶段"
        ):
            _render_trace_detail(trace)


def _render_trace_detail(trace: TraceRecord) -> None:
    """渲染单条追踪的详情。

    参数:
        trace: 单条追踪记录
    """
    # 基本信息
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"**Trace ID**: `{trace.trace_id}`")
        st.caption(
            f"**开始时间**: {trace.started_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
        )
    with col2:
        st.caption(f"**总耗时**: {trace.total_elapsed_ms:.1f}ms")
        if trace.finished_at:
            finished = datetime.fromtimestamp(trace.finished_at).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            st.caption(f"**结束时间**: {finished}")

    st.markdown("---")

    # 阶段耗时瀑布图
    if trace.stages:
        st.markdown("**⏱️ 各阶段耗时分布**")
        _render_waterfall_chart(trace)
    else:
        st.info("该记录无阶段数据。")


def _render_waterfall_chart(trace: TraceRecord) -> None:
    """渲染阶段耗时瀑布图（横向条形图）。

    参数:
        trace: 单条追踪记录
    """
    # 准备数据
    stage_names = []
    stage_times = []
    stage_labels = []

    for stage in trace.stages:
        label = _stage_label(stage.name)
        stage_names.append(label)
        stage_times.append(stage.elapsed_ms)
        # 构造详细标签
        detail_parts = [f"{stage.elapsed_ms:.1f}ms"]
        if stage.method:
            detail_parts.append(stage.method)
        for key, value in stage.extra.items():
            if key not in ("action",) and value is not None:
                detail_parts.append(f"{key}={value}")
        stage_labels.append(" | ".join(detail_parts))

    if not stage_times:
        return

    # 使用 Streamlit 原生横向条形图
    import pandas as pd

    df = pd.DataFrame({
        "阶段": stage_names,
        "耗时(ms)": stage_times,
        "详情": stage_labels,
    })

    # 横向条形图
    st.bar_chart(
        df.set_index("阶段")["耗时(ms)"],
        horizontal=True,
        height=max(200, len(stage_names) * 40),
    )

    # 详情表格
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


def _stage_label(stage_name: str) -> str:
    """阶段名称中文标签。

    参数:
        stage_name: 英文阶段名

    返回:
        中文标签
    """
    labels = {
        "integrity": "完整性检查",
        "load": "加载文件",
        "split": "文本切分",
        "transform": "增强处理",
        "embed": "向量编码",
        "upsert": "存储写入",
    }
    return labels.get(stage_name, stage_name)
