"""Query 追踪页面。

展示查询历史列表、各阶段耗时对比、Dense vs Sparse 对比、Rerank 前后变化。
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import streamlit as st

from src.observability.dashboard.services.trace_service import (
    TraceRecord,
    TraceService,
)


def render_query_traces(trace_service: TraceService) -> None:
    """渲染 Query 追踪页面。

    参数:
        trace_service: TraceService 实例
    """
    st.title("🔍 Query 追踪")
    st.markdown("查看查询历史与各阶段检索对比。")
    st.markdown("---")

    traces = trace_service.get_query_traces()

    if not traces:
        st.info("📭 暂无查询追踪记录，请先执行一次 query 操作。")
        return

    # 概览指标
    _render_summary(traces)

    st.markdown("---")

    # 搜索过滤
    search_keyword = st.text_input(
        "🔎 搜索查询关键词",
        placeholder="输入关键词过滤...",
        help="按查询内容或 trace_id 搜索",
    )

    # 过滤
    if search_keyword:
        traces = _filter_traces(traces, search_keyword)
        if not traces:
            st.warning(f"未找到匹配「{search_keyword}」的记录。")
            return

    # 追踪记录列表
    _render_trace_list(traces)


def _render_summary(traces: List[TraceRecord]) -> None:
    """渲染概览指标。

    参数:
        traces: query 追踪记录列表
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("总查询数", len(traces))

    with col2:
        avg_ms = sum(t.total_elapsed_ms for t in traces) / len(traces)
        st.metric("平均耗时", f"{avg_ms:.0f}ms")

    with col3:
        if traces:
            latest = traces[0].started_datetime.strftime("%m-%d %H:%M")
            st.metric("最近查询", latest)


def _filter_traces(
    traces: List[TraceRecord], keyword: str
) -> List[TraceRecord]:
    """按关键词过滤追踪记录。

    参数:
        traces: 追踪记录列表
        keyword: 搜索关键词

    返回:
        过滤后的记录列表
    """
    keyword_lower = keyword.lower()
    result = []
    for t in traces:
        # 搜索 trace_id
        if keyword_lower in t.trace_id.lower():
            result.append(t)
            continue
        # 搜索 stages 中的 original_query
        for stage in t.stages:
            if stage.name == "query_processing":
                query_text = stage.extra.get("original_query", "")
                if keyword_lower in query_text.lower():
                    result.append(t)
                    break
    return result


def _render_trace_list(traces: List[TraceRecord]) -> None:
    """渲染追踪记录列表。

    参数:
        traces: query 追踪记录列表（已按时间倒序）
    """
    st.subheader("📋 查询历史")

    for i, trace in enumerate(traces):
        trace_id_short = trace.trace_id[:8]
        time_str = trace.started_datetime.strftime("%Y-%m-%d %H:%M:%S")
        total_ms = trace.total_elapsed_ms
        stage_count = len(trace.stages)

        # 从 query_processing 阶段提取查询文本
        query_text = _extract_query_text(trace)

        label = f"`{trace_id_short}...` | {time_str} | 耗时 {total_ms:.0f}ms"
        if query_text:
            label = f"🔎 {query_text[:40]} | {label}"

        with st.expander(label):
            _render_trace_detail(trace)


def _extract_query_text(trace: TraceRecord) -> str:
    """从追踪记录中提取查询文本。

    参数:
        trace: 追踪记录

    返回:
        查询文本（可能为空）
    """
    for stage in trace.stages:
        if stage.name == "query_processing":
            return stage.extra.get("original_query", "")
    return ""


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

    # 查询关键词
    query_stage = trace.stage_map.get("query_processing")
    if query_stage:
        original_query = query_stage.extra.get("original_query", "")
        keywords = query_stage.extra.get("keywords", [])
        if original_query:
            st.markdown(f"**查询**: {original_query}")
        if keywords:
            st.caption(f"**关键词**: {', '.join(keywords)}")

    st.markdown("---")

    # 耗时瀑布图
    if trace.stages:
        st.markdown("**⏱️ 各阶段耗时分布**")
        _render_waterfall_chart(trace)

    st.markdown("---")

    # Dense vs Sparse 对比
    _render_retrieval_comparison(trace)

    # Rerank 前后对比
    _render_rerank_comparison(trace)


def _render_waterfall_chart(trace: TraceRecord) -> None:
    """渲染阶段耗时瀑布图。

    参数:
        trace: 单条追踪记录
    """
    import pandas as pd

    stage_names = []
    stage_times = []
    stage_labels = []

    for stage in trace.stages:
        label = _stage_label(stage.name)
        stage_names.append(label)
        stage_times.append(stage.elapsed_ms)
        detail_parts = [f"{stage.elapsed_ms:.1f}ms"]
        if stage.method:
            detail_parts.append(stage.method)
        for key, value in stage.extra.items():
            if key not in ("original_query", "keywords") and value is not None:
                detail_parts.append(f"{key}={value}")
        stage_labels.append(" | ".join(detail_parts))

    if not stage_times:
        return

    df = pd.DataFrame({
        "阶段": stage_names,
        "耗时(ms)": stage_times,
        "详情": stage_labels,
    })

    st.bar_chart(
        df.set_index("阶段")["耗时(ms)"],
        horizontal=True,
        height=max(200, len(stage_names) * 40),
    )

    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_retrieval_comparison(trace: TraceRecord) -> None:
    """渲染 Dense vs Sparse 并列对比。

    参数:
        trace: 单条追踪记录
    """
    dense_stage = trace.stage_map.get("dense_retrieval")
    sparse_stage = trace.stage_map.get("sparse_retrieval")

    if not dense_stage and not sparse_stage:
        return

    st.markdown("**🔀 Dense vs Sparse 检索对比**")

    col_dense, col_sparse = st.columns(2)

    with col_dense:
        st.markdown("**🟢 Dense Retrieval**")
        if dense_stage:
            st.metric("耗时", f"{dense_stage.elapsed_ms:.1f}ms")
            st.metric("结果数", dense_stage.extra.get("result_count", "?"))
            st.caption(f"方法: {dense_stage.method}")
            if dense_stage.extra.get("query_length"):
                st.caption(f"查询长度: {dense_stage.extra['query_length']}")
        else:
            st.info("无 Dense 阶段数据")

    with col_sparse:
        st.markdown("**🔵 Sparse Retrieval**")
        if sparse_stage:
            st.metric("耗时", f"{sparse_stage.elapsed_ms:.1f}ms")
            st.metric("结果数", sparse_stage.extra.get("result_count", "?"))
            st.caption(f"方法: {sparse_stage.method}")
            if sparse_stage.extra.get("keyword_count"):
                st.caption(f"关键词数: {sparse_stage.extra['keyword_count']}")
        else:
            st.info("无 Sparse 阶段数据")

    # Fusion 阶段
    fusion_stage = trace.stage_map.get("fusion")
    if fusion_stage:
        st.markdown("**🔗 RRF 融合结果**")
        st.metric("融合后结果数", fusion_stage.extra.get("result_count", "?"))
        st.caption(f"k={fusion_stage.extra.get('k', '?')}, 耗时 {fusion_stage.elapsed_ms:.1f}ms")


def _render_rerank_comparison(trace: TraceRecord) -> None:
    """渲染 Rerank 前后排名变化。

    参数:
        trace: 单条追踪记录
    """
    rerank_stage = trace.stage_map.get("reranker")
    if not rerank_stage:
        return

    st.markdown("**📊 Rerank 变化**")

    col_before, col_after = st.columns(2)

    with col_before:
        st.metric("Rerank 前结果数", rerank_stage.extra.get("input_count", "?"))

    with col_after:
        st.metric("Rerank 后结果数", rerank_stage.extra.get("output_count", "?"))

    is_fallback = rerank_stage.extra.get("fallback", False)
    if is_fallback:
        st.warning("⚠️ Reranker 使用了回退路径（原始排序未改变）")
    else:
        st.caption(f"方法: {rerank_stage.method}, 耗时 {rerank_stage.elapsed_ms:.1f}ms")


def _stage_label(stage_name: str) -> str:
    """阶段名称中文标签。

    参数:
        stage_name: 英文阶段名

    返回:
        中文标签
    """
    labels = {
        "query_processing": "查询处理",
        "dense_retrieval": "Dense 检索",
        "sparse_retrieval": "Sparse 检索",
        "fusion": "RRF 融合",
        "hybrid_search": "混合检索",
        "reranker": "Rerank 重排",
    }
    return labels.get(stage_name, stage_name)
