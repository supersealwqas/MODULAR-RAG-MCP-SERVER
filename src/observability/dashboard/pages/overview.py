"""系统总览页面。

展示组件配置卡片、向量库数据统计、追踪日志概览。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from src.observability.dashboard.services.config_service import ConfigService


def _load_collection_stats(vector_store_config) -> List[Dict[str, Any]]:
    """加载 ChromaDB collection 统计信息。

    参数:
        vector_store_config: VectorStoreConfig 对象

    返回:
        collection 统计列表
    """
    try:
        from src.libs.vector_store.chroma_store import ChromaStore
        store = ChromaStore(
            collection_name="__list_placeholder__",
            persist_directory=vector_store_config.persist_directory,
        )
        return store.list_collections()
    except Exception as e:
        return [{"name": "(加载失败)", "count": 0, "error": str(e)}]


def _load_trace_stats(traces_file: str) -> Dict[str, Any]:
    """加载追踪日志统计信息。

    参数:
        traces_file: traces.jsonl 文件路径

    返回:
        统计信息字典
    """
    stats = {
        "total": 0,
        "query_count": 0,
        "ingestion_count": 0,
        "recent_traces": [],
    }

    traces_path = Path(traces_file)
    if not traces_path.exists():
        return stats

    try:
        lines = traces_path.read_text(encoding="utf-8").strip().splitlines()
        stats["total"] = len(lines)

        recent_lines = lines[-5:]  # 最近 5 条
        for line in reversed(recent_lines):
            try:
                trace = json.loads(line)
                trace_type = trace.get("trace_type", "unknown")
                if trace_type == "query":
                    stats["query_count"] += 1
                elif trace_type == "ingestion":
                    stats["ingestion_count"] += 1
                stats["recent_traces"].append(trace)
            except json.JSONDecodeError:
                continue

        # 统计全部（不只是最近 5 条）
        stats["query_count"] = sum(
            1 for line in lines
            if '"query"' in line and '"trace_type"' in line
        )
        stats["ingestion_count"] = sum(
            1 for line in lines
            if '"ingestion"' in line and '"trace_type"' in line
        )
    except Exception:
        pass

    return stats


def render_overview(config_service: ConfigService) -> None:
    """渲染系统总览页面。

    参数:
        config_service: 配置服务实例
    """
    st.title("📊 系统总览")
    st.markdown("---")

    # === 组件配置卡片 ===
    st.subheader("⚙️ 组件配置")
    cards = config_service.get_component_cards()

    # 每行 3 个卡片
    cols_per_row = 3
    for i in range(0, len(cards), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(cards):
                break
            card = cards[idx]
            with col:
                with st.container(border=True):
                    st.markdown(f"**{card['icon']} {card['title']}**")
                    for key, value in card["items"].items():
                        st.caption(f"`{key}`: {value}")

    st.markdown("---")

    # === 数据统计 ===
    st.subheader("📦 数据统计")
    col_vec, col_trace = st.columns(2)

    with col_vec:
        with st.container(border=True):
            st.markdown("**🗄️ 向量库统计**")
            settings = config_service.get_settings()
            collections = _load_collection_stats(settings.vector_store)
            if collections:
                for col_info in collections:
                    name = col_info.get("name", "?")
                    count = col_info.get("count", 0)
                    st.metric(
                        label=f"Collection: {name}",
                        value=f"{count} 条记录",
                    )
            else:
                st.info("暂无 collection 数据")

    with col_trace:
        with st.container(border=True):
            st.markdown("**📈 追踪日志统计**")
            settings = config_service.get_settings()
            trace_stats = _load_trace_stats(settings.observability.traces_file)
            st.metric("总追踪数", trace_stats["total"])
            st.metric("查询追踪", trace_stats["query_count"])
            st.metric("摄取追踪", trace_stats["ingestion_count"])

    st.markdown("---")

    # === 最近追踪记录 ===
    st.subheader("🕐 最近追踪记录")
    settings = config_service.get_settings()
    trace_stats = _load_trace_stats(settings.observability.traces_file)

    if trace_stats["recent_traces"]:
        for trace in trace_stats["recent_traces"]:
            trace_type = trace.get("trace_type", "unknown")
            trace_id = trace.get("trace_id", "?")[:8]
            total_ms = trace.get("total_elapsed_ms", 0)
            started = trace.get("started_at", "?")
            icon = "🔍" if trace_type == "query" else "📥"
            st.caption(
                f"{icon} `{trace_id}...` | {trace_type} | "
                f"耗时 {total_ms:.0f}ms | {started}"
            )
    else:
        st.info("暂无追踪记录，请先执行 ingest 或 query 操作")

    # === 配置文件原始内容（可折叠）===
    with st.expander("📄 查看原始配置（脱敏）"):
        raw_config = config_service.get_raw_config_dict()
        st.json(raw_config)
