"""Dashboard 多页面应用入口。

使用 Streamlit st.navigation() 注册六个页面：
1. 系统总览 — 组件配置与数据统计
2. 数据浏览器 — 文档/Chunk/图片浏览（G3）
3. Ingestion 管理 — 文件上传与摄取（G4）
4. Ingestion 追踪 — 摄取历史与耗时瀑布图（G5）
5. Query 追踪 — 查询历史与阶段对比（G6）
6. 评估面板 — 评估运行与指标（H4）

使用方式:
    streamlit run src/observability/dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.observability.dashboard.services.config_service import ConfigService
from src.observability.dashboard.services.data_service import DataService
from src.observability.dashboard.services.trace_service import TraceService


def _get_config_service() -> ConfigService:
    """获取或创建 ConfigService 单例（缓存在 session_state）。

    返回:
        ConfigService 实例
    """
    if "config_service" not in st.session_state:
        st.session_state.config_service = ConfigService()
    return st.session_state.config_service


def _get_data_service() -> DataService:
    """获取或创建 DataService 单例（缓存在 session_state）。

    返回:
        DataService 实例
    """
    if "data_service" not in st.session_state:
        st.session_state.data_service = DataService()
    return st.session_state.data_service


def _get_trace_service() -> TraceService:
    """获取或创建 TraceService 单例（缓存在 session_state）。

    返回:
        TraceService 实例
    """
    if "trace_service" not in st.session_state:
        st.session_state.trace_service = TraceService()
    return st.session_state.trace_service


def _placeholder_page(title: str, planned_task: str) -> None:
    """渲染占位页面（尚未实现的功能）。

    参数:
        title: 页面标题
        planned_task: 计划实现的任务编号
    """
    st.title(title)
    st.info(f"🚧 此页面尚未实现，计划在 {planned_task} 中完成。")


# === 页面定义 ===

def page_overview():
    """系统总览页面。"""
    from src.observability.dashboard.pages.overview import render_overview
    render_overview(_get_config_service())


def page_data_browser():
    """数据浏览器页面。"""
    from src.observability.dashboard.pages.data_browser import render_data_browser
    render_data_browser(_get_data_service())


def page_ingestion_manager():
    """Ingestion 管理页面。"""
    from src.observability.dashboard.pages.ingestion_manager import render_ingestion_manager
    render_ingestion_manager(_get_data_service())


def page_ingestion_traces():
    """Ingestion 追踪页面。"""
    from src.observability.dashboard.pages.ingestion_traces import render_ingestion_traces
    render_ingestion_traces(_get_trace_service())


def page_query_traces():
    """Query 追踪页面（G6 占位）。"""
    _placeholder_page("🔍 Query 追踪", "G6")


def page_evaluation():
    """评估面板页面（H4 占位）。"""
    _placeholder_page("📈 评估面板", "H4")


# === 导航配置 ===

pages = [
    st.Page(page_overview, title="系统总览", icon="📊", url_path="overview"),
    st.Page(page_data_browser, title="数据浏览器", icon="📂", url_path="data-browser"),
    st.Page(page_ingestion_manager, title="Ingestion 管理", icon="📥", url_path="ingestion"),
    st.Page(page_ingestion_traces, title="Ingestion 追踪", icon="📊", url_path="ingestion-traces"),
    st.Page(page_query_traces, title="Query 追踪", icon="🔍", url_path="query-traces"),
    st.Page(page_evaluation, title="评估面板", icon="📈", url_path="evaluation"),
]

# === 应用入口 ===

def main():
    """Dashboard 主入口函数。"""
    st.set_page_config(
        page_title="RAG Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 侧边栏标题
    with st.sidebar:
        st.title("📊 RAG Dashboard")
        st.caption("Modular RAG MCP Server")
        st.markdown("---")

    # 注册导航
    nav = st.navigation(pages, position="sidebar")
    nav.run()


if __name__ == "__main__":
    main()
