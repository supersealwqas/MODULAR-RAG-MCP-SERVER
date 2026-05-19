"""数据浏览器页面。

展示已摄入的文档列表、Chunk 详情与图片预览。
支持按集合筛选文档，点击展开查看 chunk 内容与 metadata。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from src.observability.dashboard.services.data_service import DataService


def render_data_browser(data_service: DataService) -> None:
    """渲染数据浏览器页面。

    参数:
        data_service: DataService 实例
    """
    st.title("📂 数据浏览器")
    st.markdown("浏览已摄入的文档、Chunk 详情与关联图片。")
    st.markdown("---")

    # ----------------------------------------------------------
    # 集合筛选与统计
    # ----------------------------------------------------------
    stats_col, filter_col = st.columns([2, 1])

    with stats_col:
        _render_collection_overview(data_service)

    with filter_col:
        selected_collection = _render_collection_filter(data_service)

    st.markdown("---")

    # ----------------------------------------------------------
    # 文档列表
    # ----------------------------------------------------------
    _render_document_list(data_service, selected_collection)


def _render_collection_overview(data_service: DataService) -> None:
    """渲染集合总览指标卡片。

    参数:
        data_service: DataService 实例
    """
    st.subheader("📊 数据总览")

    try:
        stats = data_service.get_collection_stats()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("文档数", stats.document_count)
        col2.metric("Chunk 数", stats.chunk_count)
        col3.metric("图片数", stats.image_count)
        col4.metric(
            "文件大小",
            _format_file_size(stats.total_file_size),
        )
    except Exception as e:
        st.warning(f"加载统计信息失败: {e}")


def _render_collection_filter(data_service: DataService) -> Optional[str]:
    """渲染集合筛选下拉框。

    参数:
        data_service: DataService 实例

    返回:
        选中的集合名称（None 表示全部）
    """
    st.subheader("🔍 集合筛选")

    try:
        collections = data_service.list_collections()
    except Exception:
        collections = []

    options = ["全部"] + collections
    selected = st.selectbox(
        "选择集合",
        options=options,
        label_visibility="collapsed",
    )

    return None if selected == "全部" else selected


def _render_document_list(
    data_service: DataService,
    collection: Optional[str],
) -> None:
    """渲染文档列表与详情展开。

    参数:
        data_service: DataService 实例
        collection: 选中的集合名称
    """
    st.subheader("📄 文档列表")

    try:
        documents = data_service.list_documents(collection=collection)
    except Exception as e:
        st.error(f"加载文档列表失败: {e}")
        return

    if not documents:
        st.info("📭 暂无已摄入的文档。请先运行 `ingest.py` 摄取数据。")
        return

    # 显示文档数量
    st.caption(f"共 {len(documents)} 个文档")

    for doc in documents:
        with st.expander(
            f"📄 {_shorten_path(doc.source_path)} "
            f"| {doc.chunk_count} chunks "
            f"| {doc.image_count} images "
            f"| {doc.collection}",
        ):
            # 文档摘要信息
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.markdown(f"**源文件**: `{doc.source_path}`")
                st.markdown(f"**集合**: {doc.collection}")
                st.markdown(f"**文件大小**: {_format_file_size(doc.file_size)}")
            with info_col2:
                st.markdown(f"**Chunk 数**: {doc.chunk_count}")
                st.markdown(f"**图片数**: {doc.image_count}")
                st.markdown(f"**处理时间**: {doc.processed_at}")

            # 加载详情
            detail = data_service.get_document_detail(doc.source_path)
            if detail is None:
                st.warning("无法加载文档详情")
                continue

            # Chunk 详情
            if detail.chunks:
                st.markdown("---")
                st.markdown(f"**Chunk 列表** ({len(detail.chunks)} 条)")
                _render_chunks(detail.chunks)

            # 图片预览
            if detail.images:
                st.markdown("---")
                st.markdown(f"**关联图片** ({len(detail.images)} 张)")
                _render_images(detail.images)


def _render_chunks(chunks: List[Dict[str, Any]]) -> None:
    """渲染 chunk 详情列表。

    参数:
        chunks: chunk 字典列表（包含 id、text、metadata）
    """
    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get("id", "unknown")
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})

        with st.container(border=True):
            st.markdown(f"**Chunk {i + 1}** (`{chunk_id}`)")
            st.text(text[:500] + ("..." if len(text) > 500 else ""))

            # 显示关键 metadata
            if metadata:
                meta_items = []
                for key in ["chunk_index", "source_ref", "doc_type"]:
                    if key in metadata:
                        meta_items.append(f"**{key}**: `{metadata[key]}`")
                if meta_items:
                    st.markdown(" | ".join(meta_items))

                # 可展开完整 metadata
                with st.expander("完整 Metadata"):
                    st.json(metadata)


def _render_images(images: List[Dict[str, Any]]) -> None:
    """渲染图片预览列表。

    参数:
        images: 图片信息列表
    """
    cols = st.columns(min(len(images), 4))
    for i, img in enumerate(images):
        col = cols[i % len(cols)]
        with col:
            image_id = img.get("image_id", "unknown")
            file_path = img.get("file_path", "")
            page_num = img.get("page_num", 0)

            st.markdown(f"**{image_id}**")
            st.caption(f"页码: {page_num}")

            # 尝试显示图片
            if file_path:
                try:
                    st.image(file_path, use_container_width=True)
                except Exception:
                    st.caption("图片加载失败")


# ============================================================
# 工具函数
# ============================================================


def _shorten_path(path: str, max_len: int = 60) -> str:
    """缩短文件路径用于显示。

    参数:
        path: 完整路径
        max_len: 最大显示长度

    返回:
        缩短后的路径
    """
    if len(path) <= max_len:
        return path
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 3:
        return f".../{parts[-2]}/{parts[-1]}"
    return "..." + path[-(max_len - 3):]


def _format_file_size(size_bytes: int) -> str:
    """格式化文件大小。

    参数:
        size_bytes: 字节数

    返回:
        人类可读的文件大小
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
