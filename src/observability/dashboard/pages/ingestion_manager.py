"""Ingestion 管理页面。

支持文件上传触发摄取、实时进度展示、已有文档删除。
"""

from __future__ import annotations

import os
from typing import Optional

import streamlit as st

from src.observability.dashboard.services.data_service import DataService


def render_ingestion_manager(data_service: DataService) -> None:
    """渲染 Ingestion 管理页面。

    参数:
        data_service: DataService 实例
    """
    st.title("📥 Ingestion 管理")
    st.markdown("上传文件触发摄取、查看进度、删除已有文档。")
    st.markdown("---")

    # ----------------------------------------------------------
    # 文件上传与摄取
    # ----------------------------------------------------------
    _render_upload_section()

    st.markdown("---")

    # ----------------------------------------------------------
    # 文档删除
    # ----------------------------------------------------------
    _render_delete_section(data_service)


def _render_upload_section() -> None:
    """渲染文件上传与摄取区域。"""
    st.subheader("📤 上传并摄取")

    col1, col2 = st.columns([3, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "选择 PDF 文件",
            type=["pdf"],
            help="上传后将自动摄取到向量库和 BM25 索引",
        )

    with col2:
        collection = st.text_input(
            "集合名称",
            value="default",
            help="文档所属集合",
        )

    if uploaded_file is None:
        st.info("请先选择一个 PDF 文件。")
        return

    st.markdown(f"**文件**: {uploaded_file.name} ({_format_size(uploaded_file.size)})")

    if st.button("🚀 开始摄取", type="primary", use_container_width=True):
        _run_ingestion(uploaded_file, collection)


def _run_ingestion(uploaded_file, collection: str) -> None:
    """执行摄取流程（带进度条）。

    参数:
        uploaded_file: Streamlit UploadedFile 对象
        collection: 集合名称
    """
    # 将上传文件保存到 data/documents/{collection}/ 目录
    save_dir = os.path.join("data", "documents", collection)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, uploaded_file.name)

    try:
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # 创建进度条
        progress_bar = st.progress(0, text="准备摄取...")
        status_text = st.empty()

        # 进度回调
        def on_progress(stage_name: str, current: int, total: int) -> None:
            progress = current / total
            progress_bar.progress(
                progress,
                text=f"阶段 {current}/{total}: {_stage_label(stage_name)}",
            )
            status_text.text(f"正在处理: {_stage_label(stage_name)}...")

        # 执行 Pipeline
        from src.core.settings import load_settings
        from src.ingestion.pipeline import IngestionPipeline

        settings = load_settings()
        pipeline = IngestionPipeline(settings)
        result = pipeline.run(
            file_path=save_path,
            collection=collection,
            force=True,
            on_progress=on_progress,
        )

        progress_bar.progress(1.0, text="摄取完成！")
        status_text.empty()

        # 显示结果
        st.success(
            f"✅ 摄取成功！\n\n"
            f"- 文档 ID: `{result.doc_id}`\n"
            f"- Chunk 数: {result.chunk_count}\n"
            f"- 写入记录: {result.record_count}\n"
            f"- 耗时: {result.elapsed_ms:.0f}ms"
        )

        # 显示各阶段耗时
        if result.stage_times:
            with st.expander("各阶段耗时"):
                for stage, elapsed in result.stage_times.items():
                    st.markdown(f"- **{stage}**: {elapsed:.1f}ms")

    except Exception as e:
        st.error(f"❌ 摄取失败: {e}")


def _render_delete_section(data_service: DataService) -> None:
    """渲染文档删除区域。

    参数:
        data_service: DataService 实例
    """
    st.subheader("🗑️ 删除文档")

    try:
        documents = data_service.list_documents()
    except Exception as e:
        st.warning(f"加载文档列表失败: {e}")
        return

    if not documents:
        st.info("📭 暂无已摄入的文档。")
        return

    st.caption(f"共 {len(documents)} 个文档")

    for doc in documents:
        col_info, col_action = st.columns([4, 1])

        with col_info:
            st.markdown(
                f"**{_shorten_name(doc.source_path)}** "
                f"| 集合: {doc.collection} "
                f"| {doc.chunk_count} chunks "
                f"| {doc.image_count} images"
            )

        with col_action:
            if st.button(
                "🗑️ 删除",
                key=f"del_{doc.file_hash}",
                help=f"删除 {doc.source_path}",
            ):
                _delete_document(data_service, doc.source_path)


def _delete_document(
    data_service: DataService,
    source_path: str,
) -> None:
    """删除文档并显示结果。

    参数:
        data_service: DataService 实例
        source_path: 文档路径
    """
    manager = data_service._get_document_manager()
    result = manager.delete_document(source_path)

    if result.success:
        st.success(
            f"✅ 删除成功: {source_path}\n\n"
            f"- ChromaDB: {result.chunks_deleted} chunks\n"
            f"- BM25: {result.bm25_deleted} chunks\n"
            f"- 图片: {result.images_deleted} 张\n"
            f"- 完整性记录: {'已移除' if result.integrity_removed else '未找到'}"
        )
        st.rerun()
    else:
        st.error(f"❌ 删除部分失败: {'; '.join(result.errors)}")


# ============================================================
# 工具函数
# ============================================================


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
        "encode": "向量编码",
        "store": "存储写入",
    }
    return labels.get(stage_name, stage_name)


def _format_size(size_bytes: Optional[int]) -> str:
    """格式化文件大小。

    参数:
        size_bytes: 字节数

    返回:
        人类可读的文件大小
    """
    if size_bytes is None:
        return "未知"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _shorten_name(path: str, max_len: int = 40) -> str:
    """缩短文件名用于显示。

    参数:
        path: 完整路径
        max_len: 最大长度

    返回:
        缩短后的文件名
    """
    name = os.path.basename(path)
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."
