"""DocumentChunker 适配器模块。

将 libs.splitter 的纯文本切分结果（List[str]）转换为符合 core.types 契约的
Chunk 业务对象，附加 ID 生成、元数据继承、图片引用分发等业务逻辑。
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List

from src.core.settings import Settings
from src.core.types import (
    IMAGE_PLACEHOLDER_PREFIX,
    IMAGE_PLACEHOLDER_SUFFIX,
    Chunk,
    Document,
    ImageRef,
)
from src.libs.splitter.splitter_factory import SplitterFactory


class DocumentChunker:
    """文档切分适配器，连接 libs.splitter 与 Ingestion Pipeline。

    核心职责（相比 libs.splitter 的增值）：
    1. Chunk ID 生成：唯一且确定性（格式：{doc_id}_{index:04d}_{hash_8chars}）
    2. 元数据继承：Document.metadata → Chunk.metadata
    3. chunk_index：记录 chunk 在文档中的序号
    4. source_ref：指向父 Document.id
    5. 图片引用按需分发：扫描占位符，提取对应 ImageRef 子集
    6. 类型转换：List[str] → List[Chunk]
    """

    def __init__(self, settings: Settings) -> None:
        """初始化 DocumentChunker。

        参数:
            settings: 全局配置对象，用于读取 splitter 配置
        """
        splitter_cfg = settings.splitter
        # 构建工厂参数，过滤空 separators
        kwargs: Dict[str, Any] = {}
        if splitter_cfg.separators:
            kwargs["separators"] = splitter_cfg.separators
        kwargs["keep_code_blocks"] = splitter_cfg.keep_code_blocks
        kwargs["keep_headers"] = splitter_cfg.keep_headers

        self._splitter = SplitterFactory.create(
            strategy=splitter_cfg.strategy,
            chunk_size=splitter_cfg.chunk_size,
            chunk_overlap=splitter_cfg.chunk_overlap,
            **kwargs,
        )

    def split_document(self, document: Document) -> List[Chunk]:
        """将 Document 切分为 Chunk 列表。

        完整流程：调用 splitter 切分文本 → 逐个生成 Chunk 业务对象。

        参数:
            document: 待切分的 Document 对象

        返回:
            Chunk 对象列表，包含完整元数据和图片引用
        """
        # 1. 调用 libs.splitter 切分纯文本
        text_chunks = self._splitter.split_text(document.text)

        # 2. 预处理：计算每个 chunk 在原文中的偏移位置
        offsets = self._compute_offsets(document.text, text_chunks)

        # 3. 构建 Chunk 业务对象
        chunks: List[Chunk] = []
        for index, chunk_text in enumerate(text_chunks):
            chunk_id = self._generate_chunk_id(document.id, index, chunk_text)
            metadata = self._inherit_metadata(document, index, chunk_text)
            start_offset, end_offset = offsets[index] if index < len(offsets) else (0, 0)

            chunk = Chunk(
                id=chunk_id,
                text=chunk_text,
                metadata=metadata,
                start_offset=start_offset,
                end_offset=end_offset,
                source_ref=document.id,
            )
            chunks.append(chunk)

        return chunks

    def _generate_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        """生成唯一且确定性的 Chunk ID。

        格式：{doc_id}_{index:04d}_{hash_8chars}
        hash 基于 chunk 文本内容，确保相同输入产生相同 ID。

        参数:
            doc_id: 父文档 ID
            index: chunk 在文档中的序号
            text: chunk 文本内容

        返回:
            Chunk ID 字符串
        """
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"{doc_id}_{index:04d}_{text_hash}"

    def _inherit_metadata(self, document: Document, chunk_index: int, chunk_text: str) -> Dict[str, Any]:
        """继承 Document 元数据并附加 chunk 级信息。

        包含：
        - 复制 Document.metadata 所有字段（source_path, doc_type, title 等）
        - 添加 chunk_index
        - 图片引用按需分发：扫描 chunk_text 中的 [IMAGE: id] 占位符，
          从 Document.metadata["images"] 中提取对应 ImageRef 子集

        参数:
            document: 父 Document 对象
            chunk_index: chunk 序号
            chunk_text: chunk 文本内容（用于扫描占位符）

        返回:
            Chunk metadata 字典
        """
        # 复制文档级元数据（浅拷贝，避免修改原始 Document）
        metadata = dict(document.metadata)

        # 添加 chunk 级字段
        metadata["chunk_index"] = chunk_index

        # 图片引用按需分发
        image_ids_in_chunk = self._extract_image_ids(chunk_text)
        if image_ids_in_chunk:
            # 从文档级 images 中筛选出本 chunk 引用的子集
            doc_images = document.metadata.get("images", [])
            chunk_images = [
                img.to_dict() if isinstance(img, ImageRef) else img
                for img in doc_images
                if (img.id if isinstance(img, ImageRef) else img.get("id", "")) in image_ids_in_chunk
            ]
            metadata["images"] = chunk_images
            metadata["image_refs"] = list(image_ids_in_chunk)
        else:
            # 无占位符的 chunk 不含 images 字段
            metadata.pop("images", None)
            metadata.pop("image_refs", None)

        return metadata

    def _extract_image_ids(self, text: str) -> List[str]:
        """从文本中提取所有 [IMAGE: {id}] 占位符的 image_id。

        参数:
            text: 待扫描的文本

        返回:
            image_id 列表（保持出现顺序，去重）
        """
        pattern = re.escape(IMAGE_PLACEHOLDER_PREFIX) + r"([^" + re.escape(IMAGE_PLACEHOLDER_SUFFIX) + r"]+)" + re.escape(IMAGE_PLACEHOLDER_SUFFIX)
        matches = re.findall(pattern, text)
        # 保持顺序去重
        seen = set()
        result = []
        for img_id in matches:
            if img_id not in seen:
                seen.add(img_id)
                result.append(img_id)
        return result

    def _compute_offsets(self, original_text: str, chunks: List[str]) -> List[tuple]:
        """计算每个 chunk 在原文中的字符偏移范围。

        使用顺序搜索定位每个 chunk 在原文中的位置。
        策略：先在上一 chunk 终点直接验证（处理重复字符文本），
        不匹配再用 find 搜索（处理 overlap 和非均匀文本）。

        参数:
            original_text: 文档原始全文
            chunks: 切分后的文本块列表

        返回:
            [(start_offset, end_offset), ...] 列表
        """
        offsets = []
        expected_pos = 0
        for chunk_text in chunks:
            if not chunk_text:
                offsets.append((expected_pos, expected_pos))
                continue

            chunk_len = len(chunk_text)

            # 策略1：直接在预期位置验证（处理重复字符文本的歧义）
            if (expected_pos + chunk_len <= len(original_text)
                    and original_text[expected_pos:expected_pos + chunk_len] == chunk_text):
                idx = expected_pos
            else:
                # 策略2：用 find 搜索（处理 overlap 或 strip 后的文本）
                idx = original_text.find(chunk_text, expected_pos)

            if idx >= 0:
                offsets.append((idx, idx + chunk_len))
                # 下一 chunk 的预期起点：当前起点 + chunk 长度
                # overlap 场景下 find 会自动找到正确位置
                expected_pos = idx + chunk_len
            else:
                # 精准匹配失败（如 strip 导致），保持游标不剧烈跳动
                offsets.append((expected_pos, expected_pos + chunk_len))
        return offsets
