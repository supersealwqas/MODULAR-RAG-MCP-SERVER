"""多模态返回组装模块。

当检索命中 chunk 含 image_refs 时，读取图片并 base64 编码，
返回 MCP 格式的 ImageContent。
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from typing import Any, Dict, List, Optional

from src.core.types import RetrievalResult

logger = logging.getLogger(__name__)

# 支持的图片 MIME 类型
SUPPORTED_IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def _get_mime_type(file_path: str) -> str:
    """根据文件扩展名获取 MIME 类型。

    参数:
        file_path: 文件路径

    返回:
        MIME 类型字符串，默认 image/png
    """
    ext = os.path.splitext(file_path)[1].lower()
    return SUPPORTED_IMAGE_TYPES.get(ext, "image/png")


def _read_image_as_base64(file_path: str) -> Optional[str]:
    """读取图片文件并转换为 base64 字符串。

    参数:
        file_path: 图片文件路径

    返回:
        base64 编码的字符串，失败时返回 None
    """
    try:
        if not os.path.isfile(file_path):
            logger.warning("图片文件不存在: %s", file_path)
            return None

        with open(file_path, "rb") as f:
            image_data = f.read()

        return base64.b64encode(image_data).decode("utf-8")

    except Exception as e:
        logger.error("读取图片失败 %s: %s", file_path, e)
        return None


def assemble_image_content(
    file_path: str,
    mime_type: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """组装单个图片的 MCP ImageContent。

    参数:
        file_path: 图片文件路径
        mime_type: MIME 类型（可选，自动检测）

    返回:
        MCP ImageContent 字典，失败时返回 None
    """
    if mime_type is None:
        mime_type = _get_mime_type(file_path)

    base64_data = _read_image_as_base64(file_path)
    if base64_data is None:
        return None

    return {
        "type": "image",
        "data": base64_data,
        "mimeType": mime_type,
    }


def assemble_multimodal_response(
    results: List[RetrievalResult],
    text_content: str,
    max_images: int = 5,
) -> List[Dict[str, Any]]:
    """组装多模态响应（Text + Image）。

    从检索结果中提取 image_refs，读取图片并生成 MCP content 列表。

    参数:
        results: 检索结果列表
        text_content: Markdown 文本内容
        max_images: 最大图片数量（防止响应过大）

    返回:
        MCP content 列表，包含 TextContent 和 ImageContent
    """
    content: List[Dict[str, Any]] = []

    # 添加文本内容
    content.append({"type": "text", "text": text_content})

    # 收集所有图片引用
    image_refs = []
    for result in results:
        images = result.metadata.get("images", [])
        for img in images:
            if isinstance(img, dict):
                image_refs.append(img)
            elif hasattr(img, "to_dict"):
                image_refs.append(img.to_dict())

    # 去重（按 image_id）
    seen_ids = set()
    unique_refs = []
    for ref in image_refs:
        img_id = ref.get("id", "")
        if img_id and img_id not in seen_ids:
            seen_ids.add(img_id)
            unique_refs.append(ref)

    # 限制图片数量
    unique_refs = unique_refs[:max_images]

    # 读取并组装图片
    for ref in unique_refs:
        file_path = ref.get("path", "")
        if not file_path:
            continue

        image_content = assemble_image_content(file_path)
        if image_content:
            content.append(image_content)
            logger.debug("添加图片: %s", file_path)

    return content
