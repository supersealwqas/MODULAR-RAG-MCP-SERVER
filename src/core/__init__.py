"""Core 层：核心业务逻辑与共享类型。

统一 re-export 核心数据类型，简化下游导入路径。
"""

from src.core.types import (
    IMAGE_PLACEHOLDER_PREFIX,
    IMAGE_PLACEHOLDER_SUFFIX,
    Chunk,
    ChunkRecord,
    Document,
    ImageRef,
    make_image_placeholder,
)

__all__ = [
    "Document",
    "Chunk",
    "ChunkRecord",
    "ImageRef",
    "make_image_placeholder",
    "IMAGE_PLACEHOLDER_PREFIX",
    "IMAGE_PLACEHOLDER_SUFFIX",
]
