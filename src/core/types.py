"""核心数据类型/契约模块。

定义全链路（Ingestion → Retrieval → MCP Tools）共用的数据结构。
所有下游模块必须使用此处定义的类型，禁止在子模块内重复定义。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================
# 图片引用规范
# ============================================================
# metadata["images"] 字段结构：
#   List[ImageRef]
#   每个 ImageRef 包含：
#     - id:       全局唯一图片标识符（格式：{doc_hash}_{page}_{seq}）
#     - path:     图片文件存储路径（约定：data/images/{collection}/{image_id}.png）
#     - page:     图片在原文档中的页码（可选）
#     - text_offset: 占位符在 Document.text 中的起始字符位置
#     - text_length: 占位符的字符长度
#     - position: 图片在原文档中的物理位置信息（可选）
#
# 文本中图片占位符格式：[IMAGE: {image_id}]
# ============================================================

IMAGE_PLACEHOLDER_PREFIX = "[IMAGE: "
IMAGE_PLACEHOLDER_SUFFIX = "]"


def make_image_placeholder(image_id: str) -> str:
    """生成图片占位符文本。

    参数:
        image_id: 图片唯一标识符

    返回:
        占位符字符串，如 "[IMAGE: abc123]"
    """
    return f"{IMAGE_PLACEHOLDER_PREFIX}{image_id}{IMAGE_PLACEHOLDER_SUFFIX}"


@dataclass
class ImageRef:
    """图片引用数据类，记录文档中一张图片的元信息。

    属性:
        id: 全局唯一图片标识符（建议格式：{doc_hash}_{page}_{seq}）
        path: 图片文件存储路径
        page: 图片在原文档中的页码（可选）
        text_offset: 占位符在 Document.text 中的起始字符位置
        text_length: 占位符的字符长度
        position: 图片在原文档中的物理位置信息（可选）
    """
    id: str
    path: str
    page: int = 0
    text_offset: int = 0
    text_length: int = 0
    position: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ImageRef:
        """从字典反序列化。"""
        return cls(**data)


# ============================================================
# Document — 原始文档
# ============================================================

@dataclass
class Document:
    """原始文档数据类，表示 Loader 解析后的完整文档。

    属性:
        id: 文档唯一标识（建议使用文件 SHA256 前 16 位）
        text: 文档全文（含图片占位符 [IMAGE: {id}]）
        metadata: 附加元数据，必须包含 source_path
    """
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """校验 metadata 中必须包含 source_path。"""
        if "source_path" not in self.metadata:
            raise ValueError("Document.metadata 必须包含 'source_path' 字段")

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典，支持 JSON 输出。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Document:
        """从字典反序列化。"""
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=data.get("metadata", {}),
        )

    def to_json(self, **kwargs) -> str:
        """序列化为 JSON 字符串。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)

    @classmethod
    def from_json(cls, json_str: str) -> Document:
        """从 JSON 字符串反序列化。"""
        return cls.from_dict(json.loads(json_str))

    def get_images(self) -> List[ImageRef]:
        """获取文档中所有图片引用列表。"""
        raw = self.metadata.get("images", [])
        return [ImageRef.from_dict(img) if isinstance(img, dict) else img for img in raw]


# ============================================================
# Chunk — 切分后的文本片段
# ============================================================

@dataclass
class Chunk:
    """文本片段数据类，表示 Splitter 切分后的单个 Chunk。

    属性:
        id: Chunk 唯一标识（格式：{doc_id}_{index:04d}_{hash_8chars}）
        text: 片段文本内容（可能含图片占位符）
        metadata: 附加元数据，继承自 Document.metadata 并附加 chunk_index 等
        start_offset: 片段在原文档 text 中的起始字符偏移
        end_offset: 片段在原文档 text 中的结束字符偏移
        source_ref: 父 Document.id，用于溯源
    """
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_offset: int = 0
    end_offset: int = 0
    source_ref: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典，支持 JSON 输出。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Chunk:
        """从字典反序列化。"""
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=data.get("metadata", {}),
            start_offset=data.get("start_offset", 0),
            end_offset=data.get("end_offset", 0),
            source_ref=data.get("source_ref"),
        )

    def to_json(self, **kwargs) -> str:
        """序列化为 JSON 字符串。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)

    @classmethod
    def from_json(cls, json_str: str) -> Chunk:
        """从 JSON 字符串反序列化。"""
        return cls.from_dict(json.loads(json_str))

    def get_images(self) -> List[ImageRef]:
        """获取当前 Chunk 引用的图片列表（按需分发的子集）。"""
        raw = self.metadata.get("images", [])
        return [ImageRef.from_dict(img) if isinstance(img, dict) else img for img in raw]

    def get_image_refs(self) -> List[str]:
        """获取当前 Chunk 引用的图片 ID 列表。"""
        return self.metadata.get("image_refs", [])


# ============================================================
# ChunkRecord — 存储/检索载体
# ============================================================

@dataclass
class ChunkRecord:
    """存储记录数据类，用于向量库和检索引擎之间的数据交换。

    属性:
        id: 记录唯一标识（与 Chunk.id 一致）
        text: 文本内容
        metadata: 附加元数据
        dense_vector: 稠密向量（可选，C8 填充）
        sparse_vector: 稀疏向量/term weights（可选，C9 填充）
    """
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    dense_vector: Optional[List[float]] = None
    sparse_vector: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典，支持 JSON 输出。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChunkRecord:
        """从字典反序列化。"""
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=data.get("metadata", {}),
            dense_vector=data.get("dense_vector"),
            sparse_vector=data.get("sparse_vector"),
        )

    def to_json(self, **kwargs) -> str:
        """序列化为 JSON 字符串。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)

    @classmethod
    def from_json(cls, json_str: str) -> ChunkRecord:
        """从 JSON 字符串反序列化。"""
        return cls.from_dict(json.loads(json_str))


# ============================================================
# ProcessedQuery — 查询预处理结果
# ============================================================

@dataclass
class ProcessedQuery:
    """查询预处理结果数据类，表示 QueryProcessor 的输出。

    属性:
        original: 原始查询文本
        keywords: 提取的关键词列表（非空）
        filters: 解析出的过滤条件字典（可为空 dict）
    """
    original: str
    keywords: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProcessedQuery:
        """从字典反序列化。"""
        return cls(
            original=data["original"],
            keywords=data.get("keywords", []),
            filters=data.get("filters", {}),
        )


# ============================================================
# RetrievalResult — 检索结果
# ============================================================

@dataclass
class RetrievalResult:
    """检索结果数据类，表示单条检索命中。

    属性:
        chunk_id: 命中 Chunk 的 ID
        score: 相似度/相关性分数
        text: Chunk 文本内容
        metadata: 附加元数据
    """
    chunk_id: str
    score: float
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RetrievalResult:
        """从字典反序列化。"""
        return cls(
            chunk_id=data["chunk_id"],
            score=data["score"],
            text=data.get("text", ""),
            metadata=data.get("metadata", {}),
        )
