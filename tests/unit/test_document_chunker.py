"""C4 DocumentChunker 单元测试。

使用 FakeSplitter 隔离测试，无需真实外部依赖。
覆盖验收标准：配置驱动、ID 唯一性/确定性、元数据完整性、图片分发、溯源链接、类型契约。
"""

from __future__ import annotations

import re
from typing import List

import pytest

from src.core.settings import SplitterConfig
from src.core.types import (
    IMAGE_PLACEHOLDER_PREFIX,
    IMAGE_PLACEHOLDER_SUFFIX,
    Chunk,
    Document,
    ImageRef,
    make_image_placeholder,
)
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.splitter_factory import register_splitter


# ============================================================
# FakeSplitter：可控的切分器，用于隔离测试
# ============================================================

# 用一个唯一名称注册，避免污染其他测试
_FAKE_STRATEGY = "test_fake_chunker"


@register_splitter(_FAKE_STRATEGY)
class FakeSplitter(BaseSplitter):
    """测试用切分器，按 chunk_size 硬切分。"""

    def split_text(self, text: str, **kwargs) -> List[str]:
        """按 chunk_size 硬切分文本。"""
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size
        return chunks


# ============================================================
# 辅助函数
# ============================================================

def _make_settings(chunk_size: int = 100, chunk_overlap: int = 0) -> SplitterConfig:
    """创建测试用 SplitterConfig。"""
    return SplitterConfig(
        strategy=_FAKE_STRATEGY,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def _make_doc(
    doc_id: str = "doc_test_001",
    text: str = "这是一段测试文本。" * 20,
    images: list | None = None,
) -> Document:
    """创建测试用 Document。"""
    metadata = {"source_path": "/test/document.pdf", "doc_type": "pdf"}
    if images is not None:
        metadata["images"] = images
    return Document(id=doc_id, text=text, metadata=metadata)


class _SettingsStub:
    """模拟 Settings 对象，只包含 splitter 配置。"""

    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 0):
        self.splitter = _make_settings(chunk_size, chunk_overlap)


# ============================================================
# 测试：基本功能
# ============================================================

class TestDocumentChunkerBasic:
    """基本功能测试。"""

    def test_split_document_returns_chunks(self):
        """split_document 返回 Chunk 对象列表。"""
        settings = _SettingsStub(chunk_size=50)
        chunker = DocumentChunker(settings)
        doc = _make_doc(text="短文本内容。")
        chunks = chunker.split_document(doc)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_empty_text_returns_empty_list(self):
        """空文本应返回空列表。"""
        settings = _SettingsStub(chunk_size=50)
        chunker = DocumentChunker(settings)
        doc = _make_doc(text="")
        chunks = chunker.split_document(doc)
        assert chunks == []

    def test_short_text_single_chunk(self):
        """短于 chunk_size 的文本应产生单个 chunk。"""
        settings = _SettingsStub(chunk_size=1000)
        chunker = DocumentChunker(settings)
        doc = _make_doc(text="很短的文本。")
        chunks = chunker.split_document(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "很短的文本。"


# ============================================================
# 测试：配置驱动
# ============================================================

class TestConfigDriven:
    """验收标准：配置驱动——chunk_size 变化导致 chunk 数量变化。"""

    def test_larger_chunk_size_fewer_chunks(self):
        """较大的 chunk_size 产生较少的 chunk。"""
        text = "A" * 500

        settings_small = _SettingsStub(chunk_size=100)
        chunker_small = DocumentChunker(settings_small)
        chunks_small = chunker_small.split_document(_make_doc(text=text))

        settings_large = _SettingsStub(chunk_size=250)
        chunker_large = DocumentChunker(settings_large)
        chunks_large = chunker_large.split_document(_make_doc(text=text))

        assert len(chunks_small) > len(chunks_large)

    def test_chunk_size_affects_length(self):
        """chunk_size 约束每个 chunk 的最大长度。"""
        settings = _SettingsStub(chunk_size=80)
        chunker = DocumentChunker(settings)
        doc = _make_doc(text="B" * 300)
        chunks = chunker.split_document(doc)
        for chunk in chunks:
            assert len(chunk.text) <= 80


# ============================================================
# 测试：ID 唯一性与确定性
# ============================================================

class TestChunkId:
    """验收标准：ID 唯一性 + 确定性。"""

    def test_ids_unique(self):
        """同一文档切分后所有 Chunk ID 唯一。"""
        settings = _SettingsStub(chunk_size=50)
        chunker = DocumentChunker(settings)
        doc = _make_doc(text="C" * 200)
        chunks = chunker.split_document(doc)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids)), f"存在重复 ID: {ids}"

    def test_ids_deterministic(self):
        """同一 Document 对象重复切分产生相同的 Chunk ID 序列。"""
        settings = _SettingsStub(chunk_size=60)
        chunker = DocumentChunker(settings)
        doc = _make_doc(text="D" * 150)

        chunks_1 = chunker.split_document(doc)
        chunks_2 = chunker.split_document(doc)

        ids_1 = [c.id for c in chunks_1]
        ids_2 = [c.id for c in chunks_2]
        assert ids_1 == ids_2

    def test_id_format(self):
        """Chunk ID 格式为 {doc_id}_{index:04d}_{hash_8chars}。"""
        settings = _SettingsStub(chunk_size=50)
        chunker = DocumentChunker(settings)
        doc = _make_doc(doc_id="my_doc_42", text="E" * 100)
        chunks = chunker.split_document(doc)

        for i, chunk in enumerate(chunks):
            parts = chunk.id.split("_")
            # 最后一部分是 8 字符 hash
            assert len(parts[-1]) == 8
            # 倒数第二部分是 4 位 index
            assert parts[-2] == f"{i:04d}"

    def test_same_text_same_hash(self):
        """相同文本内容产生相同的 hash 部分。"""
        chunker = DocumentChunker(_SettingsStub(chunk_size=1000))
        doc1 = _make_doc(doc_id="doc_a", text="相同的文本内容")
        doc2 = _make_doc(doc_id="doc_b", text="相同的文本内容")

        chunks1 = chunker.split_document(doc1)
        chunks2 = chunker.split_document(doc2)

        # hash 部分相同（文本相同）
        hash1 = chunks1[0].id.split("_")[-1]
        hash2 = chunks2[0].id.split("_")[-1]
        assert hash1 == hash2


# ============================================================
# 测试：元数据完整性
# ============================================================

class TestMetadataInheritance:
    """验收标准：Chunk.metadata 包含所有 Document.metadata 字段 + chunk_index。"""

    def test_metadata_inherits_source_path(self):
        """Chunk 继承 Document 的 source_path。"""
        chunker = DocumentChunker(_SettingsStub(chunk_size=50))
        doc = _make_doc(text="F" * 100)
        chunks = chunker.split_document(doc)
        for chunk in chunks:
            assert chunk.metadata["source_path"] == "/test/document.pdf"

    def test_metadata_inherits_doc_type(self):
        """Chunk 继承 Document 的 doc_type。"""
        chunker = DocumentChunker(_SettingsStub(chunk_size=50))
        doc = _make_doc(text="G" * 100)
        chunks = chunker.split_document(doc)
        for chunk in chunks:
            assert chunk.metadata["doc_type"] == "pdf"

    def test_metadata_has_chunk_index(self):
        """每个 Chunk 的 metadata 包含 chunk_index（从 0 开始）。"""
        chunker = DocumentChunker(_SettingsStub(chunk_size=30))
        doc = _make_doc(text="H" * 100)
        chunks = chunker.split_document(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i

    def test_metadata_custom_fields_preserved(self):
        """自定义元数据字段（如 title）也被继承。"""
        chunker = DocumentChunker(_SettingsStub(chunk_size=50))
        doc = _make_doc(text="I" * 100)
        doc.metadata["title"] = "测试文档标题"
        doc.metadata["custom_field"] = 42
        chunks = chunker.split_document(doc)
        for chunk in chunks:
            assert chunk.metadata["title"] == "测试文档标题"
            assert chunk.metadata["custom_field"] == 42


# ============================================================
# 测试：source_ref 溯源链接
# ============================================================

class TestSourceRef:
    """验收标准：所有 Chunk.source_ref 正确指向父 Document.id。"""

    def test_source_ref_points_to_doc_id(self):
        """每个 Chunk 的 source_ref 等于父 Document.id。"""
        chunker = DocumentChunker(_SettingsStub(chunk_size=50))
        doc = _make_doc(doc_id="parent_doc_xyz", text="J" * 200)
        chunks = chunker.split_document(doc)
        for chunk in chunks:
            assert chunk.source_ref == "parent_doc_xyz"

    def test_source_ref_not_none(self):
        """source_ref 不为 None。"""
        chunker = DocumentChunker(_SettingsStub(chunk_size=50))
        doc = _make_doc(text="K" * 100)
        chunks = chunker.split_document(doc)
        for chunk in chunks:
            assert chunk.source_ref is not None


# ============================================================
# 测试：图片引用按需分发
# ============================================================

class TestImageDistribution:
    """验收标准：含占位符的 chunk 只包含对应图片子集；无占位符的 chunk 无 images 字段。"""

    def _make_doc_with_images(self) -> Document:
        """创建包含两个图片占位符的 Document。"""
        img1 = {"id": "img_001", "path": "data/images/test/img_001.png", "page": 1,
                "text_offset": 10, "text_length": 16, "position": {}}
        img2 = {"id": "img_002", "path": "data/images/test/img_002.png", "page": 2,
                "text_offset": 60, "text_length": 16, "position": {}}

        placeholder1 = make_image_placeholder("img_001")
        placeholder2 = make_image_placeholder("img_002")

        # 构造文本，让两个占位符落在不同 chunk 中
        # chunk_size=40，前 40 字符包含 placeholder1，后 40 字符包含 placeholder2
        text = f"前缀文本。{placeholder1}中间填充文本AAAA{placeholder2}后缀文本。"
        return _make_doc(text=text, images=[img1, img2])

    def test_chunk_with_image_gets_subset(self):
        """含占位符的 chunk 其 metadata['images'] 仅包含该 chunk 引用的图片。"""
        chunker = DocumentChunker(_SettingsStub(chunk_size=40, chunk_overlap=0))
        doc = self._make_doc_with_images()
        chunks = chunker.split_document(doc)

        for chunk in chunks:
            if "[IMAGE: img_001]" in chunk.text and "[IMAGE: img_002]" in chunk.text:
                # 两个占位符都在同一 chunk 中
                assert len(chunk.metadata["images"]) == 2
                assert len(chunk.metadata["image_refs"]) == 2
            elif "[IMAGE: img_001]" in chunk.text:
                img_ids = [img["id"] if isinstance(img, dict) else img.id for img in chunk.metadata["images"]]
                assert "img_001" in img_ids
                assert "img_002" not in img_ids
            elif "[IMAGE: img_002]" in chunk.text:
                img_ids = [img["id"] if isinstance(img, dict) else img.id for img in chunk.metadata["images"]]
                assert "img_002" in img_ids
                assert "img_001" not in img_ids
            else:
                # 无占位符的 chunk 不含 images 字段
                assert "images" not in chunk.metadata
                assert "image_refs" not in chunk.metadata

    def test_no_placeholder_no_images(self):
        """不含占位符的 chunk 无 images 和 image_refs 字段。"""
        chunker = DocumentChunker(_SettingsStub(chunk_size=50))
        doc = _make_doc(text="纯文本没有任何图片占位符。" * 10)
        chunks = chunker.split_document(doc)
        for chunk in chunks:
            assert "images" not in chunk.metadata
            assert "image_refs" not in chunk.metadata

    def test_image_refs_list_matches_placeholders(self):
        """metadata['image_refs'] 列表与文本中的占位符一致。"""
        img1 = {"id": "pic_a", "path": "data/images/test/pic_a.png", "page": 1,
                "text_offset": 0, "text_length": 0, "position": {}}
        placeholder = make_image_placeholder("pic_a")
        text = f"开头{placeholder}结尾"

        chunker = DocumentChunker(_SettingsStub(chunk_size=1000))
        doc = _make_doc(text=text, images=[img1])
        chunks = chunker.split_document(doc)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.metadata["image_refs"] == ["pic_a"]
        assert len(chunk.metadata["images"]) == 1

    def test_multiple_same_placeholder_dedup(self):
        """同一 chunk 中重复的占位符在 image_refs 中去重。"""
        img1 = {"id": "dup_img", "path": "data/images/test/dup_img.png", "page": 1,
                "text_offset": 0, "text_length": 0, "position": {}}
        placeholder = make_image_placeholder("dup_img")
        text = f"{placeholder}一些文字{placeholder}更多文字"

        chunker = DocumentChunker(_SettingsStub(chunk_size=1000))
        doc = _make_doc(text=text, images=[img1])
        chunks = chunker.split_document(doc)

        assert len(chunks) == 1
        assert chunks[0].metadata["image_refs"] == ["dup_img"]
        assert len(chunks[0].metadata["images"]) == 1


# ============================================================
# 测试：类型契约（序列化）
# ============================================================

class TestTypeContract:
    """验收标准：输出的 Chunk 对象符合 core.types 契约（可序列化、字段完整）。"""

    def test_chunk_to_dict_roundtrip(self):
        """Chunk 可序列化为 dict 并反序列化回来。"""
        chunker = DocumentChunker(_SettingsStub(chunk_size=50))
        doc = _make_doc(text="L" * 100)
        chunks = chunker.split_document(doc)

        for chunk in chunks:
            d = chunk.to_dict()
            chunk2 = Chunk.from_dict(d)
            assert chunk == chunk2

    def test_chunk_to_json_roundtrip(self):
        """Chunk 可序列化为 JSON 并反序列化回来。"""
        chunker = DocumentChunker(_SettingsStub(chunk_size=50))
        doc = _make_doc(text="M" * 100)
        chunks = chunker.split_document(doc)

        for chunk in chunks:
            j = chunk.to_json()
            chunk2 = Chunk.from_json(j)
            assert chunk == chunk2

    def test_chunk_has_required_fields(self):
        """Chunk 对象包含所有必要字段。"""
        chunker = DocumentChunker(_SettingsStub(chunk_size=50))
        doc = _make_doc(text="N" * 100)
        chunks = chunker.split_document(doc)

        for chunk in chunks:
            assert hasattr(chunk, "id")
            assert hasattr(chunk, "text")
            assert hasattr(chunk, "metadata")
            assert hasattr(chunk, "start_offset")
            assert hasattr(chunk, "end_offset")
            assert hasattr(chunk, "source_ref")


# ============================================================
# 测试：offset 计算
# ============================================================

class TestOffsets:
    """测试 start_offset / end_offset 计算。"""

    def test_offsets_cover_full_text(self):
        """所有 chunk 的 offset 拼接应覆盖原文范围。"""
        text = "O" * 200
        chunker = DocumentChunker(_SettingsStub(chunk_size=50))
        doc = _make_doc(text=text)
        chunks = chunker.split_document(doc)

        # 第一个 chunk 从 0 开始
        assert chunks[0].start_offset == 0
        # 最后一个 chunk 结束于文本末尾
        assert chunks[-1].end_offset == len(text)

    def test_offsets_sequential(self):
        """连续 chunk 的 offset 应顺序递增（无 overlap 时）。"""
        text = "P" * 150
        chunker = DocumentChunker(_SettingsStub(chunk_size=50, chunk_overlap=0))
        doc = _make_doc(text=text)
        chunks = chunker.split_document(doc)

        for i in range(1, len(chunks)):
            assert chunks[i].start_offset >= chunks[i - 1].start_offset


# ============================================================
# 测试：边界场景
# ============================================================

class TestEdgeCases:
    """边界场景测试。"""

    def test_unicode_text(self):
        """中文/Unicode 文本正常切分。"""
        text = "这是一段包含中文和English混合的文本。" * 10
        chunker = DocumentChunker(_SettingsStub(chunk_size=50))
        doc = _make_doc(text=text)
        chunks = chunker.split_document(doc)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.text

    def test_single_char_chunks(self):
        """极端小的 chunk_size 不崩溃。"""
        chunker = DocumentChunker(_SettingsStub(chunk_size=5))
        doc = _make_doc(text="短")
        chunks = chunker.split_document(doc)
        assert len(chunks) == 1

    def test_document_without_images_field(self):
        """Document.metadata 不含 images 字段时正常工作。"""
        chunker = DocumentChunker(_SettingsStub(chunk_size=50))
        doc = _make_doc(text="Q" * 100, images=None)
        # images=None 时 metadata 中不包含 images 字段
        chunks = chunker.split_document(doc)
        for chunk in chunks:
            assert "images" not in chunk.metadata

    def test_empty_images_list(self):
        """Document.metadata.images 为空列表时正常工作。"""
        chunker = DocumentChunker(_SettingsStub(chunk_size=50))
        doc = _make_doc(text="R" * 100, images=[])
        chunks = chunker.split_document(doc)
        for chunk in chunks:
            assert "images" not in chunk.metadata
