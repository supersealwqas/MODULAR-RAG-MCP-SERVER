"""C1 核心数据类型单元测试。

覆盖 Document/Chunk/ChunkRecord/ImageRef 的：
- 创建与字段校验
- 序列化/反序列化（dict/JSON roundtrip）
- metadata.images 字段规范
- 文本图片占位符规范
"""

import json

import pytest

from src.core.types import (
    IMAGE_PLACEHOLDER_PREFIX,
    IMAGE_PLACEHOLDER_SUFFIX,
    Chunk,
    ChunkRecord,
    Document,
    ImageRef,
    make_image_placeholder,
)


# ============================================================
# ImageRef 测试
# ============================================================

class TestImageRef:
    """ImageRef 数据类测试。"""

    def test_create_with_defaults(self):
        """默认值正确：page=0, text_offset=0, text_length=0, position={}。"""
        ref = ImageRef(id="img_001", path="data/images/test/img_001.png")
        assert ref.id == "img_001"
        assert ref.path == "data/images/test/img_001.png"
        assert ref.page == 0
        assert ref.text_offset == 0
        assert ref.text_length == 0
        assert ref.position == {}

    def test_create_with_all_fields(self):
        """所有字段可正确赋值。"""
        ref = ImageRef(
            id="doc123_1_0",
            path="data/images/test/doc123_1_0.png",
            page=1,
            text_offset=42,
            text_length=20,
            position={"x": 100, "y": 200, "width": 300, "height": 400},
        )
        assert ref.page == 1
        assert ref.text_offset == 42
        assert ref.text_length == 20
        assert ref.position["x"] == 100

    def test_to_dict(self):
        """序列化为字典后字段完整。"""
        ref = ImageRef(id="img_001", path="data/images/test/img_001.png", page=2)
        d = ref.to_dict()
        assert d["id"] == "img_001"
        assert d["path"] == "data/images/test/img_001.png"
        assert d["page"] == 2
        assert d["text_offset"] == 0

    def test_from_dict(self):
        """从字典反序列化后字段一致。"""
        data = {
            "id": "img_002",
            "path": "data/images/test/img_002.png",
            "page": 3,
            "text_offset": 10,
            "text_length": 18,
            "position": {"x": 50},
        }
        ref = ImageRef.from_dict(data)
        assert ref.id == "img_002"
        assert ref.page == 3
        assert ref.text_offset == 10
        assert ref.position == {"x": 50}

    def test_roundtrip_dict(self):
        """dict roundtrip：to_dict → from_dict 数据不丢失。"""
        original = ImageRef(
            id="doc1_page1_0",
            path="data/images/test/doc1_page1_0.png",
            page=1,
            text_offset=100,
            text_length=22,
            position={"width": 800},
        )
        restored = ImageRef.from_dict(original.to_dict())
        assert restored == original


# ============================================================
# 图片占位符工具函数测试
# ============================================================

class TestImagePlaceholder:
    """图片占位符工具函数测试。"""

    def test_make_image_placeholder(self):
        """占位符格式正确。"""
        result = make_image_placeholder("abc123")
        assert result == "[IMAGE: abc123]"

    def test_placeholder_constants(self):
        """前后缀常量正确。"""
        assert IMAGE_PLACEHOLDER_PREFIX == "[IMAGE: "
        assert IMAGE_PLACEHOLDER_SUFFIX == "]"

    def test_placeholder_in_text(self):
        """占位符可正确嵌入文本。"""
        text = f"这是前文。{make_image_placeholder('img_001')}这是后文。"
        assert "[IMAGE: img_001]" in text
        assert text.index("[IMAGE: img_001]") > 0


# ============================================================
# Document 测试
# ============================================================

class TestDocument:
    """Document 数据类测试。"""

    def test_create_basic(self):
        """基本创建成功，metadata 包含 source_path。"""
        doc = Document(
            id="doc_001",
            text="这是一段测试文本。",
            metadata={"source_path": "/path/to/doc.pdf"},
        )
        assert doc.id == "doc_001"
        assert doc.text == "这是一段测试文本。"
        assert doc.metadata["source_path"] == "/path/to/doc.pdf"

    def test_missing_source_path_raises(self):
        """metadata 缺少 source_path 时抛出 ValueError。"""
        with pytest.raises(ValueError, match="source_path"):
            Document(id="doc_001", text="text", metadata={})

    def test_empty_metadata_raises(self):
        """空 metadata 也必须校验 source_path。"""
        with pytest.raises(ValueError, match="source_path"):
            Document(id="doc_001", text="text")

    def test_to_dict(self):
        """序列化为字典后字段完整。"""
        doc = Document(
            id="doc_001",
            text="text",
            metadata={"source_path": "/path/to/doc.pdf", "title": "测试"},
        )
        d = doc.to_dict()
        assert d["id"] == "doc_001"
        assert d["text"] == "text"
        assert d["metadata"]["source_path"] == "/path/to/doc.pdf"
        assert d["metadata"]["title"] == "测试"

    def test_from_dict(self):
        """从字典反序列化后字段一致。"""
        data = {
            "id": "doc_002",
            "text": "hello",
            "metadata": {"source_path": "/a/b.pdf"},
        }
        doc = Document.from_dict(data)
        assert doc.id == "doc_002"
        assert doc.text == "hello"
        assert doc.metadata["source_path"] == "/a/b.pdf"

    def test_roundtrip_dict(self):
        """dict roundtrip：to_dict → from_dict 数据不丢失。"""
        original = Document(
            id="doc_003",
            text="完整的文本内容",
            metadata={"source_path": "/test.pdf", "doc_type": "pdf"},
        )
        restored = Document.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.text == original.text
        assert restored.metadata == original.metadata

    def test_to_json(self):
        """JSON 序列化可正常输出。"""
        doc = Document(
            id="doc_001",
            text="中文测试",
            metadata={"source_path": "/test.pdf"},
        )
        j = doc.to_json()
        parsed = json.loads(j)
        assert parsed["id"] == "doc_001"
        assert parsed["text"] == "中文测试"

    def test_from_json(self):
        """JSON 反序列化后字段一致。"""
        j = '{"id": "doc_004", "text": "json test", "metadata": {"source_path": "/x.pdf"}}'
        doc = Document.from_json(j)
        assert doc.id == "doc_004"
        assert doc.text == "json test"

    def test_roundtrip_json(self):
        """JSON roundtrip：to_json → from_json 数据不丢失。"""
        original = Document(
            id="doc_005",
            text="JSON往返测试",
            metadata={"source_path": "/json.pdf", "extra": "data"},
        )
        restored = Document.from_json(original.to_json())
        assert restored.id == original.id
        assert restored.text == original.text
        assert restored.metadata == original.metadata

    def test_metadata_images_field(self):
        """metadata.images 字段可正确存储 ImageRef 列表。"""
        images = [
            {"id": "doc1_1_0", "path": "data/images/test/doc1_1_0.png", "page": 1, "text_offset": 10, "text_length": 20, "position": {}},
            {"id": "doc1_2_0", "path": "data/images/test/doc1_2_0.png", "page": 2, "text_offset": 50, "text_length": 20, "position": {}},
        ]
        doc = Document(
            id="doc_006",
            text="文本[IMAGE: doc1_1_0]中间[IMAGE: doc1_2_0]结束",
            metadata={"source_path": "/img.pdf", "images": images},
        )
        assert len(doc.metadata["images"]) == 2
        assert doc.metadata["images"][0]["id"] == "doc1_1_0"

    def test_get_images(self):
        """get_images() 返回 ImageRef 对象列表。"""
        images = [
            {"id": "img_a", "path": "data/images/collection/img_a.png", "page": 1, "text_offset": 0, "text_length": 14, "position": {}},
        ]
        doc = Document(
            id="doc_007",
            text="[IMAGE: img_a]",
            metadata={"source_path": "/test.pdf", "images": images},
        )
        refs = doc.get_images()
        assert len(refs) == 1
        assert isinstance(refs[0], ImageRef)
        assert refs[0].id == "img_a"

    def test_metadata_extensible(self):
        """metadata 允许增量扩展字段，不破坏兼容。"""
        doc = Document(
            id="doc_008",
            text="text",
            metadata={
                "source_path": "/test.pdf",
                "title": "标题",
                "summary": "摘要",
                "tags": ["tag1", "tag2"],
                "custom_field": 42,
            },
        )
        d = doc.to_dict()
        restored = Document.from_dict(d)
        assert restored.metadata["custom_field"] == 42
        assert restored.metadata["tags"] == ["tag1", "tag2"]


# ============================================================
# Chunk 测试
# ============================================================

class TestChunk:
    """Chunk 数据类测试。"""

    def test_create_basic(self):
        """基本创建成功。"""
        chunk = Chunk(
            id="doc_001_0000_abc12345",
            text="这是第一个chunk。",
            metadata={"source_path": "/test.pdf", "chunk_index": 0},
            start_offset=0,
            end_offset=10,
            source_ref="doc_001",
        )
        assert chunk.id == "doc_001_0000_abc12345"
        assert chunk.start_offset == 0
        assert chunk.end_offset == 10
        assert chunk.source_ref == "doc_001"

    def test_default_values(self):
        """默认值正确：start_offset=0, end_offset=0, source_ref=None。"""
        chunk = Chunk(id="c1", text="text")
        assert chunk.start_offset == 0
        assert chunk.end_offset == 0
        assert chunk.source_ref is None
        assert chunk.metadata == {}

    def test_to_dict(self):
        """序列化为字典后字段完整。"""
        chunk = Chunk(
            id="c1",
            text="text",
            metadata={"chunk_index": 0},
            start_offset=10,
            end_offset=20,
            source_ref="doc_001",
        )
        d = chunk.to_dict()
        assert d["id"] == "c1"
        assert d["start_offset"] == 10
        assert d["end_offset"] == 20
        assert d["source_ref"] == "doc_001"

    def test_from_dict(self):
        """从字典反序列化后字段一致。"""
        data = {
            "id": "c2",
            "text": "chunk text",
            "metadata": {"chunk_index": 1},
            "start_offset": 100,
            "end_offset": 200,
            "source_ref": "doc_002",
        }
        chunk = Chunk.from_dict(data)
        assert chunk.id == "c2"
        assert chunk.source_ref == "doc_002"

    def test_roundtrip_dict(self):
        """dict roundtrip：to_dict → from_dict 数据不丢失。"""
        original = Chunk(
            id="c3",
            text="往返测试",
            metadata={"source_path": "/test.pdf", "chunk_index": 2},
            start_offset=30,
            end_offset=40,
            source_ref="doc_003",
        )
        restored = Chunk.from_dict(original.to_dict())
        assert restored == original

    def test_roundtrip_json(self):
        """JSON roundtrip：to_json → from_json 数据不丢失。"""
        original = Chunk(
            id="c4",
            text="JSON往返",
            metadata={"source_path": "/json.pdf"},
            source_ref="doc_004",
        )
        restored = Chunk.from_json(original.to_json())
        assert restored.id == original.id
        assert restored.text == original.text
        assert restored.source_ref == original.source_ref

    def test_metadata_images_subset(self):
        """Chunk 的 metadata.images 仅包含该 chunk 引用的图片子集。"""
        chunk_images = [
            {"id": "doc1_1_0", "path": "data/images/test/doc1_1_0.png", "page": 1, "text_offset": 10, "text_length": 20, "position": {}},
        ]
        chunk = Chunk(
            id="doc_001_0000_abc",
            text="文本[IMAGE: doc1_1_0]结束",
            metadata={
                "source_path": "/test.pdf",
                "chunk_index": 0,
                "images": chunk_images,
                "image_refs": ["doc1_1_0"],
            },
            source_ref="doc_001",
        )
        assert len(chunk.metadata["images"]) == 1
        assert chunk.metadata["images"][0]["id"] == "doc1_1_0"

    def test_get_images(self):
        """get_images() 返回该 chunk 引用的 ImageRef 子集。"""
        chunk = Chunk(
            id="c5",
            text="[IMAGE: img_x]",
            metadata={
                "images": [{"id": "img_x", "path": "data/images/test/img_x.png", "page": 1, "text_offset": 0, "text_length": 14, "position": {}}],
                "image_refs": ["img_x"],
            },
        )
        refs = chunk.get_images()
        assert len(refs) == 1
        assert refs[0].id == "img_x"

    def test_get_image_refs(self):
        """get_image_refs() 返回图片 ID 列表。"""
        chunk = Chunk(
            id="c6",
            text="[IMAGE: a][IMAGE: b]",
            metadata={"image_refs": ["a", "b"]},
        )
        assert chunk.get_image_refs() == ["a", "b"]

    def test_no_images_field_when_empty(self):
        """不含图片占位符的 chunk 不应包含 images 字段（由 DocumentChunker 负责）。"""
        chunk = Chunk(
            id="c7",
            text="纯文本chunk，无图片引用。",
            metadata={"source_path": "/test.pdf"},
        )
        assert "images" not in chunk.metadata
        assert chunk.get_images() == []
        assert chunk.get_image_refs() == []


# ============================================================
# ChunkRecord 测试
# ============================================================

class TestChunkRecord:
    """ChunkRecord 数据类测试。"""

    def test_create_basic(self):
        """基本创建成功。"""
        record = ChunkRecord(
            id="rec_001",
            text="存储记录文本",
            metadata={"source_path": "/test.pdf"},
        )
        assert record.id == "rec_001"
        assert record.dense_vector is None
        assert record.sparse_vector is None

    def test_create_with_vectors(self):
        """包含向量字段时创建成功。"""
        record = ChunkRecord(
            id="rec_002",
            text="带向量的记录",
            metadata={"source_path": "/test.pdf"},
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={"hello": 1.5, "world": 0.8},
        )
        assert len(record.dense_vector) == 3
        assert record.sparse_vector["hello"] == 1.5

    def test_to_dict(self):
        """序列化为字典后字段完整。"""
        record = ChunkRecord(
            id="rec_003",
            text="text",
            metadata={"source_path": "/test.pdf"},
            dense_vector=[0.1, 0.2],
            sparse_vector={"term": 1.0},
        )
        d = record.to_dict()
        assert d["id"] == "rec_003"
        assert d["dense_vector"] == [0.1, 0.2]
        assert d["sparse_vector"] == {"term": 1.0}

    def test_from_dict(self):
        """从字典反序列化后字段一致。"""
        data = {
            "id": "rec_004",
            "text": "text",
            "metadata": {"source_path": "/test.pdf"},
            "dense_vector": [0.5, 0.6],
            "sparse_vector": {"word": 2.0},
        }
        record = ChunkRecord.from_dict(data)
        assert record.dense_vector == [0.5, 0.6]
        assert record.sparse_vector == {"word": 2.0}

    def test_roundtrip_dict(self):
        """dict roundtrip：to_dict → from_dict 数据不丢失。"""
        original = ChunkRecord(
            id="rec_005",
            text="往返测试",
            metadata={"source_path": "/test.pdf", "chunk_index": 0},
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={"hello": 1.0},
        )
        restored = ChunkRecord.from_dict(original.to_dict())
        assert restored == original

    def test_roundtrip_json(self):
        """JSON roundtrip：to_json → from_json 数据不丢失。"""
        original = ChunkRecord(
            id="rec_006",
            text="JSON往返",
            metadata={"source_path": "/json.pdf"},
        )
        restored = ChunkRecord.from_json(original.to_json())
        assert restored.id == original.id
        assert restored.text == original.text

    def test_none_vectors_roundtrip(self):
        """None 向量字段在 roundtrip 后保持 None。"""
        original = ChunkRecord(
            id="rec_007",
            text="无向量",
            metadata={},
        )
        restored = ChunkRecord.from_dict(original.to_dict())
        assert restored.dense_vector is None
        assert restored.sparse_vector is None


# ============================================================
# Re-export 测试（从 src.core 导入）
# ============================================================

class TestCoreReExport:
    """验证 src.core.__init__.py 的 re-export 正确。"""

    def test_import_from_core(self):
        """从 src.core 可直接导入所有核心类型。"""
        from src.core import (
            IMAGE_PLACEHOLDER_PREFIX,
            IMAGE_PLACEHOLDER_SUFFIX,
            Chunk,
            ChunkRecord,
            Document,
            ImageRef,
            make_image_placeholder,
        )
        # 验证类型正确
        assert Document is not None
        assert Chunk is not None
        assert ChunkRecord is not None
        assert ImageRef is not None
        assert callable(make_image_placeholder)
        assert isinstance(IMAGE_PLACEHOLDER_PREFIX, str)
        assert isinstance(IMAGE_PLACEHOLDER_SUFFIX, str)
