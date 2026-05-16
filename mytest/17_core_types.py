"""C1 核心数据类型手动测试脚本。

测试 Document / Chunk / ChunkRecord / ImageRef 的创建、序列化、图片占位符等。
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.types import (
    Document, Chunk, ChunkRecord, ImageRef,
    make_image_placeholder, IMAGE_PLACEHOLDER_PREFIX,
)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_image_ref():
    section("ImageRef 基本操作")

    # 创建
    ref = ImageRef(id="doc1_1_0", path="data/images/test/doc1_1_0.png", page=1)
    print(f"创建: id={ref.id}, path={ref.path}, page={ref.page}")

    # 序列化
    d = ref.to_dict()
    print(f"to_dict: {d}")

    # 反序列化
    ref2 = ImageRef.from_dict(d)
    print(f"from_dict: id={ref2.id}, page={ref2.page}")
    print(f"Roundtrip 一致: {ref == ref2}")


def test_image_placeholder():
    section("图片占位符")

    placeholder = make_image_placeholder("abc123")
    print(f"占位符: '{placeholder}'")
    print(f"前缀: '{IMAGE_PLACEHOLDER_PREFIX}'")

    # 嵌入文本
    text = f"这是前文。{placeholder}这是后文。"
    print(f"嵌入文本: '{text}'")
    print(f"位置索引: {text.index(placeholder)}")


def test_document():
    section("Document 基本操作")

    # 创建
    doc = Document(
        id="doc_001",
        text="这是一段测试文本，包含图片占位符。",
        metadata={"source_path": "/path/to/test.pdf", "doc_type": "pdf"},
    )
    print(f"创建: id={doc.id}, text_len={len(doc.text)}")
    print(f"metadata: {doc.metadata}")

    # 序列化
    d = doc.to_dict()
    print(f"to_dict keys: {list(d.keys())}")

    j = doc.to_json()
    print(f"to_json length: {len(j)}")

    # 反序列化
    doc2 = Document.from_json(j)
    print(f"Roundtrip 一致: {doc.id == doc2.id and doc.text == doc2.text}")

    # 校验：缺少 source_path
    try:
        Document(id="bad", text="x", metadata={})
        print("ERROR: 应该抛出 ValueError")
    except ValueError as e:
        print(f"校验通过: {e}")


def test_document_with_images():
    section("Document 带图片")

    images = [
        {"id": "doc2_1_0", "path": "data/images/test/doc2_1_0.png", "page": 1, "text_offset": 10, "text_length": 18, "position": {}},
        {"id": "doc2_2_0", "path": "data/images/test/doc2_2_0.png", "page": 2, "text_offset": 50, "text_length": 18, "position": {}},
    ]
    text = "文本前缀。[IMAGE: doc2_1_0]中间文本[IMAGE: doc2_2_0]文本结束。"
    doc = Document(
        id="doc_002",
        text=text,
        metadata={"source_path": "/img.pdf", "images": images},
    )
    print(f"图片数量: {len(doc.metadata['images'])}")
    print(f"get_images: {[img.id for img in doc.get_images()]}")
    print(f"文本包含占位符: {'[IMAGE: doc2_1_0]' in doc.text}")


def test_chunk():
    section("Chunk 基本操作")

    chunk = Chunk(
        id="doc_001_0000_abc12345",
        text="这是第一个chunk。",
        metadata={"source_path": "/test.pdf", "chunk_index": 0},
        start_offset=0, end_offset=10,
        source_ref="doc_001",
    )
    print(f"创建: id={chunk.id}, source_ref={chunk.source_ref}")
    print(f"offsets: [{chunk.start_offset}, {chunk.end_offset})")

    # 序列化 roundtrip
    d = chunk.to_dict()
    chunk2 = Chunk.from_dict(d)
    print(f"Roundtrip 一致: {chunk == chunk2}")

    # 带图片的 chunk
    chunk_img = Chunk(
        id="doc_001_0001_def",
        text="文本[IMAGE: img_x]结束",
        metadata={
            "images": [{"id": "img_x", "path": "data/images/test/img_x.png", "page": 1, "text_offset": 2, "text_length": 14, "position": {}}],
            "image_refs": ["img_x"],
        },
        source_ref="doc_001",
    )
    print(f"Chunk 图片: {chunk_img.get_image_refs()}")


def test_chunk_record():
    section("ChunkRecord 基本操作")

    record = ChunkRecord(
        id="rec_001",
        text="存储记录",
        metadata={"source_path": "/test.pdf"},
        dense_vector=[0.1, 0.2, 0.3],
        sparse_vector={"hello": 1.5, "world": 0.8},
    )
    print(f"创建: id={record.id}")
    print(f"dense_vector dim: {len(record.dense_vector)}")
    print(f"sparse_vector keys: {list(record.sparse_vector.keys())}")

    # roundtrip
    d = record.to_dict()
    record2 = ChunkRecord.from_dict(d)
    print(f"Roundtrip 一致: {record == record2}")

    # 无向量
    record_no_vec = ChunkRecord(id="r2", text="无向量", metadata={})
    d2 = record_no_vec.to_dict()
    print(f"无向量 roundtrip: dense={ChunkRecord.from_dict(d2).dense_vector}")


def main():
    test_image_ref()
    test_image_placeholder()
    test_document()
    test_document_with_images()
    test_chunk()
    test_chunk_record()
    print(f"\n{'='*60}")
    print("  全部测试通过!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
