"""C4 DocumentChunker 手动测试脚本。

测试 DocumentChunker 的切分、ID 生成、元数据继承、图片分发等。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.settings import SplitterConfig
from src.core.types import Document, ImageRef, make_image_placeholder
# 导入 RecursiveSplitter 以触发 @register_splitter 注册
from src.libs.splitter.recursive_splitter import RecursiveSplitter  # noqa: F401
from src.ingestion.chunking.document_chunker import DocumentChunker


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


class _SettingsStub:
    """模拟 Settings 对象。"""
    def __init__(self, chunk_size=100, chunk_overlap=0, strategy="recursive"):
        self.splitter = SplitterConfig(
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def test_basic_split():
    section("基本切分")
    settings = _SettingsStub(chunk_size=200)
    chunker = DocumentChunker(settings)
    doc = Document(
        id="test_doc_001",
        text="这是一段测试文本。" * 50,
        metadata={"source_path": "/test/basic.pdf", "doc_type": "pdf"},
    )
    chunks = chunker.split_document(doc)
    print(f"文档文本长度: {len(doc.text)}")
    print(f"chunk_size: 200")
    print(f"切分结果: {len(chunks)} 个 chunk")
    for i, chunk in enumerate(chunks):
        print(f"  [{i}] id={chunk.id} len={len(chunk.text)} offset=[{chunk.start_offset},{chunk.end_offset}) source_ref={chunk.source_ref}")


def test_config_driven():
    section("配置驱动")
    text = "A" * 1000
    for size in [100, 200, 500]:
        settings = _SettingsStub(chunk_size=size)
        chunker = DocumentChunker(settings)
        doc = Document(
            id="config_test",
            text=text,
            metadata={"source_path": "/test/config.pdf"},
        )
        chunks = chunker.split_document(doc)
        print(f"  chunk_size={size:>4} → {len(chunks)} chunks, max_len={max(len(c.text) for c in chunks)}")


def test_id_determinism():
    section("ID 确定性")
    settings = _SettingsStub(chunk_size=100)
    chunker = DocumentChunker(settings)
    doc = Document(
        id="determinism_test",
        text="确定性测试文本。" * 30,
        metadata={"source_path": "/test/determinism.pdf"},
    )
    chunks_1 = chunker.split_document(doc)
    chunks_2 = chunker.split_document(doc)
    ids_1 = [c.id for c in chunks_1]
    ids_2 = [c.id for c in chunks_2]
    print(f"第一次: {ids_1}")
    print(f"第二次: {ids_2}")
    print(f"ID 完全一致: {ids_1 == ids_2}")
    print(f"ID 唯一性: {len(ids_1) == len(set(ids_1))}")


def test_metadata_inheritance():
    section("元数据继承")
    settings = _SettingsStub(chunk_size=100)
    chunker = DocumentChunker(settings)
    doc = Document(
        id="meta_test",
        text="元数据测试。" * 30,
        metadata={
            "source_path": "/test/meta.pdf",
            "doc_type": "pdf",
            "title": "测试文档标题",
            "custom_field": 42,
        },
    )
    chunks = chunker.split_document(doc)
    chunk = chunks[0]
    print(f"Chunk.metadata keys: {list(chunk.metadata.keys())}")
    print(f"  source_path: {chunk.metadata['source_path']}")
    print(f"  doc_type: {chunk.metadata['doc_type']}")
    print(f"  title: {chunk.metadata['title']}")
    print(f"  custom_field: {chunk.metadata['custom_field']}")
    print(f"  chunk_index: {chunk.metadata['chunk_index']}")
    print(f"  source_ref: {chunk.source_ref}")


def test_image_distribution():
    section("图片引用按需分发")
    img1 = {"id": "img_001", "path": "data/images/test/img_001.png", "page": 1, "text_offset": 5, "text_length": 16, "position": {}}
    img2 = {"id": "img_002", "path": "data/images/test/img_002.png", "page": 2, "text_offset": 50, "text_length": 16, "position": {}}
    p1 = make_image_placeholder("img_001")
    p2 = make_image_placeholder("img_002")
    text = f"前缀。{p1}中间填充文本AAAA{p2}后缀结束。"

    settings = _SettingsStub(chunk_size=50, chunk_overlap=0)
    chunker = DocumentChunker(settings)
    doc = Document(
        id="img_test",
        text=text,
        metadata={"source_path": "/test/images.pdf", "images": [img1, img2]},
    )
    chunks = chunker.split_document(doc)
    print(f"切分为 {len(chunks)} 个 chunk:")
    for i, chunk in enumerate(chunks):
        has_img = "images" in chunk.metadata
        refs = chunk.metadata.get("image_refs", [])
        print(f"  [{i}] len={len(chunk.text)} has_images={has_img} refs={refs}")
        if has_img:
            for img in chunk.metadata["images"]:
                print(f"       → {img['id']} ({img['path']})")


def test_serialization():
    section("序列化 roundtrip")
    settings = _SettingsStub(chunk_size=100)
    chunker = DocumentChunker(settings)
    doc = Document(
        id="serial_test",
        text="序列化测试文本。" * 20,
        metadata={"source_path": "/test/serial.pdf", "doc_type": "pdf"},
    )
    chunks = chunker.split_document(doc)
    for chunk in chunks:
        d = chunk.to_dict()
        j = chunk.to_json()
        from src.core.types import Chunk
        chunk2 = Chunk.from_json(j)
        assert chunk == chunk2
    print(f"所有 {len(chunks)} 个 chunk 序列化 roundtrip 通过")


def test_real_pdf():
    section("真实 PDF 文件（需先用 C3 Loader 加载）")
    try:
        from src.libs.loader.pdf_loader import PdfLoader
        loader = PdfLoader(image_dir="data/images", extract_images=True)

        pdf_path = "data/documents/LLM基础知识.pdf"
        if not os.path.exists(pdf_path):
            print(f"跳过: {pdf_path} 不存在")
            return

        doc = loader.load(pdf_path, collection="test")
        print(f"Document.id: {doc.id}")
        print(f"Document.text 长度: {len(doc.text)}")
        print(f"图片数量: {len(doc.get_images())}")

        settings = _SettingsStub(chunk_size=500, chunk_overlap=50)
        chunker = DocumentChunker(settings)
        chunks = chunker.split_document(doc)
        print(f"\n切分结果: {len(chunks)} 个 chunk")

        img_chunks = [c for c in chunks if "images" in c.metadata]
        print(f"含图片的 chunk: {len(img_chunks)} 个")

        for i, chunk in enumerate(chunks[:5]):
            refs = chunk.metadata.get("image_refs", [])
            print(f"  [{i}] id={chunk.id} len={len(chunk.text)} refs={refs}")

        if len(chunks) > 5:
            print(f"  ... 共 {len(chunks)} 个 chunk")

    except Exception as e:
        print(f"错误: {e}")


def main():
    test_basic_split()
    test_config_driven()
    test_id_determinism()
    test_metadata_inheritance()
    test_image_distribution()
    test_serialization()
    test_real_pdf()
    print(f"\n{'='*60}")
    print("  全部测试通过!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
