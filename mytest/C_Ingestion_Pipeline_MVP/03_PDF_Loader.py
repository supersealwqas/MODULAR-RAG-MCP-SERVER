"""C3 PDF Loader 手动测试脚本。

测试 PdfLoader 的文本提取、图片提取、占位符插入、序列化等。
可选指定 PDF 文件路径作为命令行参数。
"""

import json
import os
import sys


def safe_print(text: str):
    """安全打印，忽略无法编码的字符。"""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("gbk", errors="replace").decode("gbk"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.types import Document, ImageRef
from src.libs.loader.pdf_loader import PdfLoader


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_pdf(path: str, loader: PdfLoader):
    """测试加载指定 PDF 文件。"""
    section(f"加载: {os.path.basename(path)}")

    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        return None

    doc = loader.load(path, collection="test")
    print(f"ID:           {doc.id}")
    print(f"Text 长度:    {len(doc.text)} 字符")
    safe_print(f"Text 预览:    {doc.text[:150]}...")
    print()

    # metadata
    print("Metadata:")
    for k, v in doc.metadata.items():
        if k == "images":
            print(f"  {k}: [{len(v)} 张图片]")
        elif isinstance(v, str) and len(v) > 80:
            safe_print(f"  {k}: {v[:80]}...")
        else:
            safe_print(f"  {k}: {v}")

    # 图片
    images = doc.get_images()
    if images:
        print(f"\n图片详情 ({len(images)} 张):")
        for img in images:
            exists = os.path.isfile(img.path)
            print(f"  [{img.id}] page={img.page} size={img.position.get('width', '?')}x{img.position.get('height', '?')} saved={exists}")

    # 占位符
    placeholders = [p for p in doc.text.split("[IMAGE: ") if "]" in p]
    if placeholders:
        print(f"\n文本中的占位符 ({len(placeholders)} 个):")
        for p in placeholders[:5]:
            pid = p.split("]")[0]
            safe_print(f"  [IMAGE: {pid}]")

    # 序列化
    d = doc.to_dict()
    j = doc.to_json()
    doc2 = Document.from_json(j)
    print(f"\n序列化: dict_keys={list(d.keys())}, json_len={len(j)}, roundtrip_ok={doc.id == doc2.id}")

    return doc


def test_simple_pdf(loader: PdfLoader):
    """测试纯文本 PDF。"""
    test_pdf("tests/fixtures/sample_documents/simple.pdf", loader)


def test_image_pdf(loader: PdfLoader):
    """测试含图片 PDF。"""
    test_pdf("tests/fixtures/sample_documents/with_images.pdf", loader)


def test_real_pdf(loader: PdfLoader):
    """测试真实 PDF 文件。"""
    real = "data/documents/LLM基础知识.pdf"
    if os.path.exists(real):
        test_pdf(real, loader)
    else:
        print(f"跳过: {real} 不存在")


def test_no_image_extraction():
    section("关闭图片提取")
    loader = PdfLoader(extract_images=False)
    doc = loader.load("tests/fixtures/sample_documents/with_images.pdf")
    print(f"extract_images=False: images={len(doc.get_images())}")
    print(f"文本长度: {len(doc.text)}")


def test_file_not_found():
    section("文件不存在异常")
    loader = PdfLoader()
    try:
        loader.load("/nonexistent/file.pdf")
        print("ERROR: 应该抛出异常")
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")


def main():
    # 支持命令行指定 PDF 路径
    if len(sys.argv) > 1:
        path = sys.argv[1]
        loader = PdfLoader(image_dir="data/images", extract_images=True)
        test_pdf(path, loader)
    else:
        loader = PdfLoader(image_dir="data/images", extract_images=True)
        test_simple_pdf(loader)
        test_image_pdf(loader)
        test_real_pdf(loader)
        test_no_image_extraction()
        test_file_not_found()
        print(f"\n{'='*60}")
        print("  全部测试通过!")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
