"""C3 PDF Loader 契约测试。

覆盖 PdfLoader 的：
- 基本文本提取（纯文本 PDF）
- metadata 字段完整性（source_path, doc_type, doc_hash 等）
- 图片提取与占位符插入
- 图片提取失败时的降级行为
- 文件不存在时的异常处理
- Document 类型契约（可序列化）
"""

import json
import os

import pytest

from src.core.types import Document, ImageRef
from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.pdf_loader import PdfLoader

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures", "sample_documents")
SIMPLE_PDF = os.path.join(FIXTURES_DIR, "simple.pdf")
WITH_IMAGES_PDF = os.path.join(FIXTURES_DIR, "with_images.pdf")


@pytest.fixture
def loader(tmp_path):
    """提供使用临时图片目录的 PdfLoader。"""
    return PdfLoader(image_dir=str(tmp_path / "images"), extract_images=True)


@pytest.fixture
def loader_no_images(tmp_path):
    """提供不提取图片的 PdfLoader。"""
    return PdfLoader(image_dir=str(tmp_path / "images"), extract_images=False)


# ============================================================
# 基本加载测试
# ============================================================

class TestPdfLoaderBasic:
    """基本 PDF 加载功能测试。"""

    def test_load_simple_pdf(self, loader):
        """加载纯文本 PDF 成功，返回 Document 对象。"""
        doc = loader.load(SIMPLE_PDF)
        assert isinstance(doc, Document)
        assert len(doc.text) > 0

    def test_metadata_contains_source_path(self, loader):
        """metadata 包含 source_path 字段。"""
        doc = loader.load(SIMPLE_PDF)
        assert "source_path" in doc.metadata
        assert os.path.isabs(doc.metadata["source_path"])
        assert doc.metadata["source_path"].endswith("simple.pdf")

    def test_metadata_doc_type(self, loader):
        """metadata.doc_type 为 'pdf'。"""
        doc = loader.load(SIMPLE_PDF)
        assert doc.metadata["doc_type"] == "pdf"

    def test_metadata_doc_hash(self, loader):
        """metadata.doc_hash 为 16 位十六进制字符串。"""
        doc = loader.load(SIMPLE_PDF)
        assert "doc_hash" in doc.metadata
        assert len(doc.metadata["doc_hash"]) == 16
        assert all(c in "0123456789abcdef" for c in doc.metadata["doc_hash"])

    def test_metadata_file_size(self, loader):
        """metadata.file_size 大于 0。"""
        doc = loader.load(SIMPLE_PDF)
        assert doc.metadata["file_size"] > 0

    def test_metadata_title(self, loader):
        """metadata.title 非空。"""
        doc = loader.load(SIMPLE_PDF)
        assert "title" in doc.metadata
        assert len(doc.metadata["title"]) > 0

    def test_id_is_hash_prefix(self, loader):
        """Document.id 等于 doc_hash。"""
        doc = loader.load(SIMPLE_PDF)
        assert doc.id == doc.metadata["doc_hash"]

    def test_deterministic_load(self, loader):
        """同一文件多次加载产生相同的 id 和 text。"""
        doc1 = loader.load(SIMPLE_PDF)
        doc2 = loader.load(SIMPLE_PDF)
        assert doc1.id == doc2.id
        assert doc1.text == doc2.text


# ============================================================
# 图片提取测试
# ============================================================

class TestPdfLoaderImageExtraction:
    """图片提取功能测试。"""

    def test_extract_images_from_pdf(self, loader):
        """从包含图片的 PDF 中提取图片。"""
        doc = loader.load(WITH_IMAGES_PDF)
        images = doc.get_images()
        # 应至少提取到 1 张图片
        assert len(images) >= 1

    def test_image_ref_fields(self, loader):
        """提取的 ImageRef 字段完整。"""
        doc = loader.load(WITH_IMAGES_PDF)
        images = doc.get_images()
        for img in images:
            assert isinstance(img, ImageRef)
            assert len(img.id) > 0
            assert os.path.isabs(img.path)
            assert img.page >= 1

    def test_image_id_format(self, loader):
        """image_id 格式为 {doc_hash}_{page}_{seq}。"""
        doc = loader.load(WITH_IMAGES_PDF)
        images = doc.get_images()
        doc_hash = doc.metadata["doc_hash"]
        for img in images:
            assert img.id.startswith(doc_hash)

    def test_image_files_saved(self, loader):
        """提取的图片文件已保存到磁盘。"""
        doc = loader.load(WITH_IMAGES_PDF)
        images = doc.get_images()
        for img in images:
            assert os.path.isfile(img.path)

    def test_image_placeholder_in_text(self, loader):
        """文本中包含 [IMAGE: {id}] 占位符。"""
        doc = loader.load(WITH_IMAGES_PDF)
        images = doc.get_images()
        for img in images:
            assert f"[IMAGE: {img.id}]" in doc.text

    def test_metadata_images_field(self, loader):
        """metadata.images 包含图片信息列表。"""
        doc = loader.load(WITH_IMAGES_PDF)
        assert "images" in doc.metadata
        assert isinstance(doc.metadata["images"], list)
        assert len(doc.metadata["images"]) >= 1
        # 每条记录包含必要字段
        for img_data in doc.metadata["images"]:
            assert "id" in img_data
            assert "path" in img_data
            assert "page" in img_data

    def test_simple_pdf_no_images(self, loader):
        """纯文本 PDF 的 images 字段为空列表或不存在。"""
        doc = loader.load(SIMPLE_PDF)
        images = doc.get_images()
        assert len(images) == 0


# ============================================================
# 降级测试
# ============================================================

class TestPdfLoaderDegradation:
    """降级行为测试。"""

    def test_no_image_extraction(self, loader_no_images):
        """关闭图片提取时，不提取图片。"""
        doc = loader_no_images.load(WITH_IMAGES_PDF)
        images = doc.get_images()
        assert len(images) == 0
        # 但文本仍正常提取
        assert len(doc.text) > 0

    def test_image_extraction_error_graceful(self, tmp_path):
        """图片提取失败时不阻塞文本解析。"""
        # 使用一个有效的 PDF 但指定一个不可写的图片目录
        loader = PdfLoader(image_dir="/nonexistent/path/images", extract_images=True)
        doc = loader.load(WITH_IMAGES_PDF)
        # 文本应正常提取
        assert len(doc.text) > 0
        assert isinstance(doc, Document)


# ============================================================
# 异常处理测试
# ============================================================

class TestPdfLoaderErrors:
    """异常处理测试。"""

    def test_file_not_found(self, loader):
        """文件不存在时抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError, match="文件不存在"):
            loader.load("/nonexistent/file.pdf")

    def test_nonexistent_path_error_message(self, loader):
        """错误信息包含文件路径。"""
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load("/some/fake/path.pdf")
        assert "/some/fake/path.pdf" in str(exc_info.value)


# ============================================================
# Document 契约测试
# ============================================================

class TestDocumentContract:
    """验证输出的 Document 符合 C1 定义的契约。"""

    def test_document_serializable_dict(self, loader):
        """Document 可序列化为字典。"""
        doc = loader.load(SIMPLE_PDF)
        d = doc.to_dict()
        assert "id" in d
        assert "text" in d
        assert "metadata" in d

    def test_document_serializable_json(self, loader):
        """Document 可序列化为 JSON 字符串。"""
        doc = loader.load(SIMPLE_PDF)
        j = doc.to_json()
        parsed = json.loads(j)
        assert parsed["id"] == doc.id
        assert parsed["text"] == doc.text

    def test_document_roundtrip(self, loader):
        """Document 序列化/反序列化后数据一致。"""
        doc = loader.load(SIMPLE_PDF)
        restored = Document.from_dict(doc.to_dict())
        assert restored.id == doc.id
        assert restored.text == doc.text
        assert restored.metadata == doc.metadata


# ============================================================
# BaseLoader 抽象接口测试
# ============================================================

class TestBaseLoaderAbstract:
    """BaseLoader 抽象接口测试。"""

    def test_base_loader_is_abstract(self):
        """BaseLoader 不能直接实例化。"""
        with pytest.raises(TypeError):
            BaseLoader()

    def test_pdf_loader_is_subclass(self):
        """PdfLoader 是 BaseLoader 的子类。"""
        assert issubclass(PdfLoader, BaseLoader)

    def test_pdf_loader_instance(self, loader):
        """PdfLoader 实例是 BaseLoader 类型。"""
        assert isinstance(loader, BaseLoader)


# ============================================================
# Re-export 测试
# ============================================================

class TestReExport:
    """验证 __init__.py 的 re-export 正确。"""

    def test_import_from_loader(self):
        """从 src.libs.loader 可直接导入所有类。"""
        from src.libs.loader import BaseLoader, PdfLoader
        assert BaseLoader is not None
        assert PdfLoader is not None
