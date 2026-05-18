"""
MultimodalAssembler 单元测试

覆盖：图片读取、base64 编码、MIME 类型检测、多模态响应组装。
使用临时图片文件进行测试。
"""

import base64
import os
import tempfile

import pytest

from src.core.response.multimodal_assembler import (
    assemble_image_content,
    assemble_multimodal_response,
    _get_mime_type,
    _read_image_as_base64,
)
from src.core.types import RetrievalResult


@pytest.fixture
def temp_image():
    """创建临时图片文件。"""
    # 创建一个小的 PNG 图片（1x1 像素）
    # PNG 文件头 + IHDR chunk + IDAT chunk + IEND chunk
    png_data = (
        b'\x89PNG\r\n\x1a\n'  # PNG 签名
        b'\x00\x00\x00\rIHDR'  # IHDR chunk
        b'\x00\x00\x00\x01'  # width: 1
        b'\x00\x00\x00\x01'  # height: 1
        b'\x08\x02'  # bit depth: 8, color type: 2 (RGB)
        b'\x00\x00\x00'  # compression, filter, interlace
        b'\x90wS\xde'  # CRC
        b'\x00\x00\x00\x0cIDAT'  # IDAT chunk
        b'x\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05'  # compressed data
        b'\x18\xd8N\x00\x00\x00\x00IEND'  # IEND chunk
        b'\xaeB`\x82'  # CRC
    )

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(png_data)
        temp_path = f.name

    yield temp_path

    # 清理
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_jpg():
    """创建临时 JPEG 文件。"""
    # 最小的 JPEG 文件
    jpg_data = (
        b'\xff\xd8\xff\xe0'  # JPEG SOI + APP0 marker
        b'\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'  # JFIF header
        b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07'  # DQT
        b'\x09\x09\x08\x0a\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f'
        b'\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342'
        b'\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00'  # SOF
        b'\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00'  # DHT
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08'
        b'\t\n\x0b'  # DHT continued
        b'\xff\xda\x00\x08\x01\x01\x00\x00?\x00T\xd5'  # SOS
        b'\xff\xd9'  # EOI
    )

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(jpg_data)
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestGetMimeType:
    """_get_mime_type 测试。"""

    def test_png_extension(self):
        """应返回 image/png。"""
        assert _get_mime_type("test.png") == "image/png"

    def test_jpg_extension(self):
        """应返回 image/jpeg。"""
        assert _get_mime_type("test.jpg") == "image/jpeg"

    def test_jpeg_extension(self):
        """应返回 image/jpeg。"""
        assert _get_mime_type("test.jpeg") == "image/jpeg"

    def test_gif_extension(self):
        """应返回 image/gif。"""
        assert _get_mime_type("test.gif") == "image/gif"

    def test_webp_extension(self):
        """应返回 image/webp。"""
        assert _get_mime_type("test.webp") == "image/webp"

    def test_unknown_extension(self):
        """未知扩展名应返回 image/png（默认）。"""
        assert _get_mime_type("test.xyz") == "image/png"


class TestReadImageAsBase64:
    """_read_image_as_base64 测试。"""

    def test_read_existing_file(self, temp_image):
        """应成功读取并返回 base64 字符串。"""
        result = _read_image_as_base64(temp_image)
        assert result is not None
        # 验证是有效的 base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_read_nonexistent_file(self):
        """不存在的文件应返回 None。"""
        result = _read_image_as_base64("/nonexistent/path/image.png")
        assert result is None


class TestAssembleImageContent:
    """assemble_image_content 测试。"""

    def test_returns_image_content(self, temp_image):
        """应返回正确的 ImageContent 结构。"""
        result = assemble_image_content(temp_image)
        assert result is not None
        assert result["type"] == "image"
        assert "data" in result
        assert result["mimeType"] == "image/png"

    def test_custom_mime_type(self, temp_image):
        """应使用自定义 MIME 类型。"""
        result = assemble_image_content(temp_image, mime_type="image/jpeg")
        assert result["mimeType"] == "image/jpeg"

    def test_nonexistent_file(self):
        """不存在的文件应返回 None。"""
        result = assemble_image_content("/nonexistent/image.png")
        assert result is None


class TestAssembleMultimodalResponse:
    """assemble_multimodal_response 测试。"""

    def test_returns_text_only_when_no_images(self):
        """无图片时应只返回文本。"""
        results = [
            RetrievalResult(
                chunk_id="c1",
                score=0.9,
                text="测试文本",
                metadata={},
            )
        ]
        content = assemble_multimodal_response(results, "Markdown 内容")
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Markdown 内容"

    def test_returns_text_and_image(self, temp_image):
        """有图片时应返回文本和图片。"""
        results = [
            RetrievalResult(
                chunk_id="c1",
                score=0.9,
                text="测试文本",
                metadata={
                    "images": [
                        {"id": "img_001", "path": temp_image}
                    ]
                },
            )
        ]
        content = assemble_multimodal_response(results, "Markdown 内容")
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image"

    def test_deduplicates_images(self, temp_image):
        """应去重相同 image_id 的图片。"""
        results = [
            RetrievalResult(
                chunk_id="c1",
                score=0.9,
                text="文本1",
                metadata={"images": [{"id": "img_001", "path": temp_image}]},
            ),
            RetrievalResult(
                chunk_id="c2",
                score=0.8,
                text="文本2",
                metadata={"images": [{"id": "img_001", "path": temp_image}]},
            ),
        ]
        content = assemble_multimodal_response(results, "Markdown")
        # 只有 1 个图片（去重后）
        assert len(content) == 2

    def test_max_images_limit(self, temp_image):
        """应限制最大图片数量。"""
        images = [{"id": f"img_{i:03d}", "path": temp_image} for i in range(10)]
        results = [
            RetrievalResult(
                chunk_id="c1",
                score=0.9,
                text="测试",
                metadata={"images": images},
            )
        ]
        content = assemble_multimodal_response(results, "Markdown", max_images=3)
        # 1 个文本 + 3 个图片
        assert len(content) == 4

    def test_image_content_structure(self, temp_image):
        """ImageContent 应包含正确的字段。"""
        results = [
            RetrievalResult(
                chunk_id="c1",
                score=0.9,
                text="测试",
                metadata={"images": [{"id": "img_001", "path": temp_image}]},
            )
        ]
        content = assemble_multimodal_response(results, "Markdown")
        image = content[1]
        assert image["type"] == "image"
        assert image["mimeType"] == "image/png"
        assert "data" in image
        # 验证 base64 有效
        decoded = base64.b64decode(image["data"])
        assert len(decoded) > 0

    def test_missing_image_path_ignored(self):
        """缺少 path 的图片引用应被忽略。"""
        results = [
            RetrievalResult(
                chunk_id="c1",
                score=0.9,
                text="测试",
                metadata={"images": [{"id": "img_001"}]},  # 没有 path
            )
        ]
        content = assemble_multimodal_response(results, "Markdown")
        assert len(content) == 1  # 只有文本

    def test_nonexistent_image_ignored(self):
        """不存在的图片文件应被忽略。"""
        results = [
            RetrievalResult(
                chunk_id="c1",
                score=0.9,
                text="测试",
                metadata={"images": [{"id": "img_001", "path": "/nonexistent.png"}]},
            )
        ]
        content = assemble_multimodal_response(results, "Markdown")
        assert len(content) == 1  # 只有文本
