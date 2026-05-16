"""C7 ImageCaptioner 集成测试（真实 Vision LLM 调用）。

使用 config/settings.yaml 中配置的 Vision LLM 进行真实图片描述生成。
验证 Vision LLM 配置正确性、caption 输出质量、降级机制。

⚠️ 会产生真实 API 调用与费用。
"""

from __future__ import annotations

import os
import sys

import pytest

from src.core.settings import load_settings


def _safe_print(text: str):
    """安全打印，处理 Windows 终端 GBK 编码问题。"""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))
from src.core.types import Chunk, ImageRef, make_image_placeholder
from src.ingestion.transform.image_captioner import ImageCaptioner
# 导入 OpenAI Vision LLM 以触发 @register_vision_llm 注册
from src.libs.llm.openai_vision_llm import OpenAIVisionLLM  # noqa: F401
from src.libs.llm.llm_factory import LLMFactory


def _make_chunk_with_image(
    image_path: str,
    chunk_id: str = "integ_0000_abcd1234",
    image_id: str = "test_img_001",
) -> Chunk:
    """创建含图片引用的测试用 Chunk。

    参数:
        image_path: 图片文件路径
        chunk_id: chunk ID
        image_id: 图片 ID
    """
    placeholder = make_image_placeholder(image_id)
    return Chunk(
        id=chunk_id,
        text=f"这是一段测试文本。{placeholder}",
        metadata={
            "source_path": "/test.pdf",
            "chunk_index": 0,
            "images": [
                {
                    "id": image_id,
                    "path": image_path,
                    "page": 1,
                    "text_offset": 0,
                    "text_length": len(placeholder),
                    "position": {},
                }
            ],
            "image_refs": [image_id],
        },
        source_ref="doc_001",
    )


@pytest.mark.integration
class TestImageCaptionerLLMIntegration:
    """ImageCaptioner 真实 Vision LLM 集成测试。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """加载配置并创建 Vision LLM 实例。"""
        self.settings = load_settings()
        self.vision_llm = LLMFactory.create_vision_llm(self.settings.vision_llm)

    def test_caption_generated_for_real_image(self):
        """对真实图片生成 caption，验证输出非空且有意义。"""
        # 使用 data/images 下的真实图片
        image_path = os.path.join("data", "images", "test01.jpg")
        if not os.path.exists(image_path):
            pytest.skip(f"测试图片不存在: {image_path}")

        captioner = ImageCaptioner(
            settings=self.settings,
            vision_llm=self.vision_llm,
        )
        chunk = _make_chunk_with_image(image_path)
        result = captioner.transform([chunk])

        # caption 应已生成
        assert "image_captions" in result[0].metadata
        caption = result[0].metadata["image_captions"]["test_img_001"]
        assert len(caption) > 10, f"caption 过短: {caption}"
        _safe_print(f"\n生成的 caption: {caption}")

        # 占位符应被替换
        assert make_image_placeholder("test_img_001") not in result[0].text
        assert caption in result[0].text

        # 不应标记未处理
        assert "has_unprocessed_images" not in result[0].metadata

    def test_caption_for_second_image(self):
        """对第二张测试图片生成 caption。"""
        # 使用第二张测试图片
        image_path = os.path.join("data", "images", "test02.jpg")
        if not os.path.exists(image_path):
            pytest.skip(f"测试图片不存在: {image_path}")

        captioner = ImageCaptioner(
            settings=self.settings,
            vision_llm=self.vision_llm,
        )
        chunk = _make_chunk_with_image(image_path, image_id="pdf_img_001")
        result = captioner.transform([chunk])

        assert "image_captions" in result[0].metadata
        caption = result[0].metadata["image_captions"]["pdf_img_001"]
        assert len(caption) > 5
        _safe_print(f"\n第二张图片 caption: {caption}")

    def test_degradation_on_invalid_vision_model(self):
        """无效 Vision LLM 配置时优雅降级，不崩溃。"""
        from src.core.settings import VisionLLMConfig
        from unittest.mock import MagicMock

        # 创建使用无效配置的 settings
        bad_settings = MagicMock()
        bad_settings.vision_llm = VisionLLMConfig(
            provider="openai",
            model="nonexistent-vision-model",
            api_key="invalid-key",
            base_url=self.settings.vision_llm.base_url,
        )

        captioner = ImageCaptioner(
            settings=bad_settings,
            use_vision_llm=True,
        )
        chunk = _make_chunk_with_image("/nonexistent/image.png")
        result = captioner.transform([chunk])

        # 降级成功，不崩溃
        assert len(result) == 1
        assert result[0].metadata["has_unprocessed_images"] is True
        # image_refs 应保留
        assert result[0].metadata["image_refs"] == ["test_img_001"]
        _safe_print("\n降级测试通过，未崩溃")

    def test_no_image_chunk_passthrough(self):
        """无图片的 chunk 直接通过，不调用 Vision LLM。"""
        captioner = ImageCaptioner(
            settings=self.settings,
            vision_llm=self.vision_llm,
        )
        chunk = Chunk(
            id="no_img_0000",
            text="这是一段纯文本，没有图片引用。",
            metadata={"source_path": "/test.pdf", "chunk_index": 0},
            source_ref="doc_001",
        )
        result = captioner.transform([chunk])

        assert len(result) == 1
        assert result[0].text == "这是一段纯文本，没有图片引用。"
        assert "image_captions" not in result[0].metadata
        _safe_print("\n无图片 chunk 测试通过")
