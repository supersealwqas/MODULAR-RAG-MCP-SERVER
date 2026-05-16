"""C7 ImageCaptioner 手动测试脚本。

测试 ImageCaptioner 的图片描述生成、降级机制、Trace 记录等。
使用真实 Vision LLM 进行测试。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.settings import load_settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk, make_image_placeholder
from src.ingestion.transform.image_captioner import ImageCaptioner
# 导入 OpenAI Vision LLM 以触发 @register_vision_llm 注册
from src.libs.llm.openai_vision_llm import OpenAIVisionLLM  # noqa: F401
from src.libs.llm.llm_factory import LLMFactory


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def safe_print(text: str):
    """安全打印，忽略无法编码的字符。"""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("gbk", errors="replace").decode("gbk"))


def _make_chunk_with_image(
    image_path: str,
    chunk_id: str = "test_0000_abcd1234",
    image_id: str = "test_img_001",
    text: str = "这是一段测试文本。",
) -> Chunk:
    """创建含图片引用的测试用 Chunk。"""
    placeholder = make_image_placeholder(image_id)
    return Chunk(
        id=chunk_id,
        text=f"{text}{placeholder}",
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


def test_caption_generation():
    """测试真实图片描述生成。"""
    section("图片描述生成测试")

    settings = load_settings()
    try:
        vision_llm = LLMFactory.create_vision_llm(settings.vision_llm)
    except Exception as e:
        print(f"Vision LLM 创建失败: {e}")
        return

    captioner = ImageCaptioner(settings=settings, vision_llm=vision_llm)

    # 测试图片列表
    test_images = [
        ("data/images/test01.jpg", "img_001"),
        ("data/images/test02.jpg", "img_002"),
        ("data/images/test03.png", "img_003"),
    ]

    for image_path, image_id in test_images:
        if not os.path.exists(image_path):
            safe_print(f"  跳过（文件不存在）: {image_path}")
            continue

        safe_print(f"\n  测试图片: {image_path}")
        chunk = _make_chunk_with_image(image_path, image_id=image_id)
        result = captioner.transform([chunk])

        if "image_captions" in result[0].metadata:
            caption = result[0].metadata["image_captions"][image_id]
            safe_print(f"  Caption: {caption[:100]}...")
            safe_print(f"  占位符已替换: {make_image_placeholder(image_id) not in result[0].text}")
        else:
            safe_print(f"  未生成 caption (可能降级)")
            safe_print(f"  has_unprocessed_images: {result[0].metadata.get('has_unprocessed_images')}")


def test_multi_image_chunk():
    """测试单个 chunk 含多张图片。"""
    section("多图片 Chunk 测试")

    settings = load_settings()
    try:
        vision_llm = LLMFactory.create_vision_llm(settings.vision_llm)
    except Exception as e:
        print(f"Vision LLM 创建失败: {e}")
        return

    captioner = ImageCaptioner(settings=settings, vision_llm=vision_llm)

    # 找到可用的图片
    available_images = []
    for path in [
        "data/images/test01.jpg",
        "data/images/test02.jpg",
    ]:
        if os.path.exists(path):
            available_images.append(path)

    if len(available_images) < 2:
        print("  可用图片不足 2 张，跳过")
        return

    # 构建含两张图片的 chunk
    image_refs = []
    images_meta = []
    text_parts = ["文档内容："]
    for i, img_path in enumerate(available_images[:2]):
        img_id = f"multi_img_{i:03d}"
        placeholder = make_image_placeholder(img_id)
        image_refs.append(img_id)
        images_meta.append({
            "id": img_id,
            "path": img_path,
            "page": i + 1,
            "text_offset": 0,
            "text_length": len(placeholder),
            "position": {},
        })
        text_parts.append(f"图{i+1}：{placeholder}")

    chunk = Chunk(
        id="multi_0000_abcd1234",
        text="\n".join(text_parts),
        metadata={
            "source_path": "/test.pdf",
            "chunk_index": 0,
            "images": images_meta,
            "image_refs": image_refs,
        },
        source_ref="doc_001",
    )

    result = captioner.transform([chunk])
    captions = result[0].metadata.get("image_captions", {})

    print(f"  图片数量: {len(available_images[:2])}")
    print(f"  生成 caption 数: {len(captions)}")
    for img_id, caption in captions.items():
        safe_print(f"  [{img_id}]: {caption[:80]}...")


def test_fallback_disabled():
    """测试禁用 Vision LLM 时的降级行为。"""
    section("降级测试（禁用 Vision LLM）")

    settings = load_settings()
    captioner = ImageCaptioner(settings=settings, use_vision_llm=False)

    chunk = _make_chunk_with_image(
        "data/images/test01.jpg",
        image_id="fallback_img",
    )
    result = captioner.transform([chunk])

    print(f"  has_unprocessed_images: {result[0].metadata.get('has_unprocessed_images')}")
    print(f"  image_refs 保留: {result[0].metadata.get('image_refs')}")
    print(f"  无 image_captions: {'image_captions' not in result[0].metadata}")


def test_fallback_error():
    """测试 Vision LLM 异常时的降级行为。"""
    section("降级测试（Vision LLM 异常）")

    settings = load_settings()

    # 使用无效配置触发降级
    from unittest.mock import MagicMock
    from src.core.settings import VisionLLMConfig

    bad_settings = MagicMock()
    bad_settings.vision_llm = VisionLLMConfig(
        provider="openai",
        model="nonexistent-model",
        api_key="invalid",
        base_url=settings.vision_llm.base_url,
    )

    captioner = ImageCaptioner(settings=bad_settings, use_vision_llm=True)
    chunk = _make_chunk_with_image("/nonexistent/image.png", image_id="error_img")
    result = captioner.transform([chunk])

    print(f"  has_unprocessed_images: {result[0].metadata.get('has_unprocessed_images')}")
    print(f"  image_refs 保留: {result[0].metadata.get('image_refs')}")
    print(f"  降级成功，未崩溃: True")


def test_trace_recording():
    """测试 Trace 记录。"""
    section("Trace 记录测试")

    settings = load_settings()
    try:
        vision_llm = LLMFactory.create_vision_llm(settings.vision_llm)
    except Exception as e:
        print(f"Vision LLM 创建失败: {e}")
        return

    captioner = ImageCaptioner(settings=settings, vision_llm=vision_llm)
    trace = TraceContext(trace_type="ingestion")

    image_path = "data/images/test01.jpg"
    if not os.path.exists(image_path):
        print(f"  测试图片不存在: {image_path}")
        return

    chunk = _make_chunk_with_image(image_path, image_id="trace_img")
    captioner.transform([chunk], trace=trace)

    print(f"  trace_id: {trace.trace_id}")
    print(f"  stages: {len(trace.stages)}")
    for stage in trace.stages:
        safe_print(f"  - {stage['name']} method={stage.get('method')} "
                   f"captioned={stage.get('captioned', '-')}")


def test_prompt_loading():
    """测试 Prompt 加载。"""
    section("Prompt 加载测试")

    settings = load_settings()
    captioner = ImageCaptioner(settings=settings)

    print(f"  Prompt 长度: {len(captioner.prompt_template)}")
    print(f"  包含'描述': {'描述' in captioner.prompt_template}")
    print(f"  包含'图片': {'图片' in captioner.prompt_template}")
    safe_print(f"  内容预览: {captioner.prompt_template[:80]}...")


def main():
    test_prompt_loading()
    test_caption_generation()
    test_multi_image_chunk()
    test_fallback_disabled()
    test_fallback_error()
    test_trace_recording()
    print(f"\n{'='*60}")
    print("  全部手动测试完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
