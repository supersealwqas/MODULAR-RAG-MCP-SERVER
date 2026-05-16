"""C7 ImageCaptioner 单元测试。

使用 Mock Vision LLM 隔离测试，覆盖启用模式、降级模式、异常处理等。
验收标准：所有测试通过。
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.core.settings import Settings, VisionLLMConfig
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk, ImageRef, make_image_placeholder
from src.ingestion.transform.base_transform import BaseTransform
from src.ingestion.transform.image_captioner import ImageCaptioner
from src.libs.llm.base_llm import LLMResponse


# ============================================================
# 测试辅助函数
# ============================================================


def _make_chunk_with_images(
    chunk_id: str = "test_0000_abcd1234",
    image_ids: list[str] | None = None,
    text: str = "这是一段文本。",
) -> Chunk:
    """创建含图片引用的测试用 Chunk。

    参数:
        chunk_id: chunk ID
        image_ids: 图片 ID 列表
        text: chunk 文本（可含占位符）
    """
    if image_ids is None:
        image_ids = ["img_001"]

    # 构建图片引用列表
    images = []
    for img_id in image_ids:
        images.append({
            "id": img_id,
            "path": f"/tmp/images/{img_id}.png",
            "page": 1,
            "text_offset": 0,
            "text_length": 0,
            "position": {},
        })

    # 在文本中插入占位符
    for img_id in image_ids:
        text = text + " " + make_image_placeholder(img_id)

    return Chunk(
        id=chunk_id,
        text=text,
        metadata={
            "source_path": "/test.pdf",
            "chunk_index": 0,
            "images": images,
            "image_refs": image_ids,
        },
        source_ref="doc_001",
    )


def _make_chunk_no_images(chunk_id: str = "test_0000_abcd1234") -> Chunk:
    """创建无图片引用的测试用 Chunk。"""
    return Chunk(
        id=chunk_id,
        text="这是一段纯文本。",
        metadata={"source_path": "/test.pdf", "chunk_index": 0},
        source_ref="doc_001",
    )


def _make_settings_stub() -> Settings:
    """创建测试用 Settings stub。"""
    return Settings(
        llm=MagicMock(),
        vision_llm=VisionLLMConfig(provider="openai", model="test-vision", api_key="test-key"),
        ollama=MagicMock(),
        embedding=MagicMock(),
        splitter=MagicMock(),
        vector_store=MagicMock(),
        retrieval=MagicMock(),
        rerank=MagicMock(),
        evaluation=MagicMock(),
        observability=MagicMock(),
    )


def _make_mock_vision_llm(response: str = "图片中展示了一个图表") -> MagicMock:
    """创建 Mock Vision LLM 实例。"""
    vision_llm = MagicMock()
    vision_llm.chat_with_image.return_value = LLMResponse(
        content=response,
        model="test-vision",
    )
    return vision_llm


def _make_temp_image() -> str:
    """创建临时图片文件并返回路径。"""
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    # 写入最小 PNG 数据
    with open(path, "wb") as f:
        # 最小合法 PNG（1x1 像素透明图片）
        f.write(
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
            b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
            b"\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
    return path


# ============================================================
# 测试：BaseTransform 抽象基类
# ============================================================


class TestBaseTransformInheritance:
    """ImageCaptioner 继承关系测试。"""

    def test_is_subclass_of_base_transform(self):
        """ImageCaptioner 是 BaseTransform 的子类。"""
        assert issubclass(ImageCaptioner, BaseTransform)

    def test_can_instantiate(self):
        """ImageCaptioner 可以正常实例化。"""
        settings = _make_settings_stub()
        captioner = ImageCaptioner(settings=settings)
        assert isinstance(captioner, BaseTransform)

    def test_default_use_vision_llm_true(self):
        """默认 use_vision_llm=True。"""
        captioner = ImageCaptioner(settings=_make_settings_stub())
        assert captioner.use_vision_llm is True


# ============================================================
# 测试：启用模式（Mock Vision LLM）
# ============================================================


class TestEnabledMode:
    """启用 Vision LLM 模式测试。"""

    def test_caption_generated_for_chunk_with_images(self):
        """存在 image_refs 时应生成 caption 并写入 metadata。"""
        temp_path = _make_temp_image()
        try:
            mock_vllm = _make_mock_vision_llm("这是一张流程图")
            settings = _make_settings_stub()
            captioner = ImageCaptioner(settings=settings, vision_llm=mock_vllm)

            chunk = _make_chunk_with_images(image_ids=["img_001"])
            # 覆盖图片路径为临时文件
            chunk.metadata["images"][0]["path"] = temp_path

            result = captioner.transform([chunk])
            assert len(result) == 1
            assert "image_captions" in result[0].metadata
            assert result[0].metadata["image_captions"]["img_001"] == "这是一张流程图"
        finally:
            os.unlink(temp_path)

    def test_caption_replaces_placeholder_in_text(self):
        """caption 应替换文本中的图片占位符。"""
        temp_path = _make_temp_image()
        try:
            mock_vllm = _make_mock_vision_llm("图表描述")
            captioner = ImageCaptioner(settings=_make_settings_stub(), vision_llm=mock_vllm)

            chunk = _make_chunk_with_images(image_ids=["img_001"], text="前面的文字。")
            chunk.metadata["images"][0]["path"] = temp_path

            result = captioner.transform([chunk])
            # 占位符应被替换
            assert make_image_placeholder("img_001") not in result[0].text
            assert "图表描述" in result[0].text
        finally:
            os.unlink(temp_path)

    def test_vision_llm_called_with_correct_args(self):
        """应使用正确的 prompt 和图片路径调用 Vision LLM。"""
        temp_path = _make_temp_image()
        try:
            mock_vllm = _make_mock_vision_llm("描述")
            captioner = ImageCaptioner(settings=_make_settings_stub(), vision_llm=mock_vllm)

            chunk = _make_chunk_with_images(image_ids=["img_001"])
            chunk.metadata["images"][0]["path"] = temp_path

            captioner.transform([chunk])

            mock_vllm.chat_with_image.assert_called_once()
            call_kwargs = mock_vllm.chat_with_image.call_args
            assert call_kwargs.kwargs["image"] == temp_path
            assert len(call_kwargs.kwargs["text"]) > 0
        finally:
            os.unlink(temp_path)

    def test_multiple_images_in_one_chunk(self):
        """单个 chunk 含多张图片时应逐张生成 caption。"""
        temp_path1 = _make_temp_image()
        temp_path2 = _make_temp_image()
        try:
            responses = ["第一张图描述", "第二张图描述"]
            mock_vllm = MagicMock()
            mock_vllm.chat_with_image.side_effect = [
                LLMResponse(content=r, model="test") for r in responses
            ]
            captioner = ImageCaptioner(settings=_make_settings_stub(), vision_llm=mock_vllm)

            chunk = _make_chunk_with_images(image_ids=["img_001", "img_002"])
            chunk.metadata["images"][0]["path"] = temp_path1
            chunk.metadata["images"][1]["path"] = temp_path2

            result = captioner.transform([chunk])

            assert mock_vllm.chat_with_image.call_count == 2
            assert result[0].metadata["image_captions"]["img_001"] == "第一张图描述"
            assert result[0].metadata["image_captions"]["img_002"] == "第二张图描述"
        finally:
            os.unlink(temp_path1)
            os.unlink(temp_path2)

    def test_no_unprocessed_images_flag_on_success(self):
        """全部成功时不应标记 has_unprocessed_images。"""
        temp_path = _make_temp_image()
        try:
            mock_vllm = _make_mock_vision_llm("描述")
            captioner = ImageCaptioner(settings=_make_settings_stub(), vision_llm=mock_vllm)

            chunk = _make_chunk_with_images(image_ids=["img_001"])
            chunk.metadata["images"][0]["path"] = temp_path

            result = captioner.transform([chunk])
            assert "has_unprocessed_images" not in result[0].metadata
        finally:
            os.unlink(temp_path)


# ============================================================
# 测试：降级模式
# ============================================================


class TestFallbackMode:
    """降级模式测试（Vision LLM 不可用或异常）。"""

    def test_no_images_chunk_passes_through(self):
        """无图片引用的 chunk 应直接通过，不做任何处理。"""
        mock_vllm = _make_mock_vision_llm()
        captioner = ImageCaptioner(settings=_make_settings_stub(), vision_llm=mock_vllm)

        chunk = _make_chunk_no_images()
        result = captioner.transform([chunk])

        assert len(result) == 1
        assert result[0].text == "这是一段纯文本。"
        mock_vllm.chat_with_image.assert_not_called()

    def test_disabled_mode_marks_unprocessed(self):
        """use_vision_llm=False 时应标记 has_unprocessed_images。"""
        captioner = ImageCaptioner(settings=_make_settings_stub(), use_vision_llm=False)

        chunk = _make_chunk_with_images()
        result = captioner.transform([chunk])

        assert result[0].metadata["has_unprocessed_images"] is True
        # image_refs 应保留
        assert result[0].metadata["image_refs"] == ["img_001"]

    def test_vision_llm_creation_failure_fallback(self):
        """Vision LLM 创建失败时应降级，标记 has_unprocessed_images。"""
        settings = _make_settings_stub()
        captioner = ImageCaptioner(settings=settings, use_vision_llm=True)

        # patch LLMFactory.create_vision_llm 抛异常
        with patch(
            "src.ingestion.transform.image_captioner.LLMFactory.create_vision_llm",
            side_effect=ValueError("未知的 Vision LLM 提供者"),
        ):
            chunk = _make_chunk_with_images()
            result = captioner.transform([chunk])

        assert result[0].metadata["has_unprocessed_images"] is True
        assert result[0].metadata["image_refs"] == ["img_001"]

    def test_image_file_not_found_fallback(self):
        """图片文件不存在时应降级，标记 has_unprocessed_images。"""
        mock_vllm = _make_mock_vision_llm()
        captioner = ImageCaptioner(settings=_make_settings_stub(), vision_llm=mock_vllm)

        chunk = _make_chunk_with_images()
        # 图片路径不存在（默认 /tmp/images/img_001.png 不存在）
        result = captioner.transform([chunk])

        assert result[0].metadata["has_unprocessed_images"] is True

    def test_vision_llm_api_error_fallback(self):
        """Vision LLM API 调用异常时应降级。"""
        temp_path = _make_temp_image()
        try:
            mock_vllm = MagicMock()
            mock_vllm.chat_with_image.side_effect = RuntimeError("API 超时")
            captioner = ImageCaptioner(settings=_make_settings_stub(), vision_llm=mock_vllm)

            chunk = _make_chunk_with_images()
            chunk.metadata["images"][0]["path"] = temp_path

            result = captioner.transform([chunk])
            assert result[0].metadata["has_unprocessed_images"] is True
        finally:
            os.unlink(temp_path)

    def test_empty_response_fallback(self):
        """Vision LLM 返回空内容时应降级。"""
        temp_path = _make_temp_image()
        try:
            mock_vllm = MagicMock()
            mock_vllm.chat_with_image.return_value = LLMResponse(content="", model="test")
            captioner = ImageCaptioner(settings=_make_settings_stub(), vision_llm=mock_vllm)

            chunk = _make_chunk_with_images()
            chunk.metadata["images"][0]["path"] = temp_path

            result = captioner.transform([chunk])
            assert result[0].metadata["has_unprocessed_images"] is True
        finally:
            os.unlink(temp_path)

    def test_image_refs_preserved_on_fallback(self):
        """降级时 image_refs 应完整保留。"""
        captioner = ImageCaptioner(settings=_make_settings_stub(), use_vision_llm=False)

        chunk = _make_chunk_with_images(image_ids=["img_001", "img_002"])
        result = captioner.transform([chunk])

        assert result[0].metadata["image_refs"] == ["img_001", "img_002"]
        assert result[0].metadata["has_unprocessed_images"] is True

    def test_partial_failure_marks_unprocessed(self):
        """部分图片成功、部分失败时应标记 has_unprocessed_images。"""
        temp_path1 = _make_temp_image()
        temp_path2 = _make_temp_image()
        try:
            mock_vllm = MagicMock()
            # 第一张成功，第二张失败
            mock_vllm.chat_with_image.side_effect = [
                LLMResponse(content="成功描述", model="test"),
                RuntimeError("API 错误"),
            ]
            captioner = ImageCaptioner(settings=_make_settings_stub(), vision_llm=mock_vllm)

            chunk = _make_chunk_with_images(image_ids=["img_001", "img_002"])
            chunk.metadata["images"][0]["path"] = temp_path1
            chunk.metadata["images"][1]["path"] = temp_path2

            result = captioner.transform([chunk])
            # 成功的 caption 应保留
            assert result[0].metadata["image_captions"]["img_001"] == "成功描述"
            # 标记有未处理图片
            assert result[0].metadata["has_unprocessed_images"] is True
        finally:
            os.unlink(temp_path1)
            os.unlink(temp_path2)


# ============================================================
# 测试：异常处理
# ============================================================


class TestErrorHandling:
    """异常处理测试。"""

    def test_single_chunk_error_preserves_others(self):
        """单个 chunk 处理异常不影响其他 chunk。"""
        mock_vllm = MagicMock()
        call_count = [0]

        def side_effect(text, image):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("boom")
            return LLMResponse(content="描述", model="test")

        mock_vllm.chat_with_image.side_effect = side_effect

        temp_paths = [_make_temp_image() for _ in range(3)]
        try:
            captioner = ImageCaptioner(settings=_make_settings_stub(), vision_llm=mock_vllm)

            chunks = []
            for i, path in enumerate(temp_paths):
                c = _make_chunk_with_images(chunk_id=f"c{i}", image_ids=[f"img_{i:03d}"])
                c.metadata["images"][0]["path"] = path
                chunks.append(c)

            result = captioner.transform(chunks)
            assert len(result) == 3
        finally:
            for p in temp_paths:
                os.unlink(p)

    def test_unexpected_error_marks_metadata(self):
        """处理异常时应标记 has_unprocessed_images 和 image_caption_error。"""
        captioner = ImageCaptioner(settings=_make_settings_stub(), vision_llm=MagicMock())

        chunk = _make_chunk_with_images()
        # patch _process_single 抛异常
        with patch.object(captioner, '_process_single', side_effect=ValueError("unexpected")):
            result = captioner.transform([chunk])

        assert result[0].metadata["has_unprocessed_images"] is True
        assert "unexpected" in result[0].metadata["image_caption_error"]

    def test_empty_chunks_returns_empty(self):
        """空列表输入返回空列表。"""
        captioner = ImageCaptioner(settings=_make_settings_stub())
        result = captioner.transform([])
        assert result == []


# ============================================================
# 测试：配置开关
# ============================================================


class TestConfig:
    """配置驱动行为测试。"""

    def test_use_vision_llm_false_skips_call(self):
        """use_vision_llm=False 时不调用 Vision LLM。"""
        mock_vllm = _make_mock_vision_llm()
        captioner = ImageCaptioner(settings=_make_settings_stub(), vision_llm=mock_vllm, use_vision_llm=False)

        chunk = _make_chunk_with_images()
        captioner.transform([chunk])

        mock_vllm.chat_with_image.assert_not_called()

    def test_use_vision_llm_true_calls_vision_llm(self):
        """use_vision_llm=True 且有图片时调用 Vision LLM。"""
        temp_path = _make_temp_image()
        try:
            mock_vllm = _make_mock_vision_llm("描述")
            captioner = ImageCaptioner(settings=_make_settings_stub(), vision_llm=mock_vllm, use_vision_llm=True)

            chunk = _make_chunk_with_images()
            chunk.metadata["images"][0]["path"] = temp_path

            captioner.transform([chunk])
            mock_vllm.chat_with_image.assert_called_once()
        finally:
            os.unlink(temp_path)


# ============================================================
# 测试：Prompt 加载
# ============================================================


class TestPromptLoading:
    """Prompt 模板加载测试。"""

    def test_load_default_prompt(self):
        """默认路径加载 prompt 模板。"""
        captioner = ImageCaptioner(settings=_make_settings_stub())
        # 默认 prompt 应包含中文关键词
        assert "描述" in captioner.prompt_template or "图片" in captioner.prompt_template

    def test_load_custom_prompt(self):
        """自定义路径加载 prompt 模板。"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("自定义图片描述 prompt")
            custom_path = f.name
        try:
            captioner = ImageCaptioner(settings=_make_settings_stub(), prompt_path=custom_path)
            assert "自定义图片描述 prompt" in captioner.prompt_template
        finally:
            os.unlink(custom_path)

    def test_load_prompt_fallback(self):
        """文件不存在时使用内置 fallback。"""
        captioner = ImageCaptioner(settings=_make_settings_stub(), prompt_path="/nonexistent/path.txt")
        assert "描述" in captioner.prompt_template or "图片" in captioner.prompt_template


# ============================================================
# 测试：Trace 记录
# ============================================================


class TestTraceRecording:
    """Trace 阶段记录测试。"""

    def test_trace_records_fallback_stage(self):
        """降级模式下 trace 记录 image_caption 阶段（method=fallback）。"""
        captioner = ImageCaptioner(settings=_make_settings_stub(), use_vision_llm=False)
        trace = TraceContext()

        chunk = _make_chunk_with_images()
        captioner.transform([chunk], trace=trace)

        caption_stages = [s for s in trace.stages if s.get("name") == "image_caption"]
        assert len(caption_stages) > 0
        assert caption_stages[0]["method"] == "fallback"

    def test_trace_records_success_stage(self):
        """成功模式下 trace 记录 image_caption 阶段（method=vision_llm）。"""
        temp_path = _make_temp_image()
        try:
            mock_vllm = _make_mock_vision_llm("描述")
            captioner = ImageCaptioner(settings=_make_settings_stub(), vision_llm=mock_vllm)
            trace = TraceContext()

            chunk = _make_chunk_with_images()
            chunk.metadata["images"][0]["path"] = temp_path

            captioner.transform([chunk], trace=trace)

            caption_stages = [s for s in trace.stages if s.get("name") == "image_caption"]
            assert len(caption_stages) > 0
            assert caption_stages[0]["method"] == "vision_llm"
            assert caption_stages[0]["captioned"] == 1
        finally:
            os.unlink(temp_path)


# ============================================================
# 测试：Transform 接口契约
# ============================================================


class TestTransformContract:
    """Transform 接口契约测试。"""

    def test_transform_preserves_chunk_count(self):
        """transform 保持 chunk 数量不变。"""
        captioner = ImageCaptioner(settings=_make_settings_stub(), use_vision_llm=False)
        chunks = [
            _make_chunk_with_images(chunk_id="c1"),
            _make_chunk_no_images(chunk_id="c2"),
            _make_chunk_with_images(chunk_id="c3"),
        ]
        result = captioner.transform(chunks)
        assert len(result) == 3

    def test_transform_preserves_chunk_ids(self):
        """transform 保持 chunk ID 不变。"""
        captioner = ImageCaptioner(settings=_make_settings_stub(), use_vision_llm=False)
        chunks = [
            _make_chunk_with_images(chunk_id="id1"),
            _make_chunk_no_images(chunk_id="id2"),
        ]
        result = captioner.transform(chunks)
        assert result[0].id == "id1"
        assert result[1].id == "id2"

    def test_transform_preserves_existing_metadata(self):
        """transform 保持原有 metadata 字段。"""
        captioner = ImageCaptioner(settings=_make_settings_stub(), use_vision_llm=False)
        chunk = _make_chunk_no_images()
        chunk.metadata["custom_field"] = 42
        result = captioner.transform([chunk])
        assert result[0].metadata["custom_field"] == 42
