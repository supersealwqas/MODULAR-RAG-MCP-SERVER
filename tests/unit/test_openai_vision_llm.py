"""测试 OpenAI Vision LLM 实现。

使用 mock 测试 API 调用和图片处理逻辑，不走真实网络。
"""

import io
import sys
import base64
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.settings import VisionLLMConfig
from src.libs.llm.base_llm import LLMResponse
from src.libs.llm.base_vision_llm import BaseVisionLLM
from src.libs.llm.llm_factory import LLMFactory, _VISION_LLM_REGISTRY
from src.libs.llm.openai_vision_llm import OpenAIVisionLLM


# ─────────────────────────────────────────────
# 测试工厂路由
# ─────────────────────────────────────────────

@pytest.mark.unit
class TestOpenAIVisionLLMFactory:
    """测试 OpenAI Vision LLM 的工厂路由。"""

    def test_factory_creates_openai_vision(self):
        """provider=openai 时工厂应创建 OpenAIVisionLLM 实例。"""
        config = VisionLLMConfig(
            provider="openai",
            model="mimo-v2.5",
            api_key="sk-test",
            base_url="https://example.com/v1",
        )
        llm = LLMFactory.create_vision_llm(config)
        assert isinstance(llm, OpenAIVisionLLM)
        assert llm.model == "mimo-v2.5"
        assert llm.base_url == "https://example.com/v1"

    def test_registered_in_vision_registry(self):
        """openai 提供者应已注册在 Vision LLM 注册表中。"""
        assert "openai" in _VISION_LLM_REGISTRY
        assert _VISION_LLM_REGISTRY["openai"] is OpenAIVisionLLM

    def test_factory_passes_all_config(self):
        """所有配置字段应正确传递给实例。"""
        config = VisionLLMConfig(
            provider="openai",
            model="mimo-v2.5",
            api_key="sk-test",
            base_url="https://example.com/v1",
            temperature=0.7,
            max_tokens=2048,
            max_image_size=1024,
        )
        llm = LLMFactory.create_vision_llm(config)
        assert llm.model == "mimo-v2.5"
        assert llm.api_key == "sk-test"
        assert llm.base_url == "https://example.com/v1"
        assert llm.temperature == 0.7
        assert llm.max_tokens == 2048
        assert llm.max_image_size == 1024

    def test_factory_case_insensitive(self):
        """工厂应不区分大小写匹配提供者。"""
        config = VisionLLMConfig(provider="OPENAI", model="test", api_key="sk-test")
        llm = LLMFactory.create_vision_llm(config)
        assert isinstance(llm, OpenAIVisionLLM)


# ─────────────────────────────────────────────
# 测试初始化
# ─────────────────────────────────────────────

@pytest.mark.unit
class TestOpenAIVisionLLMInit:
    """测试 OpenAIVisionLLM 初始化参数。"""

    def test_default_values(self):
        """默认参数应正确设置。"""
        llm = OpenAIVisionLLM(model="mimo-v2.5")
        assert llm.model == "mimo-v2.5"
        assert llm.api_key == ""
        assert llm.base_url == ""
        assert llm.temperature == 0.0
        assert llm.max_tokens == 4096
        assert llm.max_image_size == 2048

    def test_custom_values(self):
        """自定义参数应正确传递。"""
        llm = OpenAIVisionLLM(
            model="gpt-4o",
            api_key="sk-test",
            base_url="https://example.com/v1",
            temperature=0.5,
            max_tokens=1024,
            max_image_size=4096,
        )
        assert llm.model == "gpt-4o"
        assert llm.base_url == "https://example.com/v1"
        assert llm.temperature == 0.5
        assert llm.max_tokens == 1024
        assert llm.max_image_size == 4096

    def test_inherits_from_base_vision_llm(self):
        """应继承自 BaseVisionLLM。"""
        llm = OpenAIVisionLLM(model="test")
        assert isinstance(llm, BaseVisionLLM)


# ─────────────────────────────────────────────
# 辅助工具：构造 mock OpenAI 响应
# ─────────────────────────────────────────────

def _make_mock_openai_response(content="图片描述结果", model="mimo-v2.5"):
    """构造 mock 的 OpenAI API 响应对象。"""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=content))]
    mock_response.model = model
    mock_response.usage = MagicMock(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
    )
    return mock_response


def _make_mock_openai_module():
    """构造 mock 的 openai 模块。"""
    mock_openai_module = MagicMock()
    mock_client = MagicMock()
    mock_openai_module.OpenAI.return_value = mock_client
    return mock_openai_module, mock_client


def _make_test_png_bytes(width=100, height=100):
    """生成测试用 PNG 图片字节。"""
    try:
        from PIL import Image
        img = Image.new("RGB", (width, height), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()
    except ImportError:
        # PIL 不可用时返回最小有效 PNG
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


# ─────────────────────────────────────────────
# 测试 chat_with_image 正常调用
# ─────────────────────────────────────────────

@pytest.mark.unit
class TestChatWithImageNormal:
    """测试 chat_with_image 正常调用场景。"""

    def test_chat_with_bytes_input(self):
        """传入 bytes 图片应正常调用 API 并返回结果。"""
        mock_module, mock_client = _make_mock_openai_module()
        mock_client.chat.completions.create.return_value = _make_mock_openai_response()

        with patch.dict(sys.modules, {"openai": mock_module}):
            llm = OpenAIVisionLLM(model="mimo-v2.5", api_key="sk-test")
            response = llm.chat_with_image("描述这张图片", b"\x89PNG\r\n\x1a\ntest")

            assert isinstance(response, LLMResponse)
            assert response.content == "图片描述结果"
            assert response.model == "mimo-v2.5"
            assert response.usage["prompt_tokens"] == 100
            assert response.usage["total_tokens"] == 120

    def test_chat_with_file_path(self, tmp_path):
        """传入文件路径应正常调用 API。"""
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"\x89PNG\r\n\x1a\ntest")

        mock_module, mock_client = _make_mock_openai_module()
        mock_client.chat.completions.create.return_value = _make_mock_openai_response("路径图片描述")

        with patch.dict(sys.modules, {"openai": mock_module}):
            llm = OpenAIVisionLLM(model="mimo-v2.5", api_key="sk-test")
            response = llm.chat_with_image("描述这张图片", str(img_file))

            assert response.content == "路径图片描述"

    def test_chat_passes_base_url(self):
        """应将 base_url 传递给 OpenAI 客户端。"""
        mock_module, mock_client = _make_mock_openai_module()
        mock_client.chat.completions.create.return_value = _make_mock_openai_response()

        with patch.dict(sys.modules, {"openai": mock_module}):
            llm = OpenAIVisionLLM(
                model="mimo-v2.5",
                api_key="sk-test",
                base_url="https://custom.api.com/v1",
            )
            llm.chat_with_image("测试", b"\x89PNG\r\n\x1a\ntest")

            mock_module.OpenAI.assert_called_once_with(
                api_key="sk-test",
                base_url="https://custom.api.com/v1",
            )

    def test_chat_passes_temperature_and_max_tokens(self):
        """应将 temperature 和 max_tokens 传递给 API 调用。"""
        mock_module, mock_client = _make_mock_openai_module()
        mock_client.chat.completions.create.return_value = _make_mock_openai_response()

        with patch.dict(sys.modules, {"openai": mock_module}):
            llm = OpenAIVisionLLM(
                model="mimo-v2.5",
                api_key="sk-test",
                temperature=0.8,
                max_tokens=1024,
            )
            llm.chat_with_image("测试", b"\x89PNG\r\n\x1a\ntest")

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["temperature"] == 0.8
            assert call_kwargs["max_tokens"] == 1024

    def test_chat_overrides_via_kwargs(self):
        """kwargs 应能覆盖 temperature 和 max_tokens。"""
        mock_module, mock_client = _make_mock_openai_module()
        mock_client.chat.completions.create.return_value = _make_mock_openai_response()

        with patch.dict(sys.modules, {"openai": mock_module}):
            llm = OpenAIVisionLLM(model="mimo-v2.5", api_key="sk-test")
            llm.chat_with_image("测试", b"\x89PNG\r\n\x1a\ntest", temperature=1.5, max_tokens=512)

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["temperature"] == 1.5
            assert call_kwargs["max_tokens"] == 512


# ─────────────────────────────────────────────
# 测试消息格式
# ─────────────────────────────────────────────

@pytest.mark.unit
class TestMessageFormat:
    """测试发送给 OpenAI API 的消息格式。"""

    def test_message_contains_text_and_image(self):
        """消息应包含 text 和 image_url 两种 content 类型。"""
        mock_module, mock_client = _make_mock_openai_module()
        mock_client.chat.completions.create.return_value = _make_mock_openai_response()

        with patch.dict(sys.modules, {"openai": mock_module}):
            llm = OpenAIVisionLLM(model="mimo-v2.5", api_key="sk-test")
            llm.chat_with_image("描述图片", b"\x89PNG\r\n\x1a\ntest")

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            messages = call_kwargs["messages"]
            assert len(messages) == 1
            assert messages[0]["role"] == "user"

            content = messages[0]["content"]
            assert len(content) == 2
            assert content[0]["type"] == "text"
            assert content[0]["text"] == "描述图片"
            assert content[1]["type"] == "image_url"
            assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_image_is_base64_encoded(self):
        """图片应以 base64 编码嵌入 data URI。"""
        mock_module, mock_client = _make_mock_openai_module()
        mock_client.chat.completions.create.return_value = _make_mock_openai_response()
        test_data = b"\x89PNG\r\n\x1a\ntest_image_data"

        with patch.dict(sys.modules, {"openai": mock_module}):
            llm = OpenAIVisionLLM(model="mimo-v2.5", api_key="sk-test")
            llm.chat_with_image("测试", test_data)

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            url = call_kwargs["messages"][0]["content"][1]["image_url"]["url"]
            # 提取 base64 部分
            b64_part = url.split(",", 1)[1]
            decoded = base64.b64decode(b64_part)
            assert decoded == test_data


# ─────────────────────────────────────────────
# 测试图片压缩
# ─────────────────────────────────────────────

@pytest.mark.unit
class TestImageCompression:
    """测试图片自动压缩逻辑。"""

    def test_small_image_not_compressed(self):
        """小于 max_image_size 的图片不应被压缩。"""
        mock_module, mock_client = _make_mock_openai_module()
        mock_client.chat.completions.create.return_value = _make_mock_openai_response()

        with patch.dict(sys.modules, {"openai": mock_module}):
            llm = OpenAIVisionLLM(model="mimo-v2.5", api_key="sk-test", max_image_size=2048)
            # 100x100 的图片，远小于 2048
            small_png = _make_test_png_bytes(100, 100)
            llm.chat_with_image("测试", small_png)

            # 验证 API 调用时图片数据未变
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            url = call_kwargs["messages"][0]["content"][1]["image_url"]["url"]
            b64_part = url.split(",", 1)[1]
            decoded = base64.b64decode(b64_part)
            assert decoded == small_png

    def test_large_image_compressed(self):
        """超过 max_image_size 的图片应被压缩。"""
        mock_module, mock_client = _make_mock_openai_module()
        mock_client.chat.completions.create.return_value = _make_mock_openai_response()

        with patch.dict(sys.modules, {"openai": mock_module}):
            llm = OpenAIVisionLLM(model="mimo-v2.5", api_key="sk-test", max_image_size=512)
            # 4000x3000 的图片，超过 512
            large_png = _make_test_png_bytes(4000, 3000)
            original_size = len(large_png)

            llm.chat_with_image("测试", large_png)

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            url = call_kwargs["messages"][0]["content"][1]["image_url"]["url"]
            b64_part = url.split(",", 1)[1]
            decoded = base64.b64decode(b64_part)

            # 压缩后的图片应更小
            assert len(decoded) < original_size

    def test_compression_preserves_aspect_ratio(self):
        """压缩应保持图片宽高比。"""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow 未安装")

        mock_module, mock_client = _make_mock_openai_module()
        mock_client.chat.completions.create.return_value = _make_mock_openai_response()

        with patch.dict(sys.modules, {"openai": mock_module}):
            llm = OpenAIVisionLLM(model="mimo-v2.5", api_key="sk-test", max_image_size=512)
            large_png = _make_test_png_bytes(2000, 1000)  # 2:1 比例
            llm.chat_with_image("测试", large_png)

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            url = call_kwargs["messages"][0]["content"][1]["image_url"]["url"]
            b64_part = url.split(",", 1)[1]
            decoded = base64.b64decode(b64_part)

            img = Image.open(io.BytesIO(decoded))
            w, h = img.size
            assert max(w, h) <= 512
            # 比例应接近 2:1
            assert abs(w / h - 2.0) < 0.1

    def test_no_pillow_graceful_fallback(self):
        """Pillow 不可用时应使用原始图片，不抛异常。"""
        mock_module, mock_client = _make_mock_openai_module()
        mock_client.chat.completions.create.return_value = _make_mock_openai_response()

        with patch.dict(sys.modules, {"openai": mock_module}):
            llm = OpenAIVisionLLM(model="mimo-v2.5", api_key="sk-test", max_image_size=10)
            original = b"\x89PNG\r\n\x1a\ntest"

            with patch("src.libs.llm.openai_vision_llm.logger") as mock_logger:
                with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
                    # PIL 不可用时应 fallback
                    response = llm.chat_with_image("测试", original)
                    assert isinstance(response, LLMResponse)


# ─────────────────────────────────────────────
# 测试 MIME 类型检测
# ─────────────────────────────────────────────

@pytest.mark.unit
class TestMimeDetection:
    """测试图片 MIME 类型检测。"""

    def test_png_from_extension(self, tmp_path):
        """.png 扩展名应检测为 image/png。"""
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"\x89PNG\r\n\x1a\n")

        mock_module, mock_client = _make_mock_openai_module()
        mock_client.chat.completions.create.return_value = _make_mock_openai_response()

        with patch.dict(sys.modules, {"openai": mock_module}):
            llm = OpenAIVisionLLM(model="test", api_key="sk-test")
            llm.chat_with_image("测试", str(img_file))

            url = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"][1]["image_url"]["url"]
            assert url.startswith("data:image/png;base64,")

    def test_jpg_from_extension(self, tmp_path):
        """.jpg 扩展名应检测为 image/jpeg。"""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"\xff\xd8\xfftest")

        mock_module, mock_client = _make_mock_openai_module()
        mock_client.chat.completions.create.return_value = _make_mock_openai_response()

        with patch.dict(sys.modules, {"openai": mock_module}):
            llm = OpenAIVisionLLM(model="test", api_key="sk-test")
            llm.chat_with_image("测试", str(img_file))

            url = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"][1]["image_url"]["url"]
            assert url.startswith("data:image/jpeg;base64,")

    def test_png_from_magic_bytes(self):
        """无文件路径时从 magic bytes 检测 PNG。"""
        mock_module, mock_client = _make_mock_openai_module()
        mock_client.chat.completions.create.return_value = _make_mock_openai_response()

        with patch.dict(sys.modules, {"openai": mock_module}):
            llm = OpenAIVisionLLM(model="test", api_key="sk-test")
            llm.chat_with_image("测试", b"\x89PNG\r\n\x1a\ntest")

            url = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"][1]["image_url"]["url"]
            assert url.startswith("data:image/png;base64,")

    def test_jpeg_from_magic_bytes(self):
        """从 magic bytes 检测 JPEG。"""
        mock_module, mock_client = _make_mock_openai_module()
        mock_client.chat.completions.create.return_value = _make_mock_openai_response()

        with patch.dict(sys.modules, {"openai": mock_module}):
            llm = OpenAIVisionLLM(model="test", api_key="sk-test")
            llm.chat_with_image("测试", b"\xff\xd8\xfftest")

            url = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"][1]["image_url"]["url"]
            assert url.startswith("data:image/jpeg;base64,")

    def test_unknown_defaults_to_png(self):
        """无法识别格式时默认为 image/png。"""
        mock_module, mock_client = _make_mock_openai_module()
        mock_client.chat.completions.create.return_value = _make_mock_openai_response()

        with patch.dict(sys.modules, {"openai": mock_module}):
            llm = OpenAIVisionLLM(model="test", api_key="sk-test")
            llm.chat_with_image("测试", b"\x00\x01\x02\x03test")

            url = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"][1]["image_url"]["url"]
            assert url.startswith("data:image/png;base64,")


# ─────────────────────────────────────────────
# 测试错误处理
# ─────────────────────────────────────────────

@pytest.mark.unit
class TestErrorHandling:
    """测试异常场景处理。"""

    def test_empty_bytes_raises_value_error(self):
        """空 bytes 应抛出 ValueError。"""
        llm = OpenAIVisionLLM(model="test", api_key="sk-test")
        with pytest.raises(ValueError, match="图片字节数据不能为空"):
            llm.chat_with_image("测试", b"")

    def test_nonexistent_path_raises_file_not_found(self):
        """不存在的文件路径应抛出 FileNotFoundError。"""
        llm = OpenAIVisionLLM(model="test", api_key="sk-test")
        with pytest.raises(FileNotFoundError, match="图片文件不存在"):
            llm.chat_with_image("测试", "/nonexistent/image.png")

    def test_api_connection_error(self):
        """API 连接失败应抛出 RuntimeError。"""
        mock_module, mock_client = _make_mock_openai_module()
        mock_client.chat.completions.create.side_effect = ConnectionError("连接超时")

        with patch.dict(sys.modules, {"openai": mock_module}):
            llm = OpenAIVisionLLM(model="mimo-v2.5", api_key="sk-test")
            with pytest.raises(RuntimeError, match="OpenAI Vision API 调用失败"):
                llm.chat_with_image("测试", b"\x89PNG\r\n\x1a\ntest")

    def test_api_auth_failure(self):
        """认证失败（401）应抛出 RuntimeError 且包含 provider 信息。"""
        mock_module, mock_client = _make_mock_openai_module()
        mock_client.chat.completions.create.side_effect = Exception("401 Unauthorized")

        with patch.dict(sys.modules, {"openai": mock_module}):
            llm = OpenAIVisionLLM(model="mimo-v2.5", api_key="sk-bad-key")
            with pytest.raises(RuntimeError, match="provider=openai"):
                llm.chat_with_image("测试", b"\x89PNG\r\n\x1a\ntest")

    def test_api_timeout(self):
        """超时应抛出 RuntimeError。"""
        mock_module, mock_client = _make_mock_openai_module()
        mock_client.chat.completions.create.side_effect = TimeoutError("请求超时")

        with patch.dict(sys.modules, {"openai": mock_module}):
            llm = OpenAIVisionLLM(model="mimo-v2.5", api_key="sk-test")
            with pytest.raises(RuntimeError, match="OpenAI Vision API 调用失败"):
                llm.chat_with_image("测试", b"\x89PNG\r\n\x1a\ntest")

    def test_error_message_includes_model(self):
        """错误信息应包含模型名称。"""
        mock_module, mock_client = _make_mock_openai_module()
        mock_client.chat.completions.create.side_effect = Exception("服务不可用")

        with patch.dict(sys.modules, {"openai": mock_module}):
            llm = OpenAIVisionLLM(model="mimo-v2.5", api_key="sk-test")
            with pytest.raises(RuntimeError, match="model=mimo-v2.5"):
                llm.chat_with_image("测试", b"\x89PNG\r\n\x1a\ntest")

    def test_openai_not_installed(self):
        """openai 库未安装时应抛出 ImportError。"""
        with patch.dict(sys.modules, {"openai": None}):
            llm = OpenAIVisionLLM(model="test", api_key="sk-test")
            with pytest.raises(ImportError, match="请安装 openai 库"):
                llm.chat_with_image("测试", b"\x89PNG\r\n\x1a\ntest")


# ─────────────────────────────────────────────
# 测试使用配置创建（集成 settings.yaml）
# ─────────────────────────────────────────────

@pytest.mark.unit
class TestVisionLLMConfigIntegration:
    """测试从 settings.yaml 配置创建 OpenAIVisionLLM。"""

    def test_create_from_settings_config(self):
        """从 settings.yaml 的 vision_llm 配置创建实例。"""
        config = VisionLLMConfig(
            provider="openai",
            model="mimo-v2.5",
            base_url="https://token-plan-cn.xiaomimimo.com/v1",
            api_key="sk-test",
        )
        llm = LLMFactory.create_vision_llm(config)
        assert isinstance(llm, OpenAIVisionLLM)
        assert llm.model == "mimo-v2.5"
        assert llm.base_url == "https://token-plan-cn.xiaomimimo.com/v1"
