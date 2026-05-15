"""测试 Vision LLM 抽象接口与工厂。

使用 Fake Vision LLM 隔离测试，不依赖真实 API。
"""

import pytest
from pathlib import Path
from typing import Union
from unittest.mock import MagicMock

from src.core.settings import VisionLLMConfig, load_settings
from src.libs.llm.base_llm import LLMResponse
from src.libs.llm.base_vision_llm import BaseVisionLLM
from src.libs.llm.llm_factory import (
    LLMFactory,
    _VISION_LLM_REGISTRY,
    register_vision_llm,
)


# ─────────────────────────────────────────────
# 测试用 Fake Vision LLM
# ─────────────────────────────────────────────

class FakeVisionLLM(BaseVisionLLM):
    """测试用 Vision LLM，返回预设响应。"""

    def __init__(self, model="fake-vision", api_key="", max_image_size=2048, **kwargs):
        super().__init__(model=model, api_key=api_key, max_image_size=max_image_size)
        self.kwargs = kwargs
        self.call_count = 0
        self.last_text = None
        self.last_image = None

    def chat_with_image(self, text: str, image: Union[str, bytes], **kwargs) -> LLMResponse:
        # 模拟真实实现：先验证图片输入
        image_bytes = self._load_image_bytes(image)
        self.call_count += 1
        self.last_text = text
        self.last_image = image_bytes
        return LLMResponse(
            content=f"[FakeVision] 对图片的描述: {text}",
            model=self.model,
        )


# 临时注册 FakeVisionLLM 用于测试
@register_vision_llm("fake_vision")
class _RegisteredFakeVisionLLM(FakeVisionLLM):
    pass


# ─────────────────────────────────────────────
# 测试 BaseVisionLLM 抽象接口
# ─────────────────────────────────────────────

@pytest.mark.unit
class TestBaseVisionLLMAbstract:
    """测试 BaseVisionLLM 抽象接口定义。"""

    def test_cannot_instantiate_directly(self):
        """BaseVisionLLM 是抽象类，不能直接实例化。"""
        with pytest.raises(TypeError):
            BaseVisionLLM(model="test")

    def test_subclass_must_implement_chat_with_image(self):
        """子类必须实现 chat_with_image 方法。"""
        class IncompleteVisionLLM(BaseVisionLLM):
            pass

        with pytest.raises(TypeError):
            IncompleteVisionLLM(model="test")

    def test_subclass_with_implementation_can_instantiate(self):
        """实现所有抽象方法的子类可以正常实例化。"""
        llm = FakeVisionLLM(model="test-model")
        assert llm.model == "test-model"

    def test_default_max_image_size(self):
        """默认 max_image_size 应为 2048。"""
        llm = FakeVisionLLM()
        assert llm.max_image_size == 2048

    def test_custom_max_image_size(self):
        """自定义 max_image_size 应正确传递。"""
        llm = FakeVisionLLM(max_image_size=1024)
        assert llm.max_image_size == 1024


# ─────────────────────────────────────────────
# 测试 _load_image_bytes 静态方法
# ─────────────────────────────────────────────

@pytest.mark.unit
class TestLoadImageBytes:
    """测试图片加载工具方法。"""

    def test_load_bytes_passthrough(self):
        """bytes 类型应直接返回。"""
        data = b"\x89PNG\r\n\x1a\n"
        result = BaseVisionLLM._load_image_bytes(data)
        assert result == data

    def test_load_empty_bytes_raises(self):
        """空 bytes 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="图片字节数据不能为空"):
            BaseVisionLLM._load_image_bytes(b"")

    def test_load_nonexistent_path_raises(self):
        """不存在的文件路径应抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError, match="图片文件不存在"):
            BaseVisionLLM._load_image_bytes("/nonexistent/image.png")

    def test_load_valid_path(self, tmp_path):
        """存在的文件路径应正确读取内容。"""
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"\x89PNGtest")
        result = BaseVisionLLM._load_image_bytes(str(img_file))
        assert result == b"\x89PNGtest"


# ─────────────────────────────────────────────
# 测试 register_vision_llm 装饰器
# ─────────────────────────────────临时注册

@pytest.mark.unit
class TestRegisterVisionLLM:
    """测试 Vision LLM 注册装饰器。"""

    def test_register_adds_to_registry(self):
        """注册后应出现在注册表中。"""
        assert "fake_vision" in _VISION_LLM_REGISTRY

    def test_registered_class_is_correct(self):
        """注册表中存储的类应正确。"""
        assert _VISION_LLM_REGISTRY["fake_vision"] is _RegisteredFakeVisionLLM

    def test_register_case_insensitive(self):
        """装饰器注册的键应为小写。"""
        @register_vision_llm("TestProvider")
        class TestVisionLLM(FakeVisionLLM):
            pass

        assert "testprovider" in _VISION_LLM_REGISTRY
        # 清理
        del _VISION_LLM_REGISTRY["testprovider"]


# ─────────────────────────────────────────────
# 测试 LLMFactory.create_vision_llm
# ─────────────────────────────────────────────

@pytest.mark.unit
class TestLLMFactoryCreateVisionLLM:
    """测试工厂方法 create_vision_llm。"""

    def test_create_vision_llm_with_registered_provider(self):
        """已注册的提供者应能创建实例。"""
        config = VisionLLMConfig(provider="fake_vision", model="test-model")
        llm = LLMFactory.create_vision_llm(config)
        assert isinstance(llm, BaseVisionLLM)
        assert llm.model == "test-model"

    def test_create_vision_llm_passes_all_config(self):
        """所有配置字段应正确传递给实例。"""
        config = VisionLLMConfig(
            provider="fake_vision",
            model="mimo-v2.5",
            api_key="test-key",
            base_url="https://example.com/v1",
            temperature=0.5,
            max_tokens=2048,
            max_image_size=1024,
        )
        llm = LLMFactory.create_vision_llm(config)
        assert llm.model == "mimo-v2.5"
        assert llm.api_key == "test-key"
        assert llm.max_image_size == 1024

    def test_create_vision_llm_unknown_provider_raises(self):
        """未注册的提供者应抛出 ValueError。"""
        config = VisionLLMConfig(provider="unknown_vision", model="test")
        with pytest.raises(ValueError, match="未知的 Vision LLM 提供者"):
            LLMFactory.create_vision_llm(config)

    def test_create_vision_llm_error_message_includes_available(self):
        """错误信息应包含可用提供者列表。"""
        config = VisionLLMConfig(provider="unknown", model="test")
        with pytest.raises(ValueError, match="可用提供者"):
            LLMFactory.create_vision_llm(config)

    def test_create_vision_llm_case_insensitive(self):
        """工厂应不区分大小写匹配提供者。"""
        config = VisionLLMConfig(provider="Fake_Vision", model="test")
        llm = LLMFactory.create_vision_llm(config)
        assert isinstance(llm, BaseVisionLLM)


# ─────────────────────────────────────────────
# 测试 LLMFactory.list_vision_providers
# ─────────────────────────────────────────────

@pytest.mark.unit
class TestListVisionProviders:
    """测试列出 Vision LLM 提供者。"""

    def test_list_vision_providers_returns_sorted(self):
        """返回的提供者列表应已排序。"""
        providers = LLMFactory.list_vision_providers()
        assert providers == sorted(providers)

    def test_list_vision_providers_includes_registered(self):
        """已注册的提供者应出现在列表中。"""
        providers = LLMFactory.list_vision_providers()
        assert "fake_vision" in providers


# ─────────────────────────────────────────────
# 测试 FakeVisionLLM 功能验证
# ─────────────────────────────────────────────

@pytest.mark.unit
class TestFakeVisionLLMFunctionality:
    """验证 FakeVisionLLM 能正常工作（端到端）。"""

    def test_chat_with_image_bytes(self):
        """传入 bytes 图片应正常返回。"""
        llm = FakeVisionLLM()
        response = llm.chat_with_image("描述这张图片", b"\x89PNGdata")
        assert isinstance(response, LLMResponse)
        assert "FakeVision" in response.content
        assert llm.call_count == 1

    def test_chat_with_image_path(self, tmp_path):
        """传入文件路径应正常返回。"""
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNGdata")
        llm = FakeVisionLLM()
        response = llm.chat_with_image("描述这张图片", str(img))
        assert isinstance(response, LLMResponse)
        assert llm.last_text == "描述这张图片"

    def test_chat_with_image_nonexistent_path_raises(self):
        """传入不存在的路径应抛出 FileNotFoundError。"""
        llm = FakeVisionLLM()
        with pytest.raises(FileNotFoundError):
            llm.chat_with_image("描述", "/nonexistent/image.png")

    def test_chat_with_image_empty_bytes_raises(self):
        """传入空 bytes 应抛出 ValueError。"""
        llm = FakeVisionLLM()
        with pytest.raises(ValueError):
            llm.chat_with_image("描述", b"")


# ─────────────────────────────────────────────
# 测试 settings.yaml 中的 vision_llm 配置
# ─────────────────────────────────────────────

@pytest.mark.unit
class TestVisionLLMConfigInSettings:
    """测试 VisionLLMConfig 在 Settings 中的集成。"""

    def test_settings_has_vision_llm_field(self):
        """Settings 应包含 vision_llm 字段。"""
        settings = load_settings()
        assert hasattr(settings, "vision_llm")
        assert isinstance(settings.vision_llm, VisionLLMConfig)

    def test_vision_llm_config_values_from_yaml(self):
        """vision_llm 配置值应从 settings.yaml 正确加载。"""
        settings = load_settings()
        assert settings.vision_llm.provider == "openai"
        assert settings.vision_llm.model == "mimo-v2.5"

    def test_vision_llm_default_max_image_size(self):
        """未配置 max_image_size 时应使用默认值 2048。"""
        settings = load_settings()
        assert settings.vision_llm.max_image_size == 2048
