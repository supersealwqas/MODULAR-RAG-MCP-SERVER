"""测试 LLM 基类和工厂。"""

import pytest
from typing import List

from src.core.settings import LLMConfig
from src.libs.llm.base_llm import BaseLLM, Message, LLMResponse
from src.libs.llm.llm_factory import LLMFactory, register_llm, _LLM_REGISTRY


# --- 用于测试的 Fake LLM ---

class FakeLLM(BaseLLM):
    """用于测试的假 LLM 实现。"""

    def __init__(self, model: str, api_key: str = "", **kwargs):
        """初始化 FakeLLM。

        参数:
            model: 模型名称
            api_key: API 密钥
            **kwargs: 其他参数
        """
        super().__init__(model, api_key, **kwargs)
        self.kwargs = kwargs

    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """发送聊天请求，返回预设响应。"""
        last_msg = messages[-1].content if messages else ""
        return LLMResponse(
            content=f"Fake response to: {last_msg}",
            model=self.model,
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )


# --- 测试用例 ---

@pytest.mark.unit
class TestMessage:
    """测试 Message 数据类。"""

    def test_create_message(self):
        """应能创建消息。"""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_system_message(self):
        """应能创建系统消息。"""
        msg = Message(role="system", content="You are helpful")
        assert msg.role == "system"


@pytest.mark.unit
class TestLLMResponse:
    """测试 LLMResponse 数据类。"""

    def test_create_response(self):
        """应能创建响应。"""
        resp = LLMResponse(content="Hi", model="test-model")
        assert resp.content == "Hi"
        assert resp.model == "test-model"
        assert resp.usage is None

    def test_response_with_usage(self):
        """应能包含 usage 信息。"""
        usage = {"prompt_tokens": 10, "completion_tokens": 5}
        resp = LLMResponse(content="Hi", model="test", usage=usage)
        assert resp.usage == usage


@pytest.mark.unit
class TestBaseLLM:
    """测试 BaseLLM 接口。"""

    def test_cannot_instantiate_abstract(self):
        """BaseLLM 是抽象类，不能直接实例化。"""
        with pytest.raises(TypeError):
            BaseLLM(model="test")

    def test_fake_llm_chat(self):
        """FakeLLM 应实现 chat 方法。"""
        llm = FakeLLM(model="fake-model")
        messages = [Message(role="user", content="Hello")]
        response = llm.chat(messages)
        assert "Fake response to: Hello" in response.content
        assert response.model == "fake-model"

    def test_chat_simple(self):
        """chat_simple 应构造消息并调用 chat。"""
        llm = FakeLLM(model="fake")
        result = llm.chat_simple("What is RAG?")
        assert "Fake response to: What is RAG?" in result

    def test_chat_simple_with_system(self):
        """chat_simple 应支持系统消息。"""
        llm = FakeLLM(model="fake")
        result = llm.chat_simple("Hello", system="You are helpful")
        assert "Fake response to: Hello" in result


@pytest.mark.unit
class TestLLMFactory:
    """测试 LLMFactory.create 路由逻辑。"""

    def setup_method(self):
        """每个测试前注册 fake 提供者。"""
        _LLM_REGISTRY["fake"] = FakeLLM

    def teardown_method(self):
        """每个测试后清理注册表。"""
        _LLM_REGISTRY.pop("fake", None)

    def test_create_registered_provider(self):
        """应为已注册的提供者创建实例。"""
        config = LLMConfig(provider="fake", model="test-model", api_key="sk-test")
        llm = LLMFactory.create(config)
        assert isinstance(llm, FakeLLM)
        assert llm.model == "test-model"
        assert llm.api_key == "sk-test"

    def test_create_passes_kwargs(self):
        """应将 temperature 和 max_tokens 传递给构造函数。"""
        config = LLMConfig(
            provider="fake", model="test", temperature=0.7, max_tokens=2048
        )
        llm = LLMFactory.create(config)
        assert llm.kwargs.get("temperature") == 0.7
        assert llm.kwargs.get("max_tokens") == 2048

    def test_unknown_provider_raises(self):
        """未注册的提供者应抛出 ValueError。"""
        config = LLMConfig(provider="unknown", model="test")
        with pytest.raises(ValueError, match="未知的 LLM 提供者.*unknown"):
            LLMFactory.create(config)

    def test_case_insensitive_provider(self):
        """提供者名称匹配应不区分大小写。"""
        config = LLMConfig(provider="FAKE", model="test")
        llm = LLMFactory.create(config)
        assert isinstance(llm, FakeLLM)

    def test_list_providers(self):
        """应列出所有已注册的提供者。"""
        providers = LLMFactory.list_providers()
        assert "fake" in providers


@pytest.mark.unit
class TestRegisterLLMDecorator:
    """测试 @register_llm 装饰器。"""

    def teardown_method(self):
        """清理注册表。"""
        _LLM_REGISTRY.pop("test_provider", None)

    def test_register_decorator(self):
        """@register_llm 应将类添加到注册表。"""
        @register_llm("test_provider")
        class TestLLM(BaseLLM):
            def chat(self, messages, **kwargs):
                return LLMResponse(content="test", model="test")

        assert "test_provider" in _LLM_REGISTRY
        assert _LLM_REGISTRY["test_provider"] is TestLLM

    def test_register_lowercase(self):
        """提供者名称应存储为小写。"""
        @register_llm("TEST_UPPER")
        class UpperLLM(BaseLLM):
            def chat(self, messages, **kwargs):
                return LLMResponse(content="test", model="test")

        assert "test_upper" in _LLM_REGISTRY
        _LLM_REGISTRY.pop("test_upper", None)
