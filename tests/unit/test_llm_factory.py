"""Tests for LLM base class and factory."""

import pytest
from typing import List

from src.core.settings import LLMConfig
from src.libs.llm.base_llm import BaseLLM, Message, LLMResponse
from src.libs.llm.llm_factory import LLMFactory, register_llm, _LLM_REGISTRY


# --- Fake LLM for testing ---

class FakeLLM(BaseLLM):
    """Fake LLM implementation for testing."""

    def __init__(self, model: str, api_key: str = "", **kwargs):
        super().__init__(model, api_key, **kwargs)
        self.kwargs = kwargs

    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        last_msg = messages[-1].content if messages else ""
        return LLMResponse(
            content=f"Fake response to: {last_msg}",
            model=self.model,
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )


# --- Tests ---

@pytest.mark.unit
class TestMessage:
    """Test Message dataclass."""

    def test_create_message(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_system_message(self):
        msg = Message(role="system", content="You are helpful")
        assert msg.role == "system"


@pytest.mark.unit
class TestLLMResponse:
    """Test LLMResponse dataclass."""

    def test_create_response(self):
        resp = LLMResponse(content="Hi", model="test-model")
        assert resp.content == "Hi"
        assert resp.model == "test-model"
        assert resp.usage is None

    def test_response_with_usage(self):
        usage = {"prompt_tokens": 10, "completion_tokens": 5}
        resp = LLMResponse(content="Hi", model="test", usage=usage)
        assert resp.usage == usage


@pytest.mark.unit
class TestBaseLLM:
    """Test BaseLLM interface."""

    def test_cannot_instantiate_abstract(self):
        """BaseLLM is abstract, cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLLM(model="test")

    def test_fake_llm_chat(self):
        """FakeLLM implements chat method."""
        llm = FakeLLM(model="fake-model")
        messages = [Message(role="user", content="Hello")]
        response = llm.chat(messages)
        assert "Fake response to: Hello" in response.content
        assert response.model == "fake-model"

    def test_chat_simple(self):
        """chat_simple should construct messages and call chat."""
        llm = FakeLLM(model="fake")
        result = llm.chat_simple("What is RAG?")
        assert "Fake response to: What is RAG?" in result

    def test_chat_simple_with_system(self):
        """chat_simple with system message."""
        llm = FakeLLM(model="fake")
        result = llm.chat_simple("Hello", system="You are helpful")
        assert "Fake response to: Hello" in result


@pytest.mark.unit
class TestLLMFactory:
    """Test LLMFactory.create routing logic."""

    def setup_method(self):
        """Register fake provider before each test."""
        _LLM_REGISTRY["fake"] = FakeLLM

    def teardown_method(self):
        """Clean up registry after each test."""
        _LLM_REGISTRY.pop("fake", None)

    def test_create_registered_provider(self):
        """Should create instance for registered provider."""
        config = LLMConfig(provider="fake", model="test-model", api_key="sk-test")
        llm = LLMFactory.create(config)
        assert isinstance(llm, FakeLLM)
        assert llm.model == "test-model"
        assert llm.api_key == "sk-test"

    def test_create_passes_kwargs(self):
        """Should pass temperature and max_tokens to constructor."""
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
        """Provider matching should be case-insensitive."""
        config = LLMConfig(provider="FAKE", model="test")
        llm = LLMFactory.create(config)
        assert isinstance(llm, FakeLLM)

    def test_list_providers(self):
        """Should list all registered providers."""
        providers = LLMFactory.list_providers()
        assert "fake" in providers


@pytest.mark.unit
class TestRegisterLLMDecorator:
    """Test @register_llm decorator."""

    def teardown_method(self):
        """Clean up registry."""
        _LLM_REGISTRY.pop("test_provider", None)

    def test_register_decorator(self):
        """@register_llm should add class to registry."""
        @register_llm("test_provider")
        class TestLLM(BaseLLM):
            def chat(self, messages, **kwargs):
                return LLMResponse(content="test", model="test")

        assert "test_provider" in _LLM_REGISTRY
        assert _LLM_REGISTRY["test_provider"] is TestLLM

    def test_register_lowercase(self):
        """Provider name should be stored lowercase."""
        @register_llm("TEST_UPPER")
        class UpperLLM(BaseLLM):
            def chat(self, messages, **kwargs):
                return LLMResponse(content="test", model="test")

        assert "test_upper" in _LLM_REGISTRY
        _LLM_REGISTRY.pop("test_upper", None)
