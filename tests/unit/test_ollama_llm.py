"""测试 Ollama LLM 实现。

使用 mock HTTP 测试，不依赖真实的 Ollama 服务。
"""

import pytest
from unittest.mock import patch, MagicMock

from src.core.settings import LLMConfig
from src.libs.llm.base_llm import Message, LLMResponse
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.ollama_llm import OllamaLLM


@pytest.mark.unit
class TestOllamaLLM:
    """测试 Ollama LLM 实现。"""

    def test_factory_creates_ollama(self):
        """provider=ollama 时工厂应创建 OllamaLLM 实例。"""
        config = LLMConfig(
            provider="ollama",
            model="gemma4",
            base_url="http://localhost:11434",
        )
        llm = LLMFactory.create(config)
        assert isinstance(llm, OllamaLLM)
        assert llm.model == "gemma4"
        assert llm.base_url == "http://localhost:11434"

    def test_default_values(self):
        """应有合理的默认值。"""
        llm = OllamaLLM()
        assert llm.model == "gemma4"
        assert llm.base_url == "http://localhost:11434"
        assert llm.temperature == 0.0
        assert llm.max_tokens == 4096

    @patch("httpx.Client")
    def test_chat_calls_api(self, mock_client_class):
        """chat 方法应正确调用 Ollama API。"""
        # 设置 mock
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_class.return_value.__exit__ = MagicMock(return_value=False)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "gemma4",
            "message": {"content": "测试回复"},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response

        # 创建实例并调用
        llm = OllamaLLM(model="gemma4")
        messages = [Message(role="user", content="你好")]
        response = llm.chat(messages)

        # 验证
        assert isinstance(response, LLMResponse)
        assert response.content == "测试回复"
        assert response.model == "gemma4"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 5

        # 验证 API 调用
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "/api/chat" in call_args[0][0]
        assert call_args[1]["json"]["model"] == "gemma4"

    @patch("httpx.Client")
    def test_chat_with_custom_base_url(self, mock_client_class):
        """应支持自定义 base_url。"""
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_class.return_value.__exit__ = MagicMock(return_value=False)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "gemma4",
            "message": {"content": "回复"},
            "prompt_eval_count": 5,
            "eval_count": 3,
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response

        llm = OllamaLLM(model="gemma4", base_url="http://192.168.1.100:11434/")
        messages = [Message(role="user", content="测试")]
        llm.chat(messages)

        # 验证 base_url（尾部斜杠应被移除）
        call_args = mock_client.post.call_args
        assert "http://192.168.1.100:11434/api/chat" in call_args[0][0]

    @patch("httpx.Client")
    def test_chat_connect_error(self, mock_client_class):
        """连接失败时应抛出可读的 RuntimeError。"""
        import httpx

        mock_client = MagicMock()
        mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")

        llm = OllamaLLM(model="gemma4")
        messages = [Message(role="user", content="测试")]

        with pytest.raises(RuntimeError, match="Ollama 连接失败"):
            llm.chat(messages)

    @patch("httpx.Client")
    def test_chat_timeout_error(self, mock_client_class):
        """超时时应抛出可读的 RuntimeError。"""
        import httpx

        mock_client = MagicMock()
        mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.TimeoutException("Timeout")

        llm = OllamaLLM(model="gemma4")
        messages = [Message(role="user", content="测试")]

        with pytest.raises(RuntimeError, match="Ollama 请求超时"):
            llm.chat(messages)

    @patch("httpx.Client")
    def test_chat_simple_method(self, mock_client_class):
        """chat_simple 便捷方法应正常工作。"""
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_class.return_value.__exit__ = MagicMock(return_value=False)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "gemma4",
            "message": {"content": "简单回复"},
            "prompt_eval_count": 5,
            "eval_count": 3,
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response

        llm = OllamaLLM(model="gemma4")
        result = llm.chat_simple("你好", system="你是一个助手")

        assert result == "简单回复"
        call_kwargs = mock_client.post.call_args[1]["json"]
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"


@pytest.mark.unit
class TestOllamaFactoryRouting:
    """测试 Ollama 工厂路由。"""

    def test_case_insensitive_routing(self):
        """提供者名称应不区分大小写。"""
        config = LLMConfig(provider="OLLAMA", model="test")
        llm = LLMFactory.create(config)
        assert isinstance(llm, OllamaLLM)

    def test_list_providers_includes_ollama(self):
        """应列出 ollama 提供者。"""
        providers = LLMFactory.list_providers()
        assert "ollama" in providers
