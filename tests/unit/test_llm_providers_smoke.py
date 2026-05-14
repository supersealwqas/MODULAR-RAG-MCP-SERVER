"""测试 LLM 提供者实现的冒烟测试。

使用 mock 测试工厂路由和基本功能，不走真实网络。
"""

import pytest
from unittest.mock import patch, MagicMock

from src.core.settings import LLMConfig
from src.libs.llm.base_llm import Message, LLMResponse
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.openai_llm import OpenAILLM


@pytest.mark.unit
class TestOpenAILLM:
    """测试 OpenAI 兼容 LLM 实现。"""

    def test_factory_creates_openai(self):
        """provider=openai 时工厂应创建 OpenAILLM 实例。"""
        config = LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
        )
        llm = LLMFactory.create(config)
        assert isinstance(llm, OpenAILLM)
        assert llm.model == "gpt-4o"
        assert llm.base_url == "https://api.openai.com/v1"

    def test_factory_creates_with_custom_base_url(self):
        """应支持自定义 base_url（如阿里云百炼）。"""
        config = LLMConfig(
            provider="openai",
            model="deepseek-v4-flash",
            api_key="sk-test",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        llm = LLMFactory.create(config)
        assert isinstance(llm, OpenAILLM)
        assert llm.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"

    @patch("openai.OpenAI")
    def test_chat_calls_api(self, mock_openai_class):
        """chat 方法应正确调用 OpenAI API。"""
        # 设置 mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="测试回复"))]
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        mock_client.chat.completions.create.return_value = mock_response

        # 创建实例并调用
        llm = OpenAILLM(model="gpt-4o", api_key="sk-test")
        messages = [Message(role="user", content="你好")]
        response = llm.chat(messages)

        # 验证
        assert isinstance(response, LLMResponse)
        assert response.content == "测试回复"
        assert response.model == "gpt-4o"
        assert response.usage["prompt_tokens"] == 10

        # 验证 API 调用参数
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o"
        assert call_kwargs.kwargs["messages"] == [{"role": "user", "content": "你好"}]

    @patch("openai.OpenAI")
    def test_chat_with_base_url(self, mock_openai_class):
        """应将 base_url 传递给 OpenAI 客户端。"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="回复"))]
        mock_response.model = "deepseek-v4-flash"
        mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=3, total_tokens=8)
        mock_client.chat.completions.create.return_value = mock_response

        llm = OpenAILLM(
            model="deepseek-v4-flash",
            api_key="sk-test",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        messages = [Message(role="user", content="测试")]
        llm.chat(messages)

        # 验证 base_url 传递
        mock_openai_class.assert_called_once_with(
            api_key="sk-test",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    @patch("openai.OpenAI")
    def test_chat_error_handling(self, mock_openai_class):
        """API 调用失败时应抛出 RuntimeError。"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("连接超时")

        llm = OpenAILLM(model="gpt-4o", api_key="sk-test")
        messages = [Message(role="user", content="测试")]

        with pytest.raises(RuntimeError, match="OpenAI API 调用失败"):
            llm.chat(messages)

    @patch("openai.OpenAI")
    def test_chat_simple_method(self, mock_openai_class):
        """chat_simple 便捷方法应正常工作。"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="简单回复"))]
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=3, total_tokens=8)
        mock_client.chat.completions.create.return_value = mock_response

        llm = OpenAILLM(model="gpt-4o", api_key="sk-test")
        result = llm.chat_simple("你好", system="你是一个助手")

        assert result == "简单回复"
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["role"] == "user"


@pytest.mark.unit
class TestLLMFactoryRouting:
    """测试 LLM 工厂路由逻辑。"""

    def test_case_insensitive_routing(self):
        """提供者名称应不区分大小写。"""
        config = LLMConfig(provider="OPENAI", model="test", api_key="sk-test")
        llm = LLMFactory.create(config)
        assert isinstance(llm, OpenAILLM)

    def test_list_providers_includes_openai(self):
        """应列出 openai 提供者。"""
        providers = LLMFactory.list_providers()
        assert "openai" in providers
