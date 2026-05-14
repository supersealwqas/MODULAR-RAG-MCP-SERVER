"""OpenAI 兼容 LLM 实现模块。

支持 OpenAI 官方 API 及兼容 OpenAI 格式的第三方服务（如阿里云百炼、DeepSeek 等）。
"""

from __future__ import annotations

from typing import List, Optional

from src.libs.llm.base_llm import BaseLLM, LLMResponse, Message
from src.libs.llm.llm_factory import register_llm


@register_llm("openai")
class OpenAILLM(BaseLLM):
    """OpenAI 兼容 LLM 实现。

    支持通过 base_url 配置自定义 API 端点，
    适用于阿里云百炼、DeepSeek 等兼容 OpenAI 格式的服务。
    """

    def __init__(
        self,
        model: str,
        api_key: str = "",
        base_url: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs,
    ):
        """初始化 OpenAI 兼容 LLM 实例。

        参数:
            model: 模型名称（如 "gpt-4o"、"deepseek-v4-flash"）
            api_key: API 密钥
            base_url: 自定义 API 端点（为空时使用 OpenAI 默认地址）
            temperature: 生成温度（0.0-2.0）
            max_tokens: 最大生成 token 数
            **kwargs: 其他参数
        """
        super().__init__(model, api_key, **kwargs)
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """发送聊天请求到 OpenAI 兼容 API。

        参数:
            messages: 消息列表
            **kwargs: 额外参数（可覆盖 temperature、max_tokens 等）

        返回:
            LLMResponse 包含生成内容和元数据

        异常:
            ImportError: 未安装 openai 库时抛出
            RuntimeError: API 调用失败时抛出
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "请安装 openai 库: uv pip install openai"
            )

        # 构建客户端参数
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        # 构建请求参数
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        # 转换消息格式
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        try:
            client = OpenAI(**client_kwargs)
            response = client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
            )
        except Exception as e:
            raise RuntimeError(
                f"OpenAI API 调用失败 (provider=openai, model={self.model}): {e}"
            ) from e
