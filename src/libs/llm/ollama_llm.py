"""Ollama LLM 实现模块。

支持通过本地 Ollama HTTP API 调用本地部署的大语言模型。
默认地址: http://localhost:11434
"""

from __future__ import annotations

import json
from typing import List, Optional

from src.libs.llm.base_llm import BaseLLM, LLMResponse, Message
from src.libs.llm.llm_factory import register_llm


@register_llm("ollama")
class OllamaLLM(BaseLLM):
    """Ollama LLM 实现，支持本地部署的模型。

    通过 HTTP API 与本地 Ollama 服务通信。
    """

    def __init__(
        self,
        model: str = "gemma4",
        api_key: str = "",  # Ollama 不需要 api_key，保留接口一致性
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs,
    ):
        """初始化 Ollama LLM 实例。

        参数:
            model: 模型名称（如 "gemma4"、"llama3"、"qwen2"）
            api_key: 未使用，保留接口一致性
            base_url: Ollama 服务地址（默认 http://localhost:11434）
            temperature: 生成温度（0.0-2.0）
            max_tokens: 最大生成 token 数
            **kwargs: 其他参数
        """
        super().__init__(model, api_key, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """发送聊天请求到 Ollama API。

        参数:
            messages: 消息列表
            **kwargs: 额外参数（可覆盖 temperature 等）

        返回:
            LLMResponse 包含生成内容和元数据

        异常:
            RuntimeError: API 调用失败时抛出
        """
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "请安装 httpx 库: uv pip install httpx"
            )

        temperature = kwargs.get("temperature", self.temperature)

        # 转换消息格式
        ollama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        # 构建请求体
        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            return LLMResponse(
                content=data.get("message", {}).get("content", ""),
                model=data.get("model", self.model),
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": (
                        data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                    ),
                },
            )
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Ollama 连接失败 (base_url={self.base_url}): "
                f"请确保 Ollama 服务已启动。错误详情: {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise RuntimeError(
                f"Ollama 请求超时 (model={self.model}): {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Ollama API 调用失败 (provider=ollama, model={self.model}): {e}"
            ) from e
