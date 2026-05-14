"""LLM 提供者的抽象基类模块。

定义所有 LLM 实现必须遵循的接口规范。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Message:
    """聊天消息数据类。

    属性:
        role: 消息角色，可选值为 "system"（系统）、"user"（用户）、"assistant"（助手）
        content: 消息文本内容
    """
    role: str
    content: str


@dataclass
class LLMResponse:
    """LLM 响应数据类，标准化各提供者的返回格式。

    属性:
        content: 生成的文本内容
        model: 使用的模型名称
        usage: token 使用量统计（可选），包含 prompt_tokens、completion_tokens 等
    """
    content: str
    model: str
    usage: Optional[dict] = None


class BaseLLM(ABC):
    """所有 LLM 实现的抽象基类。

    子类必须实现 `chat` 方法来完成实际的 API 调用。
    提供 `chat_simple` 便捷方法用于简单的单轮对话场景。
    """

    def __init__(self, model: str, api_key: str = "", **kwargs):
        """初始化 LLM 实例。

        参数:
            model: 模型名称（如 "gpt-4o"、"llama3"）
            api_key: API 密钥（可选，部分本地模型不需要）
            **kwargs: 其他提供者特定参数
        """
        self.model = model
        self.api_key = api_key

    @abstractmethod
    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """发送聊天请求到 LLM。

        参数:
            messages: 消息列表，包含对话历史
            **kwargs: 提供者特定参数（如 temperature、max_tokens）

        返回:
            LLMResponse 对象，包含生成内容和元数据
        """
        ...

    def chat_simple(self, prompt: str, system: str = "", **kwargs) -> str:
        """简易聊天方法，自动构造消息列表。

        适用于不需要管理完整对话历史的单轮场景。

        参数:
            prompt: 用户消息内容
            system: 系统提示词（可选）
            **kwargs: 提供者特定参数

        返回:
            生成的文本内容字符串
        """
        messages = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=prompt))

        response = self.chat(messages, **kwargs)
        return response.content
