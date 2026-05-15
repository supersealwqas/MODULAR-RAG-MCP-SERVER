"""LLM 工厂模块，根据配置创建对应的 LLM 和 Vision LLM 实例。

使用注册表模式，通过装饰器注册各提供者实现，工厂根据配置自动路由。
"""

from __future__ import annotations

from typing import Dict, Type

from src.core.settings import LLMConfig, VisionLLMConfig
from src.libs.llm.base_llm import BaseLLM
from src.libs.llm.base_vision_llm import BaseVisionLLM


# LLM 提供者注册表，键为提供者名称（小写），值为对应的类
_LLM_REGISTRY: Dict[str, Type[BaseLLM]] = {}

# Vision LLM 提供者注册表
_VISION_LLM_REGISTRY: Dict[str, Type[BaseVisionLLM]] = {}


def register_llm(provider: str):
    """注册 LLM 提供者的装饰器。

    用法:
        @register_llm("openai")
        class OpenAILLM(BaseLLM):
            ...

    参数:
        provider: 提供者名称（如 "openai"、"ollama"），会自动转为小写
    """
    def decorator(cls: Type[BaseLLM]) -> Type[BaseLLM]:
        _LLM_REGISTRY[provider.lower()] = cls
        return cls
    return decorator


def register_vision_llm(provider: str):
    """注册 Vision LLM 提供者的装饰器。

    用法:
        @register_vision_llm("openai")
        class OpenAIVisionLLM(BaseVisionLLM):
            ...

    参数:
        provider: 提供者名称（如 "openai"、"ollama"），会自动转为小写
    """
    def decorator(cls: Type[BaseVisionLLM]) -> Type[BaseVisionLLM]:
        _VISION_LLM_REGISTRY[provider.lower()] = cls
        return cls
    return decorator


class LLMFactory:
    """LLM 实例工厂，根据配置创建对应的 LLM/Vision LLM 提供者实例。"""

    @staticmethod
    def create(config: LLMConfig) -> BaseLLM:
        """根据配置创建 LLM 实例。

        参数:
            config: LLMConfig 配置对象，来自 settings.yaml

        返回:
            BaseLLM 子类实例

        异常:
            ValueError: 提供者未注册时抛出
        """
        provider = config.provider.lower()
        if provider not in _LLM_REGISTRY:
            available = ", ".join(sorted(_LLM_REGISTRY.keys())) or "(none)"
            raise ValueError(
                f"未知的 LLM 提供者: '{config.provider}'，"
                f"可用提供者: {available}"
            )

        cls = _LLM_REGISTRY[provider]
        return cls(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    @staticmethod
    def create_vision_llm(config: VisionLLMConfig) -> BaseVisionLLM:
        """根据配置创建 Vision LLM 实例。

        参数:
            config: VisionLLMConfig 配置对象，来自 settings.yaml

        返回:
            BaseVisionLLM 子类实例

        异常:
            ValueError: 提供者未注册时抛出
        """
        provider = config.provider.lower()
        if provider not in _VISION_LLM_REGISTRY:
            available = ", ".join(sorted(_VISION_LLM_REGISTRY.keys())) or "(none)"
            raise ValueError(
                f"未知的 Vision LLM 提供者: '{config.provider}'，"
                f"可用提供者: {available}"
            )

        cls = _VISION_LLM_REGISTRY[provider]
        return cls(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            max_image_size=config.max_image_size,
        )

    @staticmethod
    def list_providers() -> list[str]:
        """列出所有已注册的 LLM 提供者名称。

        返回:
            提供者名称列表（已排序）
        """
        return sorted(_LLM_REGISTRY.keys())

    @staticmethod
    def list_vision_providers() -> list[str]:
        """列出所有已注册的 Vision LLM 提供者名称。

        返回:
            提供者名称列表（已排序）
        """
        return sorted(_VISION_LLM_REGISTRY.keys())
