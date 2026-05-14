"""Embedding 工厂模块，根据配置创建对应的 Embedding 实例。

使用注册表模式，通过装饰器注册各提供者实现，工厂根据配置自动路由。
"""

from __future__ import annotations

from typing import Dict, Type

from src.core.settings import EmbeddingConfig
from src.libs.embedding.base_embedding import BaseEmbedding


# Embedding 提供者注册表，键为提供者名称（小写），值为对应的类
_EMBEDDING_REGISTRY: Dict[str, Type[BaseEmbedding]] = {}


def register_embedding(provider: str):
    """注册 Embedding 提供者的装饰器。

    用法:
        @register_embedding("openai")
        class OpenAIEmbedding(BaseEmbedding):
            ...

    参数:
        provider: 提供者名称（如 "openai"、"local"），会自动转为小写
    """
    def decorator(cls: Type[BaseEmbedding]) -> Type[BaseEmbedding]:
        _EMBEDDING_REGISTRY[provider.lower()] = cls
        return cls
    return decorator


class EmbeddingFactory:
    """Embedding 实例工厂，根据配置创建对应的 Embedding 提供者实例。"""

    @staticmethod
    def create(config: EmbeddingConfig) -> BaseEmbedding:
        """根据配置创建 Embedding 实例。

        参数:
            config: EmbeddingConfig 配置对象，来自 settings.yaml

        返回:
            BaseEmbedding 子类实例

        异常:
            ValueError: 提供者未注册时抛出
        """
        provider = config.provider.lower()
        if provider not in _EMBEDDING_REGISTRY:
            available = ", ".join(sorted(_EMBEDDING_REGISTRY.keys())) or "(none)"
            raise ValueError(
                f"未知的 Embedding 提供者: '{config.provider}'，"
                f"可用提供者: {available}"
            )

        cls = _EMBEDDING_REGISTRY[provider]
        return cls(
            model=config.model,
            dimensions=config.dimensions,
            api_key=config.api_key,
        )

    @staticmethod
    def list_providers() -> list[str]:
        """列出所有已注册的 Embedding 提供者名称。

        返回:
            提供者名称列表（已排序）
        """
        return sorted(_EMBEDDING_REGISTRY.keys())
