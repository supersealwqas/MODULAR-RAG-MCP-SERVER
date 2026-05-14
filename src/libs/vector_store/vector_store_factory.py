"""向量存储工厂模块，根据配置创建对应的 VectorStore 实例。

使用注册表模式，通过装饰器注册各提供者实现，工厂根据配置自动路由。
"""

from __future__ import annotations

from typing import Dict, Type

from src.core.settings import VectorStoreConfig
from src.libs.vector_store.base_vector_store import BaseVectorStore


# VectorStore 提供者注册表，键为提供者名称（小写），值为对应的类
_VECTOR_STORE_REGISTRY: Dict[str, Type[BaseVectorStore]] = {}


def register_vector_store(provider: str):
    """注册 VectorStore 提供者的装饰器。

    用法:
        @register_vector_store("chroma")
        class ChromaVectorStore(BaseVectorStore):
            ...

    参数:
        provider: 提供者名称（如 "chroma"、"memory"），会自动转为小写
    """
    def decorator(cls: Type[BaseVectorStore]) -> Type[BaseVectorStore]:
        _VECTOR_STORE_REGISTRY[provider.lower()] = cls
        return cls
    return decorator


class VectorStoreFactory:
    """VectorStore 实例工厂，根据配置创建对应的向量存储实例。"""

    @staticmethod
    def create(config: VectorStoreConfig, **kwargs) -> BaseVectorStore:
        """根据配置创建 VectorStore 实例。

        参数:
            config: VectorStoreConfig 配置对象，来自 settings.yaml
            **kwargs: 其他提供者特定参数

        返回:
            BaseVectorStore 子类实例

        异常:
            ValueError: 提供者未注册时抛出
        """
        provider = config.provider.lower()
        if provider not in _VECTOR_STORE_REGISTRY:
            available = ", ".join(sorted(_VECTOR_STORE_REGISTRY.keys())) or "(none)"
            raise ValueError(
                f"未知的向量存储提供者: '{config.provider}'，"
                f"可用提供者: {available}"
            )

        cls = _VECTOR_STORE_REGISTRY[provider]
        return cls(
            persist_directory=config.persist_directory,
            **kwargs,
        )

    @staticmethod
    def list_providers() -> list[str]:
        """列出所有已注册的 VectorStore 提供者名称。

        返回:
            提供者名称列表（已排序）
        """
        return sorted(_VECTOR_STORE_REGISTRY.keys())
