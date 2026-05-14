"""重排序器工厂模块，根据配置创建对应的 Reranker 实例。

使用注册表模式，通过装饰器注册各提供者实现，工厂根据配置自动路由。
内置 NoneReranker 作为禁用重排序时的默认回退。
"""

from __future__ import annotations

from typing import Dict, Type

from src.libs.reranker.base_reranker import BaseReranker, NoneReranker


# Reranker 提供者注册表，键为提供者名称（小写），值为对应的类
_RERANKER_REGISTRY: Dict[str, Type[BaseReranker]] = {
    "none": NoneReranker,  # 内置默认回退
}


def register_reranker(provider: str):
    """注册 Reranker 提供者的装饰器。

    用法:
        @register_reranker("cross-encoder")
        class CrossEncoderReranker(BaseReranker):
            ...

    参数:
        provider: 提供者名称（如 "cross-encoder"、"llm"），会自动转为小写
    """
    def decorator(cls: Type[BaseReranker]) -> Type[BaseReranker]:
        _RERANKER_REGISTRY[provider.lower()] = cls
        return cls
    return decorator


class RerankerFactory:
    """Reranker 实例工厂，根据配置创建对应的重排序器实例。"""

    @staticmethod
    def create(provider: str = "none", **kwargs) -> BaseReranker:
        """根据提供者名称创建 Reranker 实例。

        参数:
            provider: 提供者名称（如 "none"、"cross-encoder"、"llm"）
            **kwargs: 提供者特定参数

        返回:
            BaseReranker 子类实例

        异常:
            ValueError: 提供者未注册时抛出
        """
        provider_lower = provider.lower()
        if provider_lower not in _RERANKER_REGISTRY:
            available = ", ".join(sorted(_RERANKER_REGISTRY.keys())) or "(none)"
            raise ValueError(
                f"未知的重排序提供者: '{provider}'，"
                f"可用提供者: {available}"
            )

        cls = _RERANKER_REGISTRY[provider_lower]
        return cls(**kwargs)

    @staticmethod
    def list_providers() -> list[str]:
        """列出所有已注册的 Reranker 提供者名称。

        返回:
            提供者名称列表（已排序）
        """
        return sorted(_RERANKER_REGISTRY.keys())
