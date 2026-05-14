"""文本切分器工厂模块，根据配置创建对应的 Splitter 实例。

使用注册表模式，通过装饰器注册各策略实现，工厂根据配置自动路由。
"""

from __future__ import annotations

from typing import Dict, Type

from src.libs.splitter.base_splitter import BaseSplitter


# Splitter 策略注册表，键为策略名称（小写），值为对应的类
_SPLITTER_REGISTRY: Dict[str, Type[BaseSplitter]] = {}


def register_splitter(strategy: str):
    """注册 Splitter 策略的装饰器。

    用法:
        @register_splitter("recursive")
        class RecursiveSplitter(BaseSplitter):
            ...

    参数:
        strategy: 策略名称（如 "recursive"、"semantic"、"fixed"），会自动转为小写
    """
    def decorator(cls: Type[BaseSplitter]) -> Type[BaseSplitter]:
        _SPLITTER_REGISTRY[strategy.lower()] = cls
        return cls
    return decorator


class SplitterFactory:
    """Splitter 实例工厂，根据策略配置创建对应的切分器实例。"""

    @staticmethod
    def create(
        strategy: str = "recursive",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        **kwargs,
    ) -> BaseSplitter:
        """根据策略创建 Splitter 实例。

        参数:
            strategy: 切分策略名称（如 "recursive"、"semantic"、"fixed"）
            chunk_size: 每个文本块的最大字符数
            chunk_overlap: 相邻文本块之间的重叠字符数
            **kwargs: 其他提供者特定参数

        返回:
            BaseSplitter 子类实例

        异常:
            ValueError: 策略未注册时抛出
        """
        strategy_lower = strategy.lower()
        if strategy_lower not in _SPLITTER_REGISTRY:
            available = ", ".join(sorted(_SPLITTER_REGISTRY.keys())) or "(none)"
            raise ValueError(
                f"未知的切分策略: '{strategy}'，"
                f"可用策略: {available}"
            )

        cls = _SPLITTER_REGISTRY[strategy_lower]
        return cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

    @staticmethod
    def list_strategies() -> list[str]:
        """列出所有已注册的切分策略名称。

        返回:
            策略名称列表（已排序）
        """
        return sorted(_SPLITTER_REGISTRY.keys())
