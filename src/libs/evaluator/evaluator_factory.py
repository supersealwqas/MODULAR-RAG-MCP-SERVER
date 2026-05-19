"""评估器工厂模块，根据配置创建对应的 Evaluator 实例。

使用注册表模式，通过装饰器注册各提供者实现，工厂根据配置自动路由。
内置 CustomEvaluator 作为默认实现。
"""

from __future__ import annotations

from typing import Dict, Type

from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.libs.evaluator.custom_evaluator import CustomEvaluator


# Evaluator 提供者注册表，键为提供者名称（小写），值为对应的类
_EVALUATOR_REGISTRY: Dict[str, Type[BaseEvaluator]] = {
    "custom": CustomEvaluator,  # 内置默认实现
}

# 导入 RagasEvaluator 以触发 @register_evaluator("ragas") 注册
# 放在文件末尾会导致循环导入，因此放在注册表之后
try:
    from src.observability.evaluation import ragas_evaluator  # noqa: F401
except ImportError:
    # ragas 依赖未安装时忽略，不影响 custom 提供者
    pass


def register_evaluator(provider: str):
    """注册 Evaluator 提供者的装饰器。

    用法:
        @register_evaluator("ragas")
        class RagasEvaluator(BaseEvaluator):
            ...

    参数:
        provider: 提供者名称（如 "custom"、"ragas"），会自动转为小写
    """
    def decorator(cls: Type[BaseEvaluator]) -> Type[BaseEvaluator]:
        _EVALUATOR_REGISTRY[provider.lower()] = cls
        return cls
    return decorator


class EvaluatorFactory:
    """Evaluator 实例工厂，根据配置创建对应的评估器实例。"""

    @staticmethod
    def create(provider: str = "custom", **kwargs) -> BaseEvaluator:
        """根据提供者名称创建 Evaluator 实例。

        参数:
            provider: 提供者名称（如 "custom"、"ragas"）
            **kwargs: 提供者特定参数

        返回:
            BaseEvaluator 子类实例

        异常:
            ValueError: 提供者未注册时抛出
        """
        provider_lower = provider.lower()
        if provider_lower not in _EVALUATOR_REGISTRY:
            available = ", ".join(sorted(_EVALUATOR_REGISTRY.keys())) or "(none)"
            raise ValueError(
                f"未知的评估器提供者: '{provider}'，"
                f"可用提供者: {available}"
            )

        cls = _EVALUATOR_REGISTRY[provider_lower]
        return cls(**kwargs)

    @staticmethod
    def list_providers() -> list[str]:
        """列出所有已注册的 Evaluator 提供者名称。

        返回:
            提供者名称列表（已排序）
        """
        return sorted(_EVALUATOR_REGISTRY.keys())
