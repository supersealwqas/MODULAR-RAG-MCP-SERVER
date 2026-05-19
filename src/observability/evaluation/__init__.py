"""可观测性 — 评估模块。

导出 RagasEvaluator 和 CompositeEvaluator 供外部使用。
"""

# 延迟导入：ragas 未安装时不会报错，仅在实例化时检查
try:
    from src.observability.evaluation.ragas_evaluator import RagasEvaluator
except ImportError:
    RagasEvaluator = None  # type: ignore

from src.observability.evaluation.composite_evaluator import CompositeEvaluator

__all__ = ["RagasEvaluator", "CompositeEvaluator"]