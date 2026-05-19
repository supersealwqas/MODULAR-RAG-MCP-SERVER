"""Ragas 评估器模块。

封装 Ragas 框架，实现 BaseEvaluator 接口，支持 Faithfulness、
Answer Relevancy、Context Precision 等指标。
Ragas 未安装时抛出明确的 ImportError 提示。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.libs.evaluator.base_evaluator import (
    BaseEvaluator,
    EvalCase,
    EvalReport,
    EvalResult,
)
from src.libs.evaluator.evaluator_factory import register_evaluator


def _check_ragas_available():
    """检查 ragas 是否可用，不可用时抛出明确提示。"""
    try:
        import ragas  # noqa: F401
    except ImportError:
        raise ImportError(
            "ragas 未安装。请执行: pip install ragas 或 uv add ragas"
        )


@register_evaluator("ragas")
class RagasEvaluator(BaseEvaluator):
    """基于 Ragas 框架的评估器。

    支持的指标:
    - faithfulness: 回答忠实度（回答是否基于检索到的上下文）
    - answer_relevancy: 回答相关性（回答与问题的相关程度）
    - context_precision: 上下文精确率（检索到的上下文中有多少是相关的）

    依赖:
        需要安装 ragas 库（pip install ragas）。
        需要配置 LLM 供 Ragas 内部使用（通过 kwargs 传入 llm 实例）。
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        llm: Any = None,
        embeddings: Any = None,
    ):
        """初始化 Ragas 评估器。

        参数:
            metrics: 要计算的指标列表，默认 ["faithfulness", "answer_relevancy", "context_precision"]
            llm: Ragas 使用的 LLM 实例（可选，Ragas 会尝试自动加载）
            embeddings: Ragas 使用的 Embedding 实例（可选）
        """
        _check_ragas_available()
        self.metrics = metrics or [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
        ]
        self._llm = llm
        self._embeddings = embeddings

    def evaluate(self, cases: List[EvalCase], **kwargs) -> EvalReport:
        """批量评估多个用例。

        将 EvalCase 转换为 Ragas 格式，调用 ragas.evaluate() 计算指标。

        参数:
            cases: 评估用例列表
            **kwargs: 额外参数
                llm: 覆盖默认 LLM
                embeddings: 覆盖默认 Embedding

        返回:
            EvalReport 评估报告
        """
        import ragas
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
        )

        # 指标映射
        metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
        }
        selected_metrics = [
            metric_map[m] for m in self.metrics if m in metric_map
        ]

        if not selected_metrics:
            return EvalReport(results=[
                EvalResult(query=c.query, metrics={}) for c in cases
            ])

        # 构造 Ragas 评估数据集
        dataset = self._build_dataset(cases)
        if not dataset:
            return EvalReport(results=[
                EvalResult(query=c.query, metrics={}) for c in cases
            ])

        # 调用 Ragas 评估
        try:
            llm = kwargs.get("llm", self._llm)
            embeddings = kwargs.get("embeddings", self._embeddings)

            evaluate_kwargs: Dict[str, Any] = {
                "metrics": selected_metrics,
            }
            if llm is not None:
                evaluate_kwargs["llm"] = llm
            if embeddings is not None:
                evaluate_kwargs["embeddings"] = embeddings

            result = ragas.evaluate(dataset, **evaluate_kwargs)

            # 解析结果
            return self._parse_results(result, cases)

        except Exception as e:
            # Ragas 调用失败时返回空指标
            return EvalReport(results=[
                EvalResult(
                    query=c.query,
                    metrics={},
                    metadata={"error": str(e)},
                )
                for c in cases
            ])

    def _build_dataset(self, cases: List[EvalCase]):
        """将 EvalCase 列表转换为 Ragas Dataset。

        参数:
            cases: 评估用例列表

        返回:
            HuggingFace Dataset 或 None（无有效数据时）
        """
        from datasets import Dataset

        valid_cases = []
        for case in cases:
            # Ragas 要求 generated_answer 和 reference_answer 非空
            if not case.generated_answer or not case.reference_answer:
                continue
            valid_cases.append(case)

        if not valid_cases:
            return None

        data = {
            "question": [c.query for c in valid_cases],
            "answer": [c.generated_answer for c in valid_cases],
            "contexts": [c.retrieved_ids for c in valid_cases],
            "ground_truth": [c.reference_answer for c in valid_cases],
        }

        return Dataset.from_dict(data)

    def _parse_results(
        self, ragas_result, cases: List[EvalCase]
    ) -> EvalReport:
        """解析 Ragas 评估结果为 EvalReport。

        参数:
            ragas_result: ragas.evaluate() 返回的结果
            cases: 原始评估用例列表

        返回:
            EvalReport 评估报告
        """
        results = []

        # ragas_result 是一个 Dataset，每行包含各指标的分数
        for i, case in enumerate(cases):
            if i < len(ragas_result):
                row = ragas_result[i]
                metrics = {}
                for metric_name in self.metrics:
                    value = row.get(metric_name)
                    if value is not None:
                        metrics[metric_name] = float(value)
                results.append(EvalResult(
                    query=case.query,
                    metrics=metrics,
                    metadata=case.metadata,
                ))
            else:
                results.append(EvalResult(
                    query=case.query,
                    metrics={},
                ))

        return EvalReport(results=results)
