"""评估运行器模块。

读取黄金测试集，执行 HybridSearch 检索，使用 Evaluator 计算指标。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.query_engine.hybrid_search import HybridSearch
from src.core.settings import Settings
from src.libs.evaluator.base_evaluator import (
    BaseEvaluator,
    EvalCase,
    EvalReport,
)
from src.libs.evaluator.evaluator_factory import EvaluatorFactory

logger = logging.getLogger(__name__)


class EvalRunner:
    """评估运行器：读取黄金测试集，执行检索并评估。

    属性:
        settings: 全局配置
        hybrid_search: 混合检索实例
        evaluator: 评估器实例
    """

    def __init__(
        self,
        settings: Settings,
        hybrid_search: Optional[HybridSearch] = None,
        evaluator: Optional[BaseEvaluator] = None,
    ):
        """初始化评估运行器。

        参数:
            settings: 全局配置对象
            hybrid_search: HybridSearch 实例（可选，不传则延迟创建）
            evaluator: Evaluator 实例（可选，不传则根据配置创建）
        """
        self._settings = settings
        self._hybrid_search = hybrid_search
        self._evaluator = evaluator

    def _get_hybrid_search(self) -> HybridSearch:
        """获取 HybridSearch 实例（延迟创建）。"""
        if self._hybrid_search is None:
            self._hybrid_search = HybridSearch(self._settings)
        return self._hybrid_search

    def _get_evaluator(self) -> BaseEvaluator:
        """获取 Evaluator 实例（延迟创建）。"""
        if self._evaluator is None:
            # 从配置读取评估后端，默认使用 custom
            backends = self._settings.evaluation.backends if hasattr(self._settings, 'evaluation') else ["custom"]
            backend = backends[0] if backends else "custom"
            self._evaluator = EvaluatorFactory.create(backend)
        return self._evaluator

    def run(
        self,
        test_set_path: str,
        top_k: int = 10,
        collection: Optional[str] = None,
        verbose: bool = False,
    ) -> EvalReport:
        """运行评估。

        读取黄金测试集，对每个 query 执行检索，计算指标。

        参数:
            test_set_path: 黄金测试集文件路径（JSON 格式）
            top_k: 检索返回的结果数
            collection: 限定检索集合（可选）
            verbose: 是否输出详细日志

        返回:
            EvalReport 评估报告
        """
        # 1. 加载黄金测试集
        test_cases = self._load_test_set(test_set_path)
        if not test_cases:
            logger.warning("黄金测试集为空: %s", test_set_path)
            return EvalReport(results=[])

        if verbose:
            logger.info("加载 %d 条测试用例", len(test_cases))

        # 2. 对每个 query 执行检索并构造 EvalCase
        eval_cases: List[EvalCase] = []
        hybrid_search = self._get_hybrid_search()

        for i, test_case in enumerate(test_cases):
            query = test_case.get("query", "")
            expected_chunk_ids = test_case.get("expected_chunk_ids", [])
            expected_sources = test_case.get("expected_sources", [])

            if not query:
                logger.warning("跳过空 query 的测试用例 (index=%d)", i)
                continue

            # 执行检索
            try:
                filters = {}
                if collection:
                    filters["collection"] = collection

                results = hybrid_search.search(
                    query, top_k=top_k, filters=filters if filters else None,
                )
                retrieved_ids = [r.chunk_id for r in results]

                if verbose:
                    logger.info(
                        "[%d/%d] query='%s' → %d 结果",
                        i + 1, len(test_cases), query[:50], len(retrieved_ids),
                    )

            except Exception as e:
                logger.error("检索失败 (query='%s'): %s", query[:50], e)
                retrieved_ids = []

            # 构造 EvalCase
            eval_cases.append(EvalCase(
                query=query,
                retrieved_ids=retrieved_ids,
                golden_ids=expected_chunk_ids,
                metadata={
                    "expected_sources": expected_sources,
                    "index": i,
                },
            ))

        # 3. 执行评估
        evaluator = self._get_evaluator()
        report = evaluator.evaluate(eval_cases)

        if verbose:
            logger.info("评估完成: %d 条用例, summary=%s", report.total_cases, report.summary)

        return report

    def _load_test_set(self, path: str) -> List[Dict[str, Any]]:
        """加载黄金测试集。

        参数:
            path: JSON 文件路径

        返回:
            测试用例列表
        """
        file_path = Path(path)
        if not file_path.exists():
            logger.error("黄金测试集文件不存在: %s", path)
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 支持两种格式：
            # 1. {"test_cases": [...]}
            # 2. 直接是 [...]
            if isinstance(data, dict):
                return data.get("test_cases", [])
            elif isinstance(data, list):
                return data
            else:
                logger.error("不支持的测试集格式: %s", type(data))
                return []

        except json.JSONDecodeError as e:
            logger.error("JSON 解析失败: %s", e)
            return []
        except Exception as e:
            logger.error("加载测试集失败: %s", e)
            return []

    def run_from_dict(
        self,
        test_cases: List[Dict[str, Any]],
        top_k: int = 10,
        collection: Optional[str] = None,
    ) -> EvalReport:
        """从字典列表直接运行评估（用于测试）。

        参数:
            test_cases: 测试用例字典列表
            top_k: 检索返回的结果数
            collection: 限定检索集合

        返回:
            EvalReport 评估报告
        """
        eval_cases: List[EvalCase] = []
        hybrid_search = self._get_hybrid_search()

        for test_case in test_cases:
            query = test_case.get("query", "")
            expected_chunk_ids = test_case.get("expected_chunk_ids", [])

            if not query:
                continue

            try:
                filters = {}
                if collection:
                    filters["collection"] = collection
                results = hybrid_search.search(
                    query, top_k=top_k, filters=filters if filters else None,
                )
                retrieved_ids = [r.chunk_id for r in results]
            except Exception:
                retrieved_ids = []

            eval_cases.append(EvalCase(
                query=query,
                retrieved_ids=retrieved_ids,
                golden_ids=expected_chunk_ids,
            ))

        evaluator = self._get_evaluator()
        return evaluator.evaluate(eval_cases)
