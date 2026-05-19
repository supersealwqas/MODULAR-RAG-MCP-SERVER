"""评估运行脚本。

读取黄金测试集，执行 HybridSearch 检索，输出评估指标。

用法:
    python scripts/evaluate.py [--test-set path] [--top-k 10] [--collection xxx] [--verbose]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.settings import load_settings
from src.observability.evaluation.eval_runner import EvalRunner

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """主入口函数。"""
    parser = argparse.ArgumentParser(
        description="运行 RAG 评估：读取黄金测试集，执行检索，输出指标"
    )
    parser.add_argument(
        "--test-set",
        default="tests/fixtures/golden_test_set.json",
        help="黄金测试集路径（默认: tests/fixtures/golden_test_set.json）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="检索返回的结果数（默认: 10）",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="限定检索集合（可选）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出详细日志",
    )

    args = parser.parse_args()

    # 加载配置
    try:
        settings = load_settings()
    except Exception as e:
        logger.error("加载配置失败: %s", e)
        sys.exit(1)

    # 创建 EvalRunner
    runner = EvalRunner(settings)

    # 运行评估
    logger.info("开始评估: test_set=%s, top_k=%d", args.test_set, args.top_k)
    report = runner.run(
        test_set_path=args.test_set,
        top_k=args.top_k,
        collection=args.collection,
        verbose=args.verbose,
    )

    # 输出结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"测试用例数: {report.total_cases}")
    print("\n汇总指标:")
    for metric, value in report.summary.items():
        print(f"  {metric}: {value:.4f}")

    if args.verbose and report.results:
        print("\n详细结果:")
        for i, result in enumerate(report.results):
            print(f"\n  [{i + 1}] query: {result.query[:50]}...")
            for metric, value in result.metrics.items():
                print(f"      {metric}: {value:.4f}")
            if result.metadata.get("error"):
                print(f"      error: {result.metadata['error']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
