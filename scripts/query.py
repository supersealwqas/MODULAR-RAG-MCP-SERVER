"""在线查询命令行入口脚本。

调用完整的 HybridSearch + Reranker 流程并输出检索结果。
用法:
    python scripts/query.py --query "如何配置 Ollama？"
    python scripts/query.py --query "什么是 RAG？" --top-k 5 --verbose
    python scripts/query.py --query "语言模型" --collection my_docs --no-rerank
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# 修复 Windows 终端中文/emoji 编码问题
if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower().replace("-", "") != "utf8":
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# 确保项目根目录在 sys.path 中
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.reranker import Reranker
from src.core.settings import load_settings
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult

# 日志配置：输出到 stderr，stdout 留给结果
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def _format_result(r: RetrievalResult, index: int) -> str:
    """格式化单条检索结果。

    参数:
        r: 检索结果
        index: 结果序号（从 1 开始）

    返回:
        格式化的字符串
    """
    source = r.metadata.get("source_path", r.metadata.get("source", "N/A"))
    page = r.metadata.get("page", "")
    page_str = f" (p.{page})" if page else ""

    text_preview = r.text[:200].replace("\n", " ")
    if len(r.text) > 200:
        text_preview += "..."

    lines = [
        f"  [{index}] score={r.score:.4f}  chunk_id={r.chunk_id}",
        f"      来源: {source}{page_str}",
        f"      文本: {text_preview}",
    ]
    return "\n".join(lines)


def _print_verbose_section(title: str, results: List[RetrievalResult], max_items: int = 5) -> None:
    """打印 verbose 模式的中间结果。

    参数:
        title: 区块标题
        results: 结果列表
        max_items: 最多显示条数
    """
    print(f"\n  ── {title} ({len(results)} 条) ──", file=sys.stderr)
    for i, r in enumerate(results[:max_items]):
        print(f"    [{i+1}] {r.chunk_id} score={r.score:.4f}", file=sys.stderr)
    if len(results) > max_items:
        print(f"    ... 还有 {len(results) - max_items} 条", file=sys.stderr)


def run_query(args: argparse.Namespace) -> int:
    """执行查询流程。

    参数:
        args: 解析后的命令行参数

    返回:
        退出码（0=成功，1=失败）
    """
    # 加载配置
    settings = load_settings(args.config)

    # 构建 filters
    filters: Dict[str, Any] = {}
    if args.collection:
        filters["collection"] = args.collection

    trace = TraceContext(trace_type="query")

    # 初始化组件
    print("🔍 正在初始化检索组件...", file=sys.stderr)
    hybrid_search = HybridSearch(settings)

    # 执行混合检索
    print(f"🔎 查询: {args.query}", file=sys.stderr)
    start_time = time.time()

    search_results = hybrid_search.search(
        query=args.query,
        top_k=args.top_k,
        filters=filters if filters else None,
        trace=trace,
    )

    if not search_results:
        print("\n📭 未找到相关文档。", file=sys.stderr)
        print("   提示: 请先运行 python scripts/ingest.py --path <文档路径> 摄取数据。", file=sys.stderr)
        return 0

    # Verbose 模式：显示 HybridSearch 中间结果
    if args.verbose:
        _print_verbose_section("HybridSearch 结果", search_results)

    # Reranker 阶段
    if args.no_rerank:
        final_results = search_results[:args.top_k]
        rerank_fallback = False
        rerank_ms = 0.0
    else:
        reranker = Reranker(settings)
        rerank_result = reranker.rerank(
            query=args.query,
            candidates=search_results,
            top_k=args.top_k,
            trace=trace,
        )
        final_results = rerank_result["results"]
        rerank_fallback = rerank_result["fallback"]
        rerank_ms = rerank_result["elapsed_ms"]

        if args.verbose:
            _print_verbose_section("Reranker 结果", final_results)

    total_ms = (time.time() - start_time) * 1000

    # 输出结果
    print(f"\n{'=' * 60}")
    print(f"📋 检索结果 (Top-{args.top_k})")
    print(f"{'=' * 60}")

    for i, r in enumerate(final_results, 1):
        print(_format_result(r, i))
        print()

    # 汇总信息
    print(f"{'─' * 60}")
    print(f"  结果数: {len(final_results)}")
    print(f"  总耗时: {total_ms:.0f}ms")
    if not args.no_rerank:
        fallback_str = " (回退)" if rerank_fallback else ""
        print(f"  Rerank: {rerank_ms:.0f}ms{fallback_str}")
    print(f"{'─' * 60}")

    # Verbose 模式：显示 Trace 阶段
    if args.verbose:
        print(f"\n{'─' * 60}", file=sys.stderr)
        print("  Trace 阶段:", file=sys.stderr)
        for stage in trace.stages:
            name = stage["name"]
            elapsed = stage.get("elapsed_ms", "N/A")
            print(f"    {name}: {elapsed}ms", file=sys.stderr)
        print(f"{'─' * 60}", file=sys.stderr)

    return 0


def parse_args(argv: list = None) -> argparse.Namespace:
    """解析命令行参数。

    参数:
        argv: 参数列表（默认读取 sys.argv）

    返回:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description="在线查询脚本 — 检索并输出 Top-K 相关文档",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/query.py --query "如何配置 Ollama？"
  python scripts/query.py --query "什么是 RAG？" --top-k 5 --verbose
  python scripts/query.py --query "语言模型" --collection my_docs --no-rerank
        """,
    )
    parser.add_argument(
        "--query",
        required=True,
        help="查询文本",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="返回结果数量（默认: 10）",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="限定检索集合",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示各阶段中间结果",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="跳过 Reranker 阶段",
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="配置文件路径（默认: config/settings.yaml）",
    )
    return parser.parse_args(argv)


def main(argv: list = None) -> int:
    """主入口。

    参数:
        argv: 参数列表（默认读取 sys.argv）

    返回:
        退出码
    """
    args = parse_args(argv)
    return run_query(args)


if __name__ == "__main__":
    sys.exit(main())
