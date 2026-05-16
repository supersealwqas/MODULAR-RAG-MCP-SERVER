"""离线数据摄取脚本。

支持单文件或目录批量摄取，通过 IngestionPipeline 完成完整流程。
用法:
    python scripts/ingest.py --path data/documents/LLM基础知识.pdf
    python scripts/ingest.py --path data/documents/ --collection my_docs
    python scripts/ingest.py --path data/documents/ --force
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List

# 确保项目根目录在 sys.path 中
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.settings import load_settings
from src.core.trace.trace_context import TraceContext
from src.ingestion.pipeline import IngestionPipeline, PipelineError, PipelineResult

# 支持的文件扩展名
_SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".docx", ".pptx"}

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def _collect_files(path: Path) -> List[Path]:
    """收集待摄取的文件列表。

    参数:
        path: 文件或目录路径

    返回:
        排序后的文件路径列表

    异常:
        FileNotFoundError: 路径不存在
        ValueError: 目录下无支持格式的文件
    """
    if not path.exists():
        raise FileNotFoundError(f"路径不存在: {path}")

    if path.is_file():
        return [path]

    # 目录模式：递归查找支持格式的文件
    files = []
    for ext in _SUPPORTED_EXTENSIONS:
        files.extend(path.rglob(f"*{ext}"))

    files.sort()
    if not files:
        raise ValueError(
            f"目录下未找到支持格式的文件 ({', '.join(_SUPPORTED_EXTENSIONS)}): {path}"
        )
    return files


def _format_stage_times(stage_times: dict) -> str:
    """格式化各阶段耗时为可读字符串。

    参数:
        stage_times: 阶段耗时字典（毫秒）

    返回:
        格式化的字符串
    """
    if not stage_times:
        return ""
    parts = [f"{name}={ms:.0f}ms" for name, ms in stage_times.items()]
    return " | ".join(parts)


def _print_progress(stage: str, current: int, total: int) -> None:
    """打印进度信息到 stderr。

    参数:
        stage: 当前阶段名称
        current: 当前步骤（从 1 开始）
        total: 总步骤数
    """
    icons = {
        "integrity": "🔍",
        "load": "📄",
        "split": "✂️",
        "transform": "🔧",
        "encode": "🧠",
        "store": "💾",
    }
    icon = icons.get(stage, "▶")
    print(f"  {icon} [{current}/{total}] {stage}", file=sys.stderr)


def _print_result(result: PipelineResult, index: int, total: int) -> None:
    """打印单文件处理结果。

    参数:
        result: PipelineResult 结果
        index: 当前文件序号（从 1 开始）
        total: 总文件数
    """
    prefix = f"[{index}/{total}]"
    if result.skipped:
        print(f"  {prefix} ⏭ 已跳过 (增量): {result.file_path}", file=sys.stderr)
    else:
        print(
            f"  {prefix} ✅ 完成: {result.file_path}",
            f"| {result.chunk_count} chunks, {result.record_count} records",
            f"| {result.elapsed_ms:.0f}ms",
            file=sys.stderr,
        )


def run_ingest(args: argparse.Namespace) -> int:
    """执行摄取流程。

    参数:
        args: 解析后的命令行参数

    返回:
        退出码（0=成功，1=有文件失败）
    """
    # 加载配置
    settings = load_settings(args.config)
    logger.info("配置已加载: %s", args.config)

    # 收集文件
    path = Path(args.path)
    try:
        files = _collect_files(path)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    print(f"📁 发现 {len(files)} 个待处理文件", file=sys.stderr)
    print(f"📦 目标集合: {args.collection}", file=sys.stderr)
    if args.force:
        print("⚡ 强制模式: 忽略增量检查", file=sys.stderr)
    print(file=sys.stderr)

    # 创建 Pipeline（所有组件懒创建）
    pipeline = IngestionPipeline(settings)

    # 逐文件处理
    success_count = 0
    skip_count = 0
    fail_count = 0
    total_start = time.time()

    for i, file_path in enumerate(files, 1):
        try:
            trace = TraceContext(trace_type="ingestion")
            result = pipeline.run(
                file_path=str(file_path),
                collection=args.collection,
                force=args.force,
                trace=trace,
                on_progress=_print_progress,
            )
            _print_result(result, i, len(files))

            if result.skipped:
                skip_count += 1
            else:
                success_count += 1

        except PipelineError as e:
            fail_count += 1
            print(
                f"  [{i}/{len(files)}] ❌ 失败: {file_path}",
                f"| 阶段: {e.stage} | 错误: {e.original_error}",
                file=sys.stderr,
            )
            logger.error("Pipeline 失败: %s, stage=%s", file_path, e.stage)

        except Exception as e:
            fail_count += 1
            print(
                f"  [{i}/{len(files)}] ❌ 异常: {file_path} | {e}",
                file=sys.stderr,
            )
            logger.error("未预期异常: %s", file_path, exc_info=True)

    # 打印汇总
    total_elapsed = (time.time() - total_start) * 1000
    print(file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    print(f"📊 摄取完成", file=sys.stderr)
    print(f"   成功: {success_count}", file=sys.stderr)
    print(f"   跳过: {skip_count}", file=sys.stderr)
    print(f"   失败: {fail_count}", file=sys.stderr)
    print(f"   总耗时: {total_elapsed:.0f}ms", file=sys.stderr)
    print("=" * 50, file=sys.stderr)

    return 1 if fail_count > 0 else 0


def parse_args(argv: list = None) -> argparse.Namespace:
    """解析命令行参数。

    参数:
        argv: 参数列表（默认读取 sys.argv）

    返回:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description="离线数据摄取脚本 — 将文件摄取到向量库和 BM25 索引",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/ingest.py --path data/documents/LLM基础知识.pdf
  python scripts/ingest.py --path data/documents/ --collection my_docs
  python scripts/ingest.py --path data/documents/ --force
        """,
    )
    parser.add_argument(
        "--path",
        required=True,
        help="待摄取的文件或目录路径",
    )
    parser.add_argument(
        "--collection",
        default="default",
        help="目标集合名称（默认: default）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新处理（忽略增量检查）",
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
    return run_ingest(args)


if __name__ == "__main__":
    sys.exit(main())
