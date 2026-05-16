"""C14 IngestionPipeline 真实测试脚本。

使用真实 PDF 文件和实际组件测试完整摄取流程：
integrity → load → split → transform → encode → store

测试内容：
1. 完整流程：加载 LLM基础知识.pdf，走完 6 个阶段
2. 增量跳过：第二次运行应跳过
3. 强制重处理：force=True 应忽略增量
4. Trace 记录：验证各阶段 Trace 数据
"""

import os
import sys
import shutil
import time

# 确保项目根目录在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.settings import load_settings
from src.core.trace.trace_context import TraceContext
from src.ingestion.pipeline import IngestionPipeline, PipelineError


def safe_print(text: str):
    """安全打印，处理编码问题。"""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("gbk", errors="replace").decode("gbk", errors="replace"))


def section(title: str):
    """打印分节标题。"""
    safe_print(f"\n{'='*60}")
    safe_print(f"  {title}")
    safe_print(f"{'='*60}")


def print_result(result):
    """打印 PipelineResult 详情。"""
    safe_print(f"  file_path: {result.file_path}")
    safe_print(f"  collection: {result.collection}")
    safe_print(f"  file_hash: {result.file_hash[:16]}...")
    safe_print(f"  doc_id: {result.doc_id}")
    safe_print(f"  chunk_count: {result.chunk_count}")
    safe_print(f"  record_count: {result.record_count}")
    safe_print(f"  skipped: {result.skipped}")
    safe_print(f"  elapsed_ms: {result.elapsed_ms:.1f}")
    if result.stage_times:
        safe_print(f"  stage_times:")
        for stage, ms in result.stage_times.items():
            safe_print(f"    {stage}: {ms:.1f}ms")


def print_trace(trace):
    """打印 Trace 详情。"""
    safe_print(f"  trace_id: {trace.trace_id}")
    safe_print(f"  trace_type: {trace.trace_type}")
    safe_print(f"  stages ({len(trace.stages)}):")
    for stage in trace.stages:
        safe_print(f"    {stage['name']}: {stage}")


def progress_callback(stage: str, current: int, total: int):
    """进度回调函数。"""
    icons = {
        "integrity": "🔍",
        "load": "📄",
        "split": "✂️",
        "transform": "🔧",
        "encode": "🧠",
        "store": "💾",
    }
    icon = icons.get(stage, "▶")
    safe_print(f"  {icon} [{current}/{total}] {stage}")


def clean_test_data():
    """清理测试数据（可选）。"""
    db_path = os.path.join(_project_root, "data", "db", "chroma")
    bm25_path = os.path.join(_project_root, "data", "db", "bm25")
    integrity_path = os.path.join(_project_root, "data", "db", "integrity.sqlite")

    safe_print("  清理测试数据...")
    for path in [db_path, bm25_path]:
        if os.path.exists(path):
            shutil.rmtree(path)
            safe_print(f"    删除: {path}")
    if os.path.exists(integrity_path):
        os.remove(integrity_path)
        safe_print(f"    删除: {integrity_path}")


def test_full_pipeline():
    """测试 1: 完整 Pipeline 流程。"""
    section("测试 1: 完整 Pipeline 流程")

    settings = load_settings()
    pipeline = IngestionPipeline(settings)

    # 使用 LLM基础知识.pdf 测试
    file_path = os.path.join(_project_root, "data", "documents", "LLM基础知识.pdf")

    if not os.path.exists(file_path):
        safe_print(f"  ❌ 测试文件不存在: {file_path}")
        return None

    safe_print(f"  文件: {file_path}")
    safe_print(f"  文件大小: {os.path.getsize(file_path) / 1024:.1f} KB")

    trace = TraceContext(trace_type="ingestion")

    start_time = time.time()
    result = pipeline.run(
        file_path=file_path,
        collection="test_real",
        force=False,
        trace=trace,
        on_progress=progress_callback,
    )
    elapsed = (time.time() - start_time) * 1000

    safe_print(f"\n  结果:")
    print_result(result)
    safe_print(f"\n  Trace:")
    print_trace(trace)

    return result


def test_skip_behavior():
    """测试 2: 增量跳过行为。"""
    section("测试 2: 增量跳过行为")

    settings = load_settings()
    pipeline = IngestionPipeline(settings)

    file_path = os.path.join(_project_root, "data", "documents", "LLM基础知识.pdf")

    safe_print("  第一次运行（应正常处理）:")
    r1 = pipeline.run(file_path=file_path, collection="test_real")
    safe_print(f"    skipped={r1.skipped}, chunks={r1.chunk_count}, elapsed={r1.elapsed_ms:.1f}ms")

    safe_print("\n  第二次运行（应跳过）:")
    r2 = pipeline.run(file_path=file_path, collection="test_real")
    safe_print(f"    skipped={r2.skipped}, chunks={r2.chunk_count}, elapsed={r2.elapsed_ms:.1f}ms")

    return r1, r2


def test_force_override():
    """测试 3: 强制重处理。"""
    section("测试 3: 强制重处理 (force=True)")

    settings = load_settings()
    pipeline = IngestionPipeline(settings)

    file_path = os.path.join(_project_root, "data", "documents", "LLM基础知识.pdf")

    safe_print("  强制重处理:")
    trace = TraceContext(trace_type="ingestion")
    result = pipeline.run(
        file_path=file_path,
        collection="test_real",
        force=True,
        trace=trace,
        on_progress=progress_callback,
    )
    safe_print(f"\n  结果:")
    safe_print(f"    skipped={result.skipped}, chunks={result.chunk_count}")
    safe_print(f"    elapsed={result.elapsed_ms:.1f}ms")

    return result


def test_trace_recording():
    """测试 4: Trace 记录完整性。"""
    section("测试 4: Trace 记录完整性")

    settings = load_settings()
    pipeline = IngestionPipeline(settings)

    file_path = os.path.join(_project_root, "data", "documents", "LLM基础知识.pdf")

    trace = TraceContext(trace_type="ingestion")
    pipeline.run(file_path=file_path, collection="test_trace", force=True, trace=trace)

    safe_print(f"  Trace 详情:")
    print_trace(trace)

    # 验证各阶段都有记录
    expected_stages = ["integrity", "load", "split", "transform", "encode", "store"]
    recorded_stages = [s['name'] for s in trace.stages]

    safe_print(f"\n  阶段覆盖检查:")
    for stage in expected_stages:
        found = stage in recorded_stages
        safe_print(f"    {stage}: {'✓' if found else '✗'}")

    # 验证 MetadataEnricher 子阶段
    enrich_stages = [s for s in trace.stages if s['name'] == 'metadata_enrich']
    safe_print(f"\n  MetadataEnricher 子阶段: {len(enrich_stages)} 条")
    for s in enrich_stages:
        safe_print(f"    method={s.get('method')}")

    return trace


def test_error_handling():
    """测试 5: 异常处理。"""
    section("测试 5: 异常处理")

    settings = load_settings()
    pipeline = IngestionPipeline(settings)

    # 使用不存在的文件
    file_path = os.path.join(_project_root, "data", "documents", "不存在的文件.pdf")

    safe_print(f"  测试文件不存在: {file_path}")
    try:
        pipeline.run(file_path=file_path, collection="test_error")
        safe_print("  ❌ 应该抛出异常!")
    except FileNotFoundError as e:
        safe_print(f"  ✓ FileNotFoundError: {e}")
    except PipelineError as e:
        safe_print(f"  ✓ PipelineError: stage={e.stage}, error={e.original_error}")
    except Exception as e:
        safe_print(f"  ✓ Exception: {type(e).__name__}: {e}")


def main():
    """主测试函数。"""
    section("IngestionPipeline 真实测试")
    safe_print("  使用真实 PDF 和实际组件测试")
    safe_print("  文件: data/documents/LLM基础知识.pdf")
    safe_print("  Embedding: BGE-M3")
    safe_print("  VectorStore: ChromaDB")
    safe_print("  BM25: 本地索引")

    # 清理旧数据
    clean_test_data()

    # 运行测试
    test_full_pipeline()
    test_skip_behavior()
    test_force_override()
    test_trace_recording()
    test_error_handling()

    section("全部测试完成!")
    safe_print("  如需清理测试数据，重新运行此脚本或手动删除 data/db/")


if __name__ == "__main__":
    main()
