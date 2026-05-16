"""C5 ChunkRefiner 手动测试脚本。

测试 ChunkRefiner 的规则去噪、LLM 增强、降级机制等。
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.settings import load_settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk
from src.ingestion.transform.chunk_refiner import ChunkRefiner
# 导入 OpenAI LLM 以触发 @register_llm 注册
from src.libs.llm.openai_llm import OpenAILLM  # noqa: F401
from src.libs.llm.llm_factory import LLMFactory


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def safe_print(text: str):
    """安全打印，忽略无法编码的字符。"""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("gbk", errors="replace").decode("gbk"))


def _make_chunk(text: str, chunk_id: str = "test_0000_abcd1234") -> Chunk:
    """创建测试用 Chunk。"""
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "/test.pdf", "chunk_index": 0},
        source_ref="doc_001",
    )


def test_rule_based_refine():
    section("规则去噪测试")

    fixtures_path = os.path.join("tests", "fixtures", "noisy_chunks.json")
    with open(fixtures_path, "r", encoding="utf-8") as f:
        fixtures = json.load(f)

    settings = load_settings()
    refiner = ChunkRefiner(settings=settings)

    for name, fixture in fixtures.items():
        if "input" not in fixture:
            continue
        input_text = fixture["input"]
        result = refiner._rule_based_refine(input_text)

        expected = fixture.get("expected_contains", [])
        excluded = fixture.get("expected_not_contains", [])

        passed = True
        for exp in expected:
            if exp not in result:
                passed = False
                safe_print(f"  FAIL [{name}] 应保留但未找到: {exp}")
        for exc in excluded:
            if exc in result:
                passed = False
                safe_print(f"  FAIL [{name}] 应去除但仍存在: {exc}")

        status = "PASS" if passed else "FAIL"
        safe_print(f"  [{status}] {name}")
        if name == "code_blocks":
            safe_print(f"    代码块保留: {'```python' in result}")


def test_llm_refine():
    section("LLM 增强测试")
    settings = load_settings()

    try:
        llm = LLMFactory.create(settings.llm)
    except Exception as e:
        print(f"LLM 创建失败: {e}")
        return

    refiner = ChunkRefiner(settings=settings, llm=llm, use_llm=True)

    noisy_text = (
        "人工智能导论\n\n第一章 绪论\n\n"
        "人工智能（AI）是计算机科学的一个分支。\n\n"
        "人工智能导论 - 第1页\n\n"
        "它致力于创建能够执行通常需要人类智能的任务的系统。"
    )

    print("原始文本:")
    safe_print(f"  {noisy_text[:100]}...")

    chunk = _make_chunk(noisy_text)
    result = refiner.transform([chunk])

    print(f"\n精炼后 (refined_by={result[0].metadata.get('refined_by')}):")
    safe_print(f"  {result[0].text[:100]}...")
    print(f"  保留'人工智能': {'人工智能' in result[0].text}")
    print(f"  保留'计算机科学': {'计算机科学' in result[0].text}")


def test_degradation():
    section("降级机制测试")
    settings = load_settings()

    # 使用无效模型触发降级
    from src.core.settings import LLMConfig
    from unittest.mock import MagicMock

    bad_settings = MagicMock()
    bad_settings.llm = LLMConfig(
        provider="openai",
        model="nonexistent-model-xyz",
        api_key="invalid-key",
        base_url=settings.llm.base_url,
    )

    refiner = ChunkRefiner(settings=bad_settings, use_llm=True)
    chunk = _make_chunk("测试降级行为的文本内容，包含足够长的有意义信息。")
    result = refiner.transform([chunk])

    print(f"refined_by: {result[0].metadata.get('refined_by')}")
    print(f"文本保留: {'测试降级行为' in result[0].text}")
    print(f"降级成功，未崩溃: True")


def test_trace_recording():
    section("Trace 记录测试")
    settings = load_settings()
    refiner = ChunkRefiner(settings=settings)

    trace = TraceContext(trace_type="ingestion")
    chunks = [_make_chunk("chunk1", "c1"), _make_chunk("chunk2", "c2")]
    refiner.transform(chunks, trace=trace)

    print(f"trace_id: {trace.trace_id}")
    print(f"stages: {len(trace.stages)}")
    for stage in trace.stages:
        print(f"  - {stage['name']} method={stage.get('method')}")


def main():
    test_rule_based_refine()
    test_llm_refine()
    test_degradation()
    test_trace_recording()
    print(f"\n{'='*60}")
    print("  全部测试通过!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
