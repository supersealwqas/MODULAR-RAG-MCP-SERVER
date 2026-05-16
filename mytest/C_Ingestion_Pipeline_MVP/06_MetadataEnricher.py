"""C6 MetadataEnricher 手动测试脚本。

测试 MetadataEnricher 的规则增强、LLM 增强、降级机制、JSON 解析等。
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.settings import load_settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk
from src.ingestion.transform.metadata_enricher import MetadataEnricher
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


def test_rule_based_enrich():
    section("规则增强测试")
    settings = load_settings()
    enricher = MetadataEnricher(settings=settings)

    test_cases = [
        ("Markdown 标题", "# RAG 系统概述\n\nRAG（检索增强生成）是一种结合检索和生成的技术。它通过检索相关文档来增强 LLM 的回答质量。"),
        ("无标题文本", "机器学习是人工智能的重要分支。深度学习使用多层神经网络进行特征学习。自然语言处理关注人机交互。"),
        ("英文文本", "# Machine Learning Basics\n\nMachine learning is a subset of artificial intelligence. It enables systems to learn from data."),
        ("短文本", "这是一段很短的文本。"),
    ]

    for name, text in test_cases:
        result = enricher._rule_based_enrich(text)
        safe_print(f"\n  [{name}]")
        safe_print(f"    title:   {result['title']}")
        safe_print(f"    summary: {result['summary'][:60]}...")
        safe_print(f"    tags:    {result['tags']}")

        # 验证非空
        assert result["title"], f"{name}: title 不能为空"
        assert result["summary"], f"{name}: summary 不能为空"
        assert len(result["tags"]) > 0, f"{name}: tags 不能为空"
        print(f"    PASS: 所有字段非空")


def test_llm_enrich():
    section("LLM 增强测试")
    settings = load_settings()

    try:
        llm = LLMFactory.create(settings.llm)
    except Exception as e:
        print(f"LLM 创建失败: {e}")
        return

    enricher = MetadataEnricher(settings=settings, llm=llm, use_llm=True)

    text = (
        "# RAG 技术详解\n\n"
        "RAG（Retrieval-Augmented Generation）是一种结合信息检索与文本生成的 AI 技术框架。"
        "它通过从外部知识库中检索相关文档片段，将其作为上下文输入给大语言模型，"
        "从而生成更准确、更有依据的回答。RAG 技术有效解决了 LLM 的幻觉问题，"
        "并在企业知识管理、智能客服等场景中得到广泛应用。"
    )

    chunk = _make_chunk(text)
    print("输入文本:")
    safe_print(f"  {text[:80]}...")

    result = enricher.transform([chunk])
    m = result[0].metadata

    print(f"\nenriched_by: {m.get('enriched_by')}")
    safe_print(f"  title:   {m.get('title')}")
    safe_print(f"  summary: {m.get('summary', '')[:80]}...")
    safe_print(f"  tags:    {m.get('tags')}")

    if m.get("enriched_by") == "llm":
        print("  PASS: LLM 增强成功")
    else:
        print("  WARN: 降级到规则模式")


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

    enricher = MetadataEnricher(settings=bad_settings, use_llm=True)
    chunk = _make_chunk("# 测试标题\n\n这是测试降级行为的文本内容。包含一些关键词信息。")
    result = enricher.transform([chunk])
    m = result[0].metadata

    print(f"enriched_by: {m.get('enriched_by')}")
    safe_print(f"  title:   {m.get('title')}")
    safe_print(f"  tags:    {m.get('tags')}")
    print(f"  降级成功，未崩溃: {m.get('enriched_by') == 'rule'}")


def test_llm_json_parsing():
    section("LLM JSON 解析测试")
    settings = load_settings()
    enricher = MetadataEnricher(settings=settings)

    test_cases = [
        ("标准 JSON", '{"title": "测试标题", "summary": "测试摘要", "tags": ["A", "B"]}'),
        ("代码块包裹", '```json\n{"title": "标题", "summary": "摘要", "tags": ["X"]}\n```'),
        ("带额外文本", '以下是结果：\n{"title": "标题", "summary": "摘要", "tags": ["Y"]}\n以上。'),
        ("缺少字段", '{"title": "只有标题"}'),
        ("非法 JSON", "这不是 JSON"),
    ]

    for name, output in test_cases:
        result = enricher._parse_llm_output(output)
        status = "PASS" if result is not None else "None"
        safe_print(f"  [{name}] -> {status}")
        if result:
            safe_print(f"    title={result['title']}, tags={result['tags']}")


def test_trace_recording():
    section("Trace 记录测试")
    settings = load_settings()
    enricher = MetadataEnricher(settings=settings)

    trace = TraceContext(trace_type="ingestion")
    chunks = [_make_chunk("# 标题1\n\n内容1。", "c1"), _make_chunk("# 标题2\n\n内容2。", "c2")]
    enricher.transform(chunks, trace=trace)

    print(f"trace_id: {trace.trace_id}")
    print(f"stages: {len(trace.stages)}")
    for stage in trace.stages:
        print(f"  - {stage['name']} method={stage.get('method')}")


def test_pipeline_integration():
    section("Pipeline 集成测试")
    settings = load_settings()

    from src.ingestion.pipeline import IngestionPipeline
    pipeline = IngestionPipeline(settings)

    # 验证 enricher 已集成
    enricher = pipeline._get_enricher()
    print(f"Pipeline enricher 类型: {type(enricher).__name__}")
    print(f"  是 MetadataEnricher: {type(enricher).__name__ == 'MetadataEnricher'}")


def main():
    test_rule_based_enrich()
    test_llm_enrich()
    test_degradation()
    test_llm_json_parsing()
    test_trace_recording()
    test_pipeline_integration()
    print(f"\n{'='*60}")
    print("  全部测试完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
