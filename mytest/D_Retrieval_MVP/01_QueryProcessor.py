"""D1 QueryProcessor 手动测试。

测试查询预处理：关键词提取（jieba/正则分词 + 停用词过滤）、
filters 解析、ProcessedQuery 输出。

测试覆盖：
1. 基本关键词提取（英文/中文/混合）
2. 停用词过滤
3. filters 解析
4. ProcessedQuery 集成
5. Trace 记录
6. 边界场景
"""

import sys
import os

# 修复 Windows 终端中文编码问题
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.core.settings import load_settings
from src.core.query_engine.query_processor import QueryProcessor
from src.core.trace.trace_context import TraceContext
from src.core.types import ProcessedQuery


def test_keyword_extraction(proc: QueryProcessor):
    """测试1: 关键词提取。"""
    print("\n" + "=" * 60)
    print("测试1: 关键词提取")
    print("=" * 60)

    test_cases = [
        ("How to configure Ollama?", "英文"),
        ("如何配置 Ollama 模型？", "中文"),
        ("BGE-M3 模型的 embedding 维度是多少？", "中英混合"),
    ]

    for query, lang in test_cases:
        keywords = proc.extract_keywords(query)
        print(f"\n  [{lang}] {query}")
        print(f"  → keywords: {keywords}")
        assert len(keywords) > 0, f"{lang}查询应提取到关键词"

    print("\n✅ 关键词提取测试通过")


def test_stopwords_filtering(proc: QueryProcessor):
    """测试2: 停用词过滤。"""
    print("\n" + "=" * 60)
    print("测试2: 停用词过滤")
    print("=" * 60)

    query = "What is the best way to learn Python?"
    keywords = proc.extract_keywords(query)
    print(f"\n  Query: {query}")
    print(f"  → keywords: {keywords}")

    # 英文停用词应被过滤
    stopwords_to_check = ["what", "is", "the", "to"]
    for sw in stopwords_to_check:
        assert sw not in keywords, f"停用词 '{sw}' 应被过滤"

    # 有效词应保留
    for word in ["best", "way", "learn", "python"]:
        assert word in keywords, f"有效词 '{word}' 应保留"

    print("✅ 停用词过滤测试通过")


def test_filters_parsing(proc: QueryProcessor):
    """测试3: filters 解析。"""
    print("\n" + "=" * 60)
    print("测试3: filters 解析")
    print("=" * 60)

    # 空 filters
    result = proc.parse_filters(None)
    assert result == {}, f"None filters 应返回空 dict，实际: {result}"
    print("  None filters → {} ✅")

    result = proc.parse_filters({})
    assert result == {}, f"空 dict 应返回空 dict，实际: {result}"
    print("  空 dict → {} ✅")

    # 有效 filters
    filters = {"collection": "test", "doc_type": "pdf"}
    result = proc.parse_filters(filters)
    assert result == filters, f"有效 filters 应保留，实际: {result}"
    print(f"  有效 filters → {result} ✅")

    # 含 None/空值的 filters
    filters = {"collection": "test", "doc_type": None, "source": "", "page": 5}
    result = proc.parse_filters(filters)
    assert result == {"collection": "test", "page": 5}, f"应移除 None/空值，实际: {result}"
    print(f"  含 None/空值 → {result} ✅")

    print("✅ filters 解析测试通过")


def test_process_integration(proc: QueryProcessor):
    """测试4: ProcessedQuery 集成。"""
    print("\n" + "=" * 60)
    print("测试4: ProcessedQuery 集成")
    print("=" * 60)

    # 基本处理
    result = proc.process("How to use BGE-M3?")
    assert isinstance(result, ProcessedQuery), f"应返回 ProcessedQuery，实际: {type(result)}"
    assert result.original == "How to use BGE-M3?"
    assert len(result.keywords) > 0
    print(f"  基本处理: keywords={result.keywords}")

    # 带 filters
    result = proc.process("test query", filters={"collection": "docs"})
    assert result.filters == {"collection": "docs"}
    print(f"  带 filters: filters={result.filters}")

    # 不带 filters
    result = proc.process("test query")
    assert result.filters == {}
    print(f"  不带 filters: filters={result.filters}")

    print("✅ ProcessedQuery 集成测试通过")


def test_trace_recording(proc: QueryProcessor):
    """测试5: Trace 记录。"""
    print("\n" + "=" * 60)
    print("测试5: Trace 记录")
    print("=" * 60)

    trace = TraceContext()
    proc.extract_keywords("test query", trace=trace)

    stages = [s for s in trace.stages if s["name"] == "query_processing"]
    assert len(stages) == 1, f"应记录1个阶段，实际: {len(stages)}"

    stage = stages[0]
    print(f"  记录阶段: {stage['name']}")
    print(f"  分词方法: {stage['method']}")
    print(f"  关键词数量: {stage['keyword_count']}")
    assert "method" in stage
    assert "keyword_count" in stage

    print("✅ Trace 记录测试通过")


def test_serialization(proc: QueryProcessor):
    """测试6: ProcessedQuery 序列化。"""
    print("\n" + "=" * 60)
    print("测试6: ProcessedQuery 序列化")
    print("=" * 60)

    pq = proc.process("BGE-M3 模型配置", filters={"collection": "tech"})

    # to_dict
    d = pq.to_dict()
    assert d["original"] == "BGE-M3 模型配置"
    assert isinstance(d["keywords"], list)
    assert d["filters"] == {"collection": "tech"}
    print(f"  to_dict: {d}")

    # from_dict
    pq2 = ProcessedQuery.from_dict(d)
    assert pq2.original == pq.original
    assert pq2.keywords == pq.keywords
    assert pq2.filters == pq.filters
    print(f"  from_dict: 还原一致 ✅")

    print("✅ 序列化测试通过")


def test_edge_cases(proc: QueryProcessor):
    """测试7: 边界场景。"""
    print("\n" + "=" * 60)
    print("测试7: 边界场景")
    print("=" * 60)

    # 空查询
    assert proc.extract_keywords("") == [], "空字符串应返回空"
    assert proc.extract_keywords("   ") == [], "纯空格应返回空"
    print("  空查询 → [] ✅")

    # 超长查询
    long_query = "test " * 1000
    keywords = proc.extract_keywords(long_query)
    assert isinstance(keywords, list), "超长查询应返回列表"
    print(f"  超长查询 → {len(keywords)} 个关键词 ✅")

    # 特殊字符
    keywords = proc.extract_keywords("@#$%^&*()_+{}|:<>?")
    assert isinstance(keywords, list), "特殊字符应返回列表"
    print(f"  特殊字符 → {len(keywords)} 个关键词 ✅")

    # 纯数字
    keywords = proc.extract_keywords("12345 67890")
    assert "12345" in keywords, "数字应被提取"
    assert "67890" in keywords, "数字应被提取"
    print(f"  纯数字 → {keywords} ✅")

    print("✅ 边界场景测试通过")


def main():
    """运行所有 D1 测试。"""
    print("=" * 60)
    print("D1 QueryProcessor 完整测试")
    print("=" * 60)

    # 加载配置
    settings = load_settings()

    # 创建 QueryProcessor（使用默认配置）
    proc = QueryProcessor(settings)
    print(f"分词器: jieba (如果可用) / regex fallback")
    print(f"最小关键词长度: {proc.min_keyword_length}")

    # 运行所有测试
    test_keyword_extraction(proc)
    test_stopwords_filtering(proc)
    test_filters_parsing(proc)
    test_process_integration(proc)
    test_trace_recording(proc)
    test_serialization(proc)
    test_edge_cases(proc)

    # 汇总
    print("\n" + "=" * 60)
    print("✅ 所有 D1 测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
