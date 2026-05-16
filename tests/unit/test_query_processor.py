"""D1 QueryProcessor 单元测试。

覆盖关键词提取、停用词过滤、filters 解析、空输入处理、
自定义分词器注入、Trace 记录、ProcessedQuery 序列化等。
验收标准：对输入 query 输出 keywords 非空，filters 为 dict。
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import pytest

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import ProcessedQuery
from src.core.query_engine.query_processor import QueryProcessor


# ============================================================
# 测试辅助函数
# ============================================================


def _make_settings_stub(**kwargs) -> Settings:
    """创建测试用 Settings stub。"""
    return Settings(
        llm=MagicMock(),
        vision_llm=MagicMock(),
        ollama=MagicMock(),
        embedding=MagicMock(),
        splitter=MagicMock(),
        vector_store=MagicMock(),
        retrieval=MagicMock(),
        rerank=MagicMock(),
        evaluation=MagicMock(),
        observability=MagicMock(),
    )


def _make_processor(**kwargs) -> QueryProcessor:
    """创建测试用 QueryProcessor 实例。"""
    settings = _make_settings_stub()
    return QueryProcessor(settings, **kwargs)


# ============================================================
# 关键词提取测试
# ============================================================


class TestExtractKeywords:
    """关键词提取功能测试。"""

    def test_basic_english_query(self):
        """英文查询能提取关键词。"""
        proc = _make_processor()
        keywords = proc.extract_keywords("How to configure Ollama?")
        assert len(keywords) > 0
        assert "configure" in keywords
        assert "ollama" in keywords

    def test_basic_chinese_query(self):
        """中文查询能提取关键词。"""
        proc = _make_processor()
        keywords = proc.extract_keywords("如何配置 Ollama 模型？")
        assert len(keywords) > 0
        # 至少应包含 "配置" 和 "ollama"
        assert any("ollama" in kw for kw in keywords)

    def test_mixed_language_query(self):
        """中英混合查询能提取关键词。"""
        proc = _make_processor()
        keywords = proc.extract_keywords("BGE-M3 模型的 embedding 维度是多少？")
        assert len(keywords) > 0
        assert "bge-m3" in keywords or "bge" in keywords
        assert "embedding" in keywords

    def test_empty_query_returns_empty(self):
        """空查询返回空列表。"""
        proc = _make_processor()
        assert proc.extract_keywords("") == []
        assert proc.extract_keywords("   ") == []
        assert proc.extract_keywords(None) == []

    def test_stopwords_filtered(self):
        """停用词被正确过滤。"""
        proc = _make_processor()
        keywords = proc.extract_keywords("What is the best way to learn?")
        # "what", "is", "the", "to" 都是停用词
        assert "what" not in keywords
        assert "is" not in keywords
        assert "the" not in keywords
        assert "to" not in keywords
        # "best", "way", "learn" 应保留
        assert "best" in keywords
        assert "learn" in keywords

    def test_chinese_stopwords_filtered(self):
        """中文停用词被正确过滤。"""
        proc = _make_processor(min_keyword_length=1)
        keywords = proc.extract_keywords("这是一个很好的测试方法")
        # "这", "是", "的" 是停用词，应被过滤
        assert "这" not in keywords
        assert "是" not in keywords
        assert "的" not in keywords
        # "很", "好", "测", "试", "方", "法" 等非停用词应保留
        assert len(keywords) > 0

    def test_keywords_deduplicated(self):
        """重复关键词被去重。"""
        proc = _make_processor()
        keywords = proc.extract_keywords("test test test hello test")
        # "test" 只应出现一次
        assert keywords.count("test") == 1
        assert "hello" in keywords

    def test_keywords_order_preserved(self):
        """去重后保持首次出现顺序。"""
        proc = _make_processor()
        keywords = proc.extract_keywords("alpha beta gamma alpha beta")
        assert keywords == ["alpha", "beta", "gamma"]

    def test_min_keyword_length(self):
        """短于 min_keyword_length 的关键词被过滤。"""
        proc = _make_processor(min_keyword_length=4)
        keywords = proc.extract_keywords("AI is the best tool for ML tasks")
        # "ai" (2), "is" (2), "ml" (2) 应被过滤
        assert "ai" not in keywords
        assert "ml" not in keywords
        # "best" (4), "tool" (4), "tasks" (5) 应保留
        assert "best" in keywords
        assert "tool" in keywords

    def test_custom_tokenizer(self):
        """自定义分词器可正确注入。"""
        def custom_tokenizer(text: str) -> List[str]:
            return text.split()

        proc = _make_processor(tokenizer=custom_tokenizer)
        keywords = proc.extract_keywords("hello world foo bar")
        assert keywords == ["hello", "world", "foo", "bar"]

    def test_custom_stopwords(self):
        """自定义停用词集合可正确注入。"""
        custom_stopwords = {"foo", "bar"}
        proc = _make_processor(stopwords=custom_stopwords)
        keywords = proc.extract_keywords("foo bar baz qux")
        assert "foo" not in keywords
        assert "bar" not in keywords
        assert "baz" in keywords
        assert "qux" in keywords

    def test_only_stopwords_query(self):
        """全停用词查询返回空列表。"""
        proc = _make_processor()
        keywords = proc.extract_keywords("的是在了")
        # 这些都是停用词，应被过滤为空列表
        assert isinstance(keywords, list)
        assert len(keywords) == 0


# ============================================================
# Filters 解析测试
# ============================================================


class TestParseFilters:
    """过滤条件解析测试。"""

    def test_none_filters_returns_empty(self):
        """None filters 返回空 dict。"""
        proc = _make_processor()
        assert proc.parse_filters(None) == {}

    def test_empty_filters_returns_empty(self):
        """空 dict filters 返回空 dict。"""
        proc = _make_processor()
        assert proc.parse_filters({}) == {}

    def test_valid_filters_preserved(self):
        """有效过滤条件被保留。"""
        proc = _make_processor()
        filters = {"collection": "test", "doc_type": "pdf"}
        result = proc.parse_filters(filters)
        assert result == {"collection": "test", "doc_type": "pdf"}

    def test_none_values_removed(self):
        """None 值被移除。"""
        proc = _make_processor()
        filters = {"collection": "test", "doc_type": None}
        result = proc.parse_filters(filters)
        assert result == {"collection": "test"}

    def test_empty_string_values_removed(self):
        """空字符串值被移除。"""
        proc = _make_processor()
        filters = {"collection": "test", "doc_type": ""}
        result = proc.parse_filters(filters)
        assert result == {"collection": "test"}

    def test_mixed_valid_invalid_filters(self):
        """混合有效和无效过滤条件。"""
        proc = _make_processor()
        filters = {
            "collection": "test",
            "doc_type": None,
            "source": "",
            "page": 5,
        }
        result = proc.parse_filters(filters)
        assert result == {"collection": "test", "page": 5}


# ============================================================
# process() 集成测试
# ============================================================


class TestProcess:
    """process() 方法集成测试。"""

    def test_process_returns_processed_query(self):
        """process 返回 ProcessedQuery 对象。"""
        proc = _make_processor()
        result = proc.process("How to use BGE-M3?")
        assert isinstance(result, ProcessedQuery)
        assert result.original == "How to use BGE-M3?"
        assert len(result.keywords) > 0
        assert isinstance(result.filters, dict)

    def test_process_with_filters(self):
        """process 正确传递 filters。"""
        proc = _make_processor()
        filters = {"collection": "docs", "doc_type": "pdf"}
        result = proc.process("test query", filters=filters)
        assert result.filters == {"collection": "docs", "doc_type": "pdf"}

    def test_process_without_filters(self):
        """不传 filters 时返回空 dict。"""
        proc = _make_processor()
        result = proc.process("test query")
        assert result.filters == {}

    def test_process_keywords_non_empty(self):
        """对非空查询，keywords 始终非空。"""
        proc = _make_processor()
        queries = [
            "Hello World",
            "如何配置 Ollama？",
            "BGE-M3 embedding model",
            "Python 测试",
        ]
        for query in queries:
            result = proc.process(query)
            assert len(result.keywords) > 0, f"query '{query}' keywords 为空"


# ============================================================
# Trace 记录测试
# ============================================================


class TestTraceRecording:
    """Trace 记录测试。"""

    def test_trace_records_query_processing(self):
        """传入 trace 时记录 query_processing 阶段。"""
        proc = _make_processor()
        trace = TraceContext()
        proc.extract_keywords("test query", trace=trace)

        stages = [s for s in trace.stages if s["name"] == "query_processing"]
        assert len(stages) == 1
        assert "method" in stages[0]
        assert "keyword_count" in stages[0]

    def test_no_trace_no_error(self):
        """不传 trace 时不报错。"""
        proc = _make_processor()
        keywords = proc.extract_keywords("test query")
        assert len(keywords) > 0


# ============================================================
# ProcessedQuery 序列化测试
# ============================================================


class TestProcessedQuerySerialization:
    """ProcessedQuery 序列化/反序列化测试。"""

    def test_to_dict(self):
        """to_dict 输出正确的字典结构。"""
        proc = _make_processor()
        pq = proc.process("test query", filters={"collection": "docs"})
        d = pq.to_dict()
        assert d["original"] == "test query"
        assert isinstance(d["keywords"], list)
        assert d["filters"] == {"collection": "docs"}

    def test_from_dict(self):
        """from_dict 能正确反序列化。"""
        data = {
            "original": "test query",
            "keywords": ["test", "query"],
            "filters": {"collection": "docs"},
        }
        pq = ProcessedQuery.from_dict(data)
        assert pq.original == "test query"
        assert pq.keywords == ["test", "query"]
        assert pq.filters == {"collection": "docs"}

    def test_roundtrip(self):
        """序列化 → 反序列化 roundtrip 保持一致。"""
        proc = _make_processor()
        pq = proc.process("BGE-M3 模型配置", filters={"collection": "tech"})
        d = pq.to_dict()
        pq2 = ProcessedQuery.from_dict(d)
        assert pq2.original == pq.original
        assert pq2.keywords == pq.keywords
        assert pq2.filters == pq.filters


# ============================================================
# 边界场景测试
# ============================================================


class TestEdgeCases:
    """边界场景测试。"""

    def test_very_long_query(self):
        """超长查询不崩溃。"""
        proc = _make_processor()
        long_query = "test " * 1000
        keywords = proc.extract_keywords(long_query)
        assert isinstance(keywords, list)

    def test_special_characters(self):
        """特殊字符查询不崩溃。"""
        proc = _make_processor()
        keywords = proc.extract_keywords("@#$%^&*()_+{}|:<>?")
        assert isinstance(keywords, list)

    def test_only_numbers(self):
        """纯数字查询。"""
        proc = _make_processor()
        keywords = proc.extract_keywords("12345 67890")
        # 数字应被提取（长度 >= min_keyword_length）
        assert "12345" in keywords
        assert "67890" in keywords

    def test_whitespace_only(self):
        """纯空白查询返回空列表。"""
        proc = _make_processor()
        assert proc.extract_keywords("   \t\n  ") == []
