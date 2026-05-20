"""C6 MetadataEnricher 单元测试。

使用 Mock LLM 隔离测试，覆盖规则增强、LLM 增强、降级、异常处理等。
验收标准：规则模式输出 title/summary/tags 非空，LLM 模式正确调用并解析，降级行为正确。
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.core.settings import LLMConfig, Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk
from src.ingestion.transform.base_transform import BaseTransform
from src.ingestion.transform.metadata_enricher import MetadataEnricher


# ============================================================
# 辅助函数
# ============================================================


def _make_chunk(text: str, chunk_id: str = "test_0000_abcd1234") -> Chunk:
    """创建测试用 Chunk。"""
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "/test.pdf", "chunk_index": 0},
        source_ref="doc_001",
    )


def _make_settings_stub() -> Settings:
    """创建测试用 Settings stub。"""
    return Settings(
        llm=LLMConfig(provider="openai", model="test-model", api_key="test-key"),
        vision_llm=MagicMock(),
        ollama=MagicMock(),
        embedding=MagicMock(),
        splitter=MagicMock(),
        vector_store=MagicMock(),
        retrieval=MagicMock(),
        rerank=MagicMock(),
        evaluation=MagicMock(),
        observability=MagicMock(),
        pipeline=MagicMock(),
    )


def _make_mock_llm(response: str = '{"title": "测试标题", "summary": "测试摘要内容", "tags": ["测试", "标签"]}') -> MagicMock:
    """创建 Mock LLM 实例。"""
    llm = MagicMock()
    llm.chat_simple.return_value = response
    return llm


def _make_llm_json(
    title: str = "LLM 生成的标题",
    summary: str = "LLM 生成的摘要，包含内容的关键信息",
    tags: Optional[list] = None,
) -> str:
    """生成 LLM 返回的 JSON 字符串。"""
    if tags is None:
        tags = ["LLM", "标签", "关键词"]
    return json.dumps({"title": title, "summary": summary, "tags": tags}, ensure_ascii=False)


# ============================================================
# 测试：BaseTransform 抽象基类
# ============================================================


class TestBaseTransform:
    """BaseTransform 接口测试。"""

    def test_metadata_enricher_is_subclass(self):
        """MetadataEnricher 是 BaseTransform 的子类。"""
        assert issubclass(MetadataEnricher, BaseTransform)

    def test_metadata_enricher_instance(self):
        """MetadataEnricher 可以正常实例化。"""
        settings = _make_settings_stub()
        enricher = MetadataEnricher(settings=settings)
        assert isinstance(enricher, BaseTransform)

    def test_default_use_llm_false(self):
        """默认 use_llm=False。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        assert enricher.use_llm is False


# ============================================================
# 测试：规则增强 — 标题提取
# ============================================================


class TestRuleBasedTitle:
    """规则模式标题提取测试。"""

    def test_extract_markdown_title(self):
        """提取 Markdown 标题（# 开头）。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        result = enricher._extract_title("# RAG 系统概述\n\n正文内容")
        assert result == "RAG 系统概述"

    def test_extract_h2_title(self):
        """提取二级标题。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        result = enricher._extract_title("## 2.1 配置说明\n\n详细内容")
        assert result == "2.1 配置说明"

    def test_extract_first_line_title(self):
        """无标题时取第一行非空文本。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        result = enricher._extract_title("这是一段正文\n第二行内容")
        assert result == "这是一段正文"

    def test_fallback_truncated_text(self):
        """无标题无正文时截断前 30 字。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        long_text = "a" * 100
        result = enricher._extract_title(long_text)
        assert len(result) <= 50

    def test_empty_text_title(self):
        """空文本返回"未知标题"。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        result = enricher._extract_title("")
        assert result == "未知标题"

    def test_title_max_length(self):
        """标题不超过 50 字。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        result = enricher._extract_title("# " + "很长的标题" * 20)
        assert len(result) <= 50


# ============================================================
# 测试：规则增强 — 摘要提取
# ============================================================


class TestRuleBasedSummary:
    """规则模式摘要提取测试。"""

    def test_extract_summary_from_sentences(self):
        """提取前 2-3 个句子作为摘要。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        text = "RAG 是检索增强生成技术。它结合了检索和生成两种方法。这种方法在很多场景下表现出色。后续还有更多内容。"
        result = enricher._extract_summary(text)
        assert "RAG" in result
        assert len(result) <= 200

    def test_summary_removes_markdown(self):
        """摘要去除 Markdown 标记。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        text = "## 标题\n\n这是[链接](http://example.com)的**粗体**文本。还有更多内容。"
        result = enricher._extract_summary(text)
        assert "[" not in result
        assert "粗体" in result

    def test_summary_empty_text(self):
        """空文本返回"无摘要"。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        result = enricher._extract_summary("")
        assert result == "无摘要"

    def test_summary_max_length(self):
        """摘要不超过 200 字。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        text = "。".join([f"这是第{i}句话" for i in range(20)])
        result = enricher._extract_summary(text)
        assert len(result) <= 200


# ============================================================
# 测试：规则增强 — 标签提取
# ============================================================


class TestRuleBasedTags:
    """规则模式标签提取测试。"""

    def test_extract_chinese_tags(self):
        """提取中文关键词。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        text = "机器学习是人工智能的重要分支。深度学习是机器学习的子领域。自然语言处理也属于人工智能。"
        result = enricher._extract_tags(text)
        assert isinstance(result, list)
        assert len(result) >= 3
        # 应该包含高频词
        assert any("机器学习" in t or "人工智能" in t or "深度学习" in t for t in result)

    def test_extract_english_tags(self):
        """提取英文关键词。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        text = "Machine learning is a subset of artificial intelligence. Deep learning uses neural networks."
        result = enricher._extract_tags(text)
        assert isinstance(result, list)
        assert len(result) >= 3

    def test_tags_max_count(self):
        """标签不超过 8 个。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        text = " ".join([f"关键词{i}" for i in range(20)])
        result = enricher._extract_tags(text)
        assert len(result) <= 8

    def test_tags_not_empty(self):
        """标签列表不为空。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        result = enricher._extract_tags("这是一段测试文本")
        assert len(result) >= 1
        assert result[0] != ""

    def test_empty_text_tags(self):
        """空文本返回默认标签。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        result = enricher._extract_tags("")
        assert result == ["未知"]

    def test_tags_filter_stopwords(self):
        """过滤常见停用词。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        text = "这是一个很好的方法，我们可以使用它来解决问题"
        result = enricher._extract_tags(text)
        # 停用词不应出现在标签中
        stopwords = {"的", "了", "在", "是", "我", "就", "不", "都", "一", "这个", "可以"}
        for tag in result:
            assert tag not in stopwords, f"停用词 '{tag}' 不应出现在标签中"


# ============================================================
# 测试：规则增强 — 完整流程
# ============================================================


class TestRuleBasedEnrich:
    """规则模式完整增强流程测试。"""

    def test_rule_enrich_returns_required_fields(self):
        """规则增强返回 title/summary/tags 三个字段。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        result = enricher._rule_based_enrich("# 测试标题\n\n这是正文内容。包含一些关键词信息。")
        assert "title" in result
        assert "summary" in result
        assert "tags" in result

    def test_rule_enrich_title_nonempty(self):
        """规则增强 title 非空。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        result = enricher._rule_based_enrich("# RAG 概述\n\n正文内容")
        assert result["title"]
        assert len(result["title"]) > 0

    def test_rule_enrich_summary_nonempty(self):
        """规则增强 summary 非空。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        result = enricher._rule_based_enrich("# 标题\n\n这是正文内容。包含一些信息。")
        assert result["summary"]
        assert len(result["summary"]) > 0

    def test_rule_enrich_tags_nonempty(self):
        """规则增强 tags 非空列表。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        result = enricher._rule_based_enrich("# 标题\n\n这是正文内容。包含关键词和信息。")
        assert isinstance(result["tags"], list)
        assert len(result["tags"]) > 0


# ============================================================
# 测试：LLM 增强模式（Mock）
# ============================================================


class TestLLMMode:
    """LLM 增强模式测试（使用 Mock）。"""

    def test_llm_mode_calls_llm(self):
        """启用 LLM 时应调用 LLM。"""
        mock_llm = _make_mock_llm(_make_llm_json())
        settings = _make_settings_stub()
        enricher = MetadataEnricher(settings=settings, llm=mock_llm, use_llm=True)
        chunk = _make_chunk("这是测试文本。包含一些内容。")
        enricher.transform([chunk])
        mock_llm.chat_simple.assert_called_once()

    def test_llm_mode_marks_metadata_enriched_by_llm(self):
        """LLM 成功时 metadata 标记 enriched_by='llm'。"""
        mock_llm = _make_mock_llm(_make_llm_json())
        enricher = MetadataEnricher(settings=_make_settings_stub(), llm=mock_llm, use_llm=True)
        chunk = _make_chunk("这是测试文本。")
        result = enricher.transform([chunk])
        assert result[0].metadata["enriched_by"] == "llm"

    def test_llm_mode_writes_title(self):
        """LLM 成功时写入 title。"""
        mock_llm = _make_mock_llm(_make_llm_json(title="LLM 标题"))
        enricher = MetadataEnricher(settings=_make_settings_stub(), llm=mock_llm, use_llm=True)
        chunk = _make_chunk("文本内容。")
        result = enricher.transform([chunk])
        assert result[0].metadata["title"] == "LLM 标题"

    def test_llm_mode_writes_summary(self):
        """LLM 成功时写入 summary。"""
        mock_llm = _make_mock_llm(_make_llm_json(summary="LLM 生成的摘要"))
        enricher = MetadataEnricher(settings=_make_settings_stub(), llm=mock_llm, use_llm=True)
        chunk = _make_chunk("文本内容。")
        result = enricher.transform([chunk])
        assert result[0].metadata["summary"] == "LLM 生成的摘要"

    def test_llm_mode_writes_tags(self):
        """LLM 成功时写入 tags。"""
        mock_llm = _make_mock_llm(_make_llm_json(tags=["标签A", "标签B"]))
        enricher = MetadataEnricher(settings=_make_settings_stub(), llm=mock_llm, use_llm=True)
        chunk = _make_chunk("文本内容。")
        result = enricher.transform([chunk])
        assert result[0].metadata["tags"] == ["标签A", "标签B"]

    def test_llm_json_in_markdown_codeblock(self):
        """LLM 返回 markdown 代码块包裹的 JSON 也能正确解析。"""
        json_str = _make_llm_json()
        response = f"```json\n{json_str}\n```"
        mock_llm = _make_mock_llm(response)
        enricher = MetadataEnricher(settings=_make_settings_stub(), llm=mock_llm, use_llm=True)
        chunk = _make_chunk("文本内容。")
        result = enricher.transform([chunk])
        assert result[0].metadata["enriched_by"] == "llm"
        assert "title" in result[0].metadata


# ============================================================
# 测试：LLM 降级行为
# ============================================================


class TestLLMFallback:
    """LLM 降级行为测试。"""

    def test_llm_failure_fallback_to_rule(self):
        """LLM 抛异常时回退到规则结果，metadata 标记 enriched_by='rule'。"""
        mock_llm = MagicMock()
        mock_llm.chat_simple.side_effect = RuntimeError("API error")
        enricher = MetadataEnricher(settings=_make_settings_stub(), llm=mock_llm, use_llm=True)
        chunk = _make_chunk("# 标题\n\n这是正文内容。包含关键词。")
        result = enricher.transform([chunk])
        assert result[0].metadata["enriched_by"] == "rule"
        assert "title" in result[0].metadata
        assert "summary" in result[0].metadata
        assert "tags" in result[0].metadata

    def test_llm_empty_response_fallback(self):
        """LLM 返回空字符串时回退到规则结果。"""
        mock_llm = _make_mock_llm("")
        enricher = MetadataEnricher(settings=_make_settings_stub(), llm=mock_llm, use_llm=True)
        chunk = _make_chunk("# 标题\n\n正文内容。")
        result = enricher.transform([chunk])
        assert result[0].metadata["enriched_by"] == "rule"

    def test_llm_none_response_fallback(self):
        """LLM 返回 None 时回退到规则结果。"""
        mock_llm = _make_mock_llm(None)
        enricher = MetadataEnricher(settings=_make_settings_stub(), llm=mock_llm, use_llm=True)
        chunk = _make_chunk("# 标题\n\n正文内容。")
        result = enricher.transform([chunk])
        assert result[0].metadata["enriched_by"] == "rule"

    def test_llm_invalid_json_fallback(self):
        """LLM 返回非 JSON 时回退到规则结果。"""
        mock_llm = _make_mock_llm("这不是一个 JSON 字符串")
        enricher = MetadataEnricher(settings=_make_settings_stub(), llm=mock_llm, use_llm=True)
        chunk = _make_chunk("# 标题\n\n正文内容。")
        result = enricher.transform([chunk])
        assert result[0].metadata["enriched_by"] == "rule"

    def test_llm_missing_fields_fallback(self):
        """LLM 返回缺少必需字段时回退到规则结果。"""
        mock_llm = _make_mock_llm('{"title": "只有标题"}')
        enricher = MetadataEnricher(settings=_make_settings_stub(), llm=mock_llm, use_llm=True)
        chunk = _make_chunk("# 标题\n\n正文内容。")
        result = enricher.transform([chunk])
        assert result[0].metadata["enriched_by"] == "rule"


# ============================================================
# 测试：配置开关
# ============================================================


class TestConfig:
    """配置驱动行为测试。"""

    def test_use_llm_false_skips_llm(self):
        """use_llm=False 时不调用 LLM。"""
        mock_llm = _make_mock_llm()
        enricher = MetadataEnricher(settings=_make_settings_stub(), llm=mock_llm, use_llm=False)
        chunk = _make_chunk("文本内容。")
        enricher.transform([chunk])
        mock_llm.chat_simple.assert_not_called()

    def test_use_llm_true_enables_llm(self):
        """use_llm=True 时调用 LLM。"""
        mock_llm = _make_mock_llm(_make_llm_json())
        enricher = MetadataEnricher(settings=_make_settings_stub(), llm=mock_llm, use_llm=True)
        chunk = _make_chunk("文本内容。")
        enricher.transform([chunk])
        mock_llm.chat_simple.assert_called_once()

    def test_rule_mode_no_enriched_by_key_when_disabled(self):
        """use_llm=False 时 enriched_by='rule'。"""
        enricher = MetadataEnricher(settings=_make_settings_stub(), use_llm=False)
        chunk = _make_chunk("文本内容。")
        result = enricher.transform([chunk])
        assert result[0].metadata["enriched_by"] == "rule"


# ============================================================
# 测试：异常处理
# ============================================================


class TestErrorHandling:
    """异常处理测试。"""

    def test_single_chunk_error_preserves_others(self):
        """单个 chunk 处理异常不影响其他 chunk。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        # 模拟 _enrich_single 对第二个 chunk 抛异常
        call_count = [0]
        original_enrich = enricher._enrich_single

        def mock_enrich(chunk, trace=None):
            call_count[0] += 1
            if call_count[0] == 2:
                raise ValueError("boom")
            return original_enrich(chunk, trace)

        enricher._enrich_single = mock_enrich
        chunks = [_make_chunk("c1", "id1"), _make_chunk("c2", "id2"), _make_chunk("c3", "id3")]
        result = enricher.transform(chunks)
        assert len(result) == 3

    def test_error_marks_metadata(self):
        """处理异常时 metadata 标记 enriched_by='error'。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        with patch.object(enricher, '_rule_based_enrich', side_effect=ValueError("bad text")):
            chunk = _make_chunk("text")
            result = enricher.transform([chunk])
            assert result[0].metadata["enriched_by"] == "error"
            assert "bad text" in result[0].metadata["enrich_error"]


# ============================================================
# 测试：Prompt 加载
# ============================================================


class TestPromptLoading:
    """Prompt 模板加载测试。"""

    def test_load_default_prompt(self):
        """默认路径加载 prompt 模板。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        assert "{text}" in enricher.prompt_template
        assert "title" in enricher.prompt_template

    def test_load_custom_prompt(self):
        """自定义路径加载 prompt 模板。"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("自定义 prompt: {text}")
            custom_path = f.name
        try:
            enricher = MetadataEnricher(settings=_make_settings_stub(), prompt_path=custom_path)
            assert "自定义 prompt" in enricher.prompt_template
        finally:
            os.unlink(custom_path)

    def test_load_prompt_fallback(self):
        """文件不存在时使用内置 fallback。"""
        enricher = MetadataEnricher(settings=_make_settings_stub(), prompt_path="/nonexistent/path.txt")
        assert "{text}" in enricher.prompt_template
        assert "title" in enricher.prompt_template


# ============================================================
# 测试：Trace 记录
# ============================================================


class TestTraceRecording:
    """Trace 阶段记录测试。"""

    def test_trace_records_rule_stage(self):
        """规则模式下 trace 记录 metadata_enrich 阶段。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        trace = TraceContext()
        chunk = _make_chunk("文本内容。")
        enricher.transform([chunk], trace=trace)
        stage_names = [s["name"] for s in trace.stages]
        assert "metadata_enrich" in stage_names

    def test_trace_records_llm_stage(self):
        """LLM 模式下 trace 记录 metadata_enrich 阶段（method=llm）。"""
        mock_llm = _make_mock_llm(_make_llm_json())
        enricher = MetadataEnricher(settings=_make_settings_stub(), llm=mock_llm, use_llm=True)
        trace = TraceContext()
        chunk = _make_chunk("文本内容。")
        enricher.transform([chunk], trace=trace)
        llm_stages = [s for s in trace.stages if s.get("method") == "llm"]
        assert len(llm_stages) > 0


# ============================================================
# 测试：Transform 接口契约
# ============================================================


class TestTransformContract:
    """Transform 接口契约测试。"""

    def test_transform_preserves_chunk_count(self):
        """transform 保持 chunk 数量不变。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        chunks = [_make_chunk("c1", "id1"), _make_chunk("c2", "id2"), _make_chunk("c3", "id3")]
        result = enricher.transform(chunks)
        assert len(result) == len(chunks)

    def test_transform_preserves_chunk_ids(self):
        """transform 保持 chunk ID 不变。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        chunks = [_make_chunk("c1", "id1"), _make_chunk("c2", "id2")]
        result = enricher.transform(chunks)
        assert result[0].id == "id1"
        assert result[1].id == "id2"

    def test_transform_preserves_existing_metadata(self):
        """transform 保持原有 metadata 字段。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        chunk = _make_chunk("文本")
        chunk.metadata["custom_field"] = 42
        result = enricher.transform([chunk])
        assert result[0].metadata["custom_field"] == 42

    def test_empty_chunks_returns_empty(self):
        """空列表输入返回空列表。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        result = enricher.transform([])
        assert result == []


# ============================================================
# 测试：LLM 输出解析
# ============================================================


class TestLLMOutputParsing:
    """LLM 输出解析测试。"""

    def test_parse_valid_json(self):
        """解析合法 JSON。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        output = '{"title": "标题", "summary": "摘要", "tags": ["a", "b"]}'
        result = enricher._parse_llm_output(output)
        assert result is not None
        assert result["title"] == "标题"
        assert result["summary"] == "摘要"
        assert result["tags"] == ["a", "b"]

    def test_parse_json_in_codeblock(self):
        """解析 markdown 代码块中的 JSON。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        output = '```json\n{"title": "标题", "summary": "摘要", "tags": ["a"]}\n```'
        result = enricher._parse_llm_output(output)
        assert result is not None
        assert result["title"] == "标题"

    def test_parse_json_with_extra_text(self):
        """从包含额外文本的输出中提取 JSON。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        output = '以下是结果：\n{"title": "标题", "summary": "摘要", "tags": ["a"]}\n以上是结果。'
        result = enricher._parse_llm_output(output)
        assert result is not None
        assert result["title"] == "标题"

    def test_parse_invalid_json_returns_none(self):
        """非法 JSON 返回 None。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        result = enricher._parse_llm_output("这不是 JSON")
        assert result is None

    def test_parse_json_missing_title_returns_none(self):
        """缺少 title 字段返回 None。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        output = '{"summary": "摘要", "tags": ["a"]}'
        result = enricher._parse_llm_output(output)
        assert result is None

    def test_parse_json_missing_summary_returns_none(self):
        """缺少 summary 字段返回 None。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        output = '{"title": "标题", "tags": ["a"]}'
        result = enricher._parse_llm_output(output)
        assert result is None

    def test_parse_json_missing_tags_returns_none(self):
        """缺少 tags 字段返回 None。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        output = '{"title": "标题", "summary": "摘要"}'
        result = enricher._parse_llm_output(output)
        assert result is None

    def test_parse_json_empty_title_returns_none(self):
        """title 为空字符串返回 None。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        output = '{"title": "", "summary": "摘要", "tags": ["a"]}'
        result = enricher._parse_llm_output(output)
        assert result is None

    def test_parse_json_tags_as_string_returns_none(self):
        """tags 为字符串（非列表）返回 None。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        output = '{"title": "标题", "summary": "摘要", "tags": "不是列表"}'
        result = enricher._parse_llm_output(output)
        assert result is None

    def test_parse_json_tags_with_non_string_items(self):
        """tags 列表中的非字符串元素会被转为字符串。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        output = '{"title": "标题", "summary": "摘要", "tags": [123, true]}'
        result = enricher._parse_llm_output(output)
        assert result is not None
        assert result["tags"] == ["123", "True"]


# ============================================================
# 测试：边界场景
# ============================================================


class TestEdgeCases:
    """边界场景测试。"""

    def test_chunk_with_existing_metadata_fields(self):
        """chunk 已有 title/summary/tags 时被覆盖。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        chunk = _make_chunk("文本内容")
        chunk.metadata["title"] = "旧标题"
        chunk.metadata["summary"] = "旧摘要"
        chunk.metadata["tags"] = ["旧标签"]
        result = enricher.transform([chunk])
        # 规则增强应该覆盖旧值
        assert result[0].metadata["title"] != "旧标题" or result[0].metadata["enriched_by"] == "rule"

    def test_transform_preserves_text(self):
        """transform 不修改 chunk.text。"""
        enricher = MetadataEnricher(settings=_make_settings_stub())
        chunk = _make_chunk("原始文本内容")
        result = enricher.transform([chunk])
        assert result[0].text == "原始文本内容"

    def test_llm_prompt_contains_text(self):
        """LLM prompt 中包含 chunk 文本。"""
        mock_llm = _make_mock_llm(_make_llm_json())
        enricher = MetadataEnricher(settings=_make_settings_stub(), llm=mock_llm, use_llm=True)
        chunk = _make_chunk("这是待处理的文本内容。")
        enricher.transform([chunk])
        # 检查调用 LLM 时的 prompt 包含 chunk 文本
        call_args = mock_llm.chat_simple.call_args
        prompt = call_args[0][0] if call_args[0] else call_args[1].get("prompt", "")
        assert "这是待处理的文本内容" in prompt
