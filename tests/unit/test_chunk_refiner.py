"""C5 ChunkRefiner 单元测试。

使用 Mock LLM 隔离测试，覆盖规则去噪、LLM 增强、降级、异常处理等。
验收标准：27 项测试全部通过。
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.core.settings import LLMConfig, Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk
from src.ingestion.transform.base_transform import BaseTransform
from src.ingestion.transform.chunk_refiner import ChunkRefiner

# ============================================================
# 测试 fixtures 路径
# ============================================================

_FIXTURES_PATH = os.path.join("tests", "fixtures", "noisy_chunks.json")


def _load_fixture(name: str) -> dict:
    """加载 noisy_chunks.json 中指定场景的 fixture。"""
    with open(_FIXTURES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[name]


def _make_chunk(text: str, chunk_id: str = "test_0000_abcd1234") -> Chunk:
    """创建测试用 Chunk。"""
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "/test.pdf", "chunk_index": 0},
        source_ref="doc_001",
    )


def _make_settings_stub(use_llm: bool = False) -> Settings:
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


def _make_mock_llm(response: str = "cleaned text") -> MagicMock:
    """创建 Mock LLM 实例。"""
    llm = MagicMock()
    llm.chat_simple.return_value = response
    return llm


# ============================================================
# 测试：BaseTransform 抽象基类
# ============================================================

class TestBaseTransform:
    """BaseTransform 接口测试。"""

    def test_base_transform_is_abstract(self):
        """BaseTransform 不能直接实例化。"""
        with pytest.raises(TypeError):
            BaseTransform()

    def test_chunk_refiner_is_subclass(self):
        """ChunkRefiner 是 BaseTransform 的子类。"""
        assert issubclass(ChunkRefiner, BaseTransform)

    def test_chunk_refiner_instance(self):
        """ChunkRefiner 可以正常实例化。"""
        settings = _make_settings_stub()
        refiner = ChunkRefiner(settings=settings)
        assert isinstance(refiner, BaseTransform)


# ============================================================
# 测试：规则去噪
# ============================================================

class TestRuleBasedRefine:
    """规则模式去噪测试。"""

    def test_removes_page_header_footer(self):
        """去除页眉页脚模式（"标题 - 第N页"）。"""
        fixture = _load_fixture("page_header_footer")
        refiner = ChunkRefiner(settings=_make_settings_stub())
        result = refiner._rule_based_refine(fixture["input"])
        for expected in fixture["expected_contains"]:
            assert expected in result, f"应保留: {expected}"
        for excluded in fixture["expected_not_contains"]:
            assert excluded not in result, f"应去除: {excluded}"

    def test_removes_excessive_whitespace(self):
        """去除多余空白（连续空格、空行、制表符）。"""
        fixture = _load_fixture("excessive_whitespace")
        refiner = ChunkRefiner(settings=_make_settings_stub())
        result = refiner._rule_based_refine(fixture["input"])
        for expected in fixture["expected_contains"]:
            assert expected in result
        assert "    " not in result, "连续空格应被压缩"
        assert "\n\n\n\n\n" not in result, "连续空行应被压缩"

    def test_removes_html_comments(self):
        """去除 HTML 注释。"""
        refiner = ChunkRefiner(settings=_make_settings_stub())
        result = refiner._rule_based_refine("文本<!-- 注释内容 -->结束")
        assert "<!--" not in result
        assert "注释内容" not in result
        assert "文本" in result
        assert "结束" in result

    def test_removes_format_markers(self):
        """去除 HTML 标签和 style 标签。"""
        fixture = _load_fixture("format_markers")
        refiner = ChunkRefiner(settings=_make_settings_stub())
        result = refiner._rule_based_refine(fixture["input"])
        for expected in fixture["expected_contains"]:
            assert expected in result
        for excluded in fixture["expected_not_contains"]:
            assert excluded not in result

    def test_removes_separator_lines(self):
        """去除分隔线（---、===、***）。"""
        refiner = ChunkRefiner(settings=_make_settings_stub())
        text = "前文\n\n---\n\n后文"
        result = refiner._rule_based_refine(text)
        assert "---" not in result
        assert "前文" in result
        assert "后文" in result

    def test_preserves_clean_text(self):
        """干净文本不应被过度清理。"""
        fixture = _load_fixture("clean_text")
        refiner = ChunkRefiner(settings=_make_settings_stub())
        result = refiner._rule_based_refine(fixture["input"])
        for expected in fixture["expected_contains"]:
            assert expected in result

    def test_preserves_code_blocks(self):
        """代码块内部格式必须保留。"""
        fixture = _load_fixture("code_blocks")
        refiner = ChunkRefiner(settings=_make_settings_stub())
        result = refiner._rule_based_refine(fixture["input"])
        for expected in fixture["expected_contains"]:
            assert expected in result, f"代码块内容应保留: {expected}"

    def test_handles_mixed_noise(self):
        """混合噪声场景：页眉页脚 + 注释 + 多余空白。"""
        fixture = _load_fixture("mixed_noise")
        refiner = ChunkRefiner(settings=_make_settings_stub())
        result = refiner._rule_based_refine(fixture["input"])
        for expected in fixture["expected_contains"]:
            assert expected in result
        for excluded in fixture["expected_not_contains"]:
            assert excluded not in result

    def test_typical_noise_scenario(self):
        """综合噪声场景测试。"""
        fixture = _load_fixture("typical_noise_scenario")
        refiner = ChunkRefiner(settings=_make_settings_stub())
        result = refiner._rule_based_refine(fixture["input"])
        for expected in fixture["expected_contains"]:
            assert expected in result
        for excluded in fixture["expected_not_contains"]:
            assert excluded not in result


# ============================================================
# 测试：LLM 模式（Mock）
# ============================================================

class TestLLMMode:
    """LLM 增强模式测试（使用 Mock）。"""

    def test_llm_mode_calls_llm(self):
        """启用 LLM 时应调用 LLM。"""
        mock_llm = _make_mock_llm("LLM cleaned text")
        settings = _make_settings_stub()
        refiner = ChunkRefiner(settings=settings, llm=mock_llm, use_llm=True)
        chunk = _make_chunk("原始文本。")
        refiner.transform([chunk])
        mock_llm.chat_simple.assert_called_once()

    def test_llm_mode_marks_metadata(self):
        """LLM 成功时 metadata 标记 refined_by='llm'。"""
        mock_llm = _make_mock_llm("LLM cleaned text")
        refiner = ChunkRefiner(settings=_make_settings_stub(), llm=mock_llm, use_llm=True)
        chunk = _make_chunk("原始文本。")
        result = refiner.transform([chunk])
        assert result[0].metadata["refined_by"] == "llm"

    def test_llm_failure_fallback_to_rule(self):
        """LLM 抛异常时回退到规则结果，metadata 标记 refined_by='rule'。"""
        mock_llm = MagicMock()
        mock_llm.chat_simple.side_effect = RuntimeError("API error")
        refiner = ChunkRefiner(settings=_make_settings_stub(), llm=mock_llm, use_llm=True)
        chunk = _make_chunk("正常文本。")
        result = refiner.transform([chunk])
        assert result[0].metadata["refined_by"] == "rule"

    def test_llm_empty_response_fallback(self):
        """LLM 返回空字符串时回退到规则结果。"""
        mock_llm = _make_mock_llm("")
        refiner = ChunkRefiner(settings=_make_settings_stub(), llm=mock_llm, use_llm=True)
        chunk = _make_chunk("正常文本。")
        result = refiner.transform([chunk])
        assert result[0].metadata["refined_by"] == "rule"

    def test_llm_none_response_fallback(self):
        """LLM 返回 None 时回退到规则结果。"""
        mock_llm = _make_mock_llm(None)
        refiner = ChunkRefiner(settings=_make_settings_stub(), llm=mock_llm, use_llm=True)
        chunk = _make_chunk("正常文本。")
        result = refiner.transform([chunk])
        assert result[0].metadata["refined_by"] == "rule"


# ============================================================
# 测试：配置开关
# ============================================================

class TestConfig:
    """配置驱动行为测试。"""

    def test_use_llm_false_skips_llm(self):
        """use_llm=False 时不调用 LLM。"""
        mock_llm = _make_mock_llm()
        refiner = ChunkRefiner(settings=_make_settings_stub(), llm=mock_llm, use_llm=False)
        chunk = _make_chunk("文本。")
        refiner.transform([chunk])
        mock_llm.chat_simple.assert_not_called()

    def test_use_llm_true_enables_llm(self):
        """use_llm=True 时调用 LLM。"""
        mock_llm = _make_mock_llm("cleaned")
        refiner = ChunkRefiner(settings=_make_settings_stub(), llm=mock_llm, use_llm=True)
        chunk = _make_chunk("文本。")
        refiner.transform([chunk])
        mock_llm.chat_simple.assert_called_once()

    def test_default_use_llm_false(self):
        """默认 use_llm=False。"""
        refiner = ChunkRefiner(settings=_make_settings_stub())
        assert refiner.use_llm is False


# ============================================================
# 测试：异常处理
# ============================================================

class TestErrorHandling:
    """异常处理测试。"""

    def test_single_chunk_error_preserves_others(self):
        """单个 chunk 处理异常不影响其他 chunk。"""
        mock_llm = MagicMock()
        call_count = [0]

        def side_effect(prompt):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("boom")
            return "cleaned"

        mock_llm.chat_simple.side_effect = side_effect
        refiner = ChunkRefiner(settings=_make_settings_stub(), llm=mock_llm, use_llm=True)

        chunks = [_make_chunk("chunk1", "c1"), _make_chunk("chunk2", "c2"), _make_chunk("chunk3", "c3")]
        result = refiner.transform(chunks)
        assert len(result) == 3

    def test_error_marks_metadata(self):
        """处理异常时 metadata 标记 refined_by='error'。"""
        mock_llm = MagicMock()
        mock_llm.chat_simple.side_effect = RuntimeError("API down")
        # 需要让规则去噪也抛异常才能触发顶层 except
        refiner = ChunkRefiner(settings=_make_settings_stub(), llm=mock_llm, use_llm=True)
        # 模拟 _rule_based_refine 抛异常
        with patch.object(refiner, '_rule_based_refine', side_effect=ValueError("bad text")):
            chunk = _make_chunk("text")
            result = refiner.transform([chunk])
            assert result[0].metadata["refined_by"] == "error"
            assert "bad text" in result[0].metadata["refine_error"]


# ============================================================
# 测试：Prompt 加载
# ============================================================

class TestPromptLoading:
    """Prompt 模板加载测试。"""

    def test_load_default_prompt(self):
        """默认路径加载 prompt 模板。"""
        refiner = ChunkRefiner(settings=_make_settings_stub())
        assert "{text}" in refiner.prompt_template

    def test_load_custom_prompt(self):
        """自定义路径加载 prompt 模板。"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("自定义 prompt: {text}")
            custom_path = f.name
        try:
            refiner = ChunkRefiner(settings=_make_settings_stub(), prompt_path=custom_path)
            assert "自定义 prompt" in refiner.prompt_template
        finally:
            os.unlink(custom_path)

    def test_load_prompt_fallback(self):
        """文件不存在时使用内置 fallback。"""
        refiner = ChunkRefiner(settings=_make_settings_stub(), prompt_path="/nonexistent/path.txt")
        assert "{text}" in refiner.prompt_template
        assert "清理" in refiner.prompt_template


# ============================================================
# 测试：Trace 记录
# ============================================================

class TestTraceRecording:
    """Trace 阶段记录测试。"""

    def test_trace_records_rule_stage(self):
        """规则模式下 trace 记录 chunk_refine 阶段。"""
        refiner = ChunkRefiner(settings=_make_settings_stub())
        trace = TraceContext()
        chunk = _make_chunk("文本。")
        refiner.transform([chunk], trace=trace)
        stage_names = [s["name"] for s in trace.stages]
        assert "chunk_refine" in stage_names

    def test_trace_records_llm_stage(self):
        """LLM 模式下 trace 记录 chunk_refine 阶段（method=llm）。"""
        mock_llm = _make_mock_llm("cleaned")
        refiner = ChunkRefiner(settings=_make_settings_stub(), llm=mock_llm, use_llm=True)
        trace = TraceContext()
        chunk = _make_chunk("文本。")
        refiner.transform([chunk], trace=trace)
        llm_stages = [s for s in trace.stages if s.get("method") == "llm"]
        assert len(llm_stages) > 0


# ============================================================
# 测试：Transform 接口契约
# ============================================================

class TestTransformContract:
    """Transform 接口契约测试。"""

    def test_transform_preserves_chunk_count(self):
        """transform 保持 chunk 数量不变。"""
        refiner = ChunkRefiner(settings=_make_settings_stub())
        chunks = [_make_chunk("c1", "id1"), _make_chunk("c2", "id2"), _make_chunk("c3", "id3")]
        result = refiner.transform(chunks)
        assert len(result) == len(chunks)

    def test_transform_preserves_chunk_ids(self):
        """transform 保持 chunk ID 不变。"""
        refiner = ChunkRefiner(settings=_make_settings_stub())
        chunks = [_make_chunk("c1", "id1"), _make_chunk("c2", "id2")]
        result = refiner.transform(chunks)
        assert result[0].id == "id1"
        assert result[1].id == "id2"

    def test_transform_preserves_metadata(self):
        """transform 保持原有 metadata 字段。"""
        refiner = ChunkRefiner(settings=_make_settings_stub())
        chunk = _make_chunk("文本")
        chunk.metadata["custom_field"] = 42
        result = refiner.transform([chunk])
        assert result[0].metadata["custom_field"] == 42

    def test_empty_chunks_returns_empty(self):
        """空列表输入返回空列表。"""
        refiner = ChunkRefiner(settings=_make_settings_stub())
        result = refiner.transform([])
        assert result == []
