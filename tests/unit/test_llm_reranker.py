"""测试 LLMReranker 实现。

使用 mock LLM 隔离测试，不走真实 API。
"""

import pytest
from typing import List
from unittest.mock import MagicMock

from src.libs.reranker.base_reranker import (
    BaseReranker,
    Candidate,
    RankedCandidate,
)
from src.libs.reranker.llm_reranker import LLMReranker
from src.libs.reranker.reranker_factory import RerankerFactory
from src.libs.llm.base_llm import BaseLLM


class FakeLLM(BaseLLM):
    """测试用 Fake LLM，返回预设响应。"""

    def __init__(self, response: str = "", model: str = "fake"):
        """初始化 FakeLLM。

        参数:
            response: 预设的响应内容
            model: 模型名称，默认为 "fake"
        """
        super().__init__(model=model)
        self._response = response
        self.last_prompt = ""

    def chat(self, messages, **kwargs):
        """发送聊天请求，返回预设响应。"""
        from src.libs.llm.base_llm import LLMResponse
        return LLMResponse(content=self._response, model=self.model)


@pytest.mark.unit
class TestLLMRerankerBasic:
    """测试 LLMReranker 基本功能。"""

    def test_is_subclass_of_base(self):
        """LLMReranker 应继承 BaseReranker。"""
        reranker = LLMReranker()
        assert isinstance(reranker, BaseReranker)

    def test_factory_creates_llm_reranker(self):
        """backend=llm 时 RerankerFactory 应创建 LLMReranker。"""
        reranker = RerankerFactory.create(provider="llm")
        assert isinstance(reranker, LLMReranker)

    def test_factory_case_insensitive(self):
        """工厂应不区分大小写。"""
        reranker = RerankerFactory.create(provider="LLM")
        assert isinstance(reranker, LLMReranker)

    def test_llm_in_list_providers(self):
        """llm 应在可用提供者列表中。"""
        providers = RerankerFactory.list_providers()
        assert "llm" in providers

    def test_set_llm(self):
        """set_llm 应注入 LLM 实例。"""
        reranker = LLMReranker()
        llm = FakeLLM()
        reranker.set_llm(llm)
        assert reranker._llm is llm


@pytest.mark.unit
class TestLLMRerankerRerank:
    """测试 LLMReranker 重排序逻辑。"""

    def _make_candidates(self, ids: List[str]) -> List[Candidate]:
        """构造测试候选文档。"""
        return [
            Candidate(id=cid, text=f"文档 {cid} 的内容", score=0.5)
            for cid in ids
        ]

    def test_rerank_basic(self):
        """基本重排序：LLM 返回有效 ID 列表。"""
        llm = FakeLLM(response="doc_2, doc_1, doc_3")
        reranker = LLMReranker(llm=llm)
        candidates = self._make_candidates(["doc_1", "doc_2", "doc_3"])

        results = reranker.rerank("如何配置 Ollama", candidates)

        assert len(results) == 3
        assert results[0].id == "doc_2"
        assert results[1].id == "doc_1"
        assert results[2].id == "doc_3"
        # 分数应递减
        assert results[0].rerank_score > results[1].rerank_score

    def test_rerank_preserves_original_score(self):
        """重排序应保留原始分数。"""
        llm = FakeLLM(response="doc_2, doc_1")
        reranker = LLMReranker(llm=llm)
        candidates = [
            Candidate(id="doc_1", text="内容1", score=0.8),
            Candidate(id="doc_2", text="内容2", score=0.6),
        ]

        results = reranker.rerank("查询文本", candidates)

        id_to_result = {r.id: r for r in results}
        assert id_to_result["doc_1"].original_score == 0.8
        assert id_to_result["doc_2"].original_score == 0.6

    def test_rerank_with_top_k(self):
        """top_k 应限制返回数量。"""
        llm = FakeLLM(response="doc_3, doc_1, doc_2")
        reranker = LLMReranker(llm=llm)
        candidates = self._make_candidates(["doc_1", "doc_2", "doc_3"])

        results = reranker.rerank("查询文本", candidates, top_k=2)

        assert len(results) == 2
        assert results[0].id == "doc_3"
        assert results[1].id == "doc_1"

    def test_rerank_empty_candidates(self):
        """空候选列表应返回空结果。"""
        reranker = LLMReranker(llm=FakeLLM())
        results = reranker.rerank("查询文本", [])
        assert results == []

    def test_rerank_missing_ids_appended(self):
        """LLM 未提及的候选应放在末尾，分数为 0。"""
        llm = FakeLLM(response="doc_1")  # 只提到 doc_1
        reranker = LLMReranker(llm=llm)
        candidates = self._make_candidates(["doc_1", "doc_2", "doc_3"])

        results = reranker.rerank("查询文本", candidates)

        assert len(results) == 3
        assert results[0].id == "doc_1"
        assert results[0].rerank_score > 0
        # 未提及的候选应在末尾
        missing_ids = {r.id for r in results[1:]}
        assert missing_ids == {"doc_2", "doc_3"}
        for r in results[1:]:
            assert r.rerank_score == 0.0
            assert r.metadata.get("rerank_missing") is True


@pytest.mark.unit
class TestLLMRerankerResponseParsing:
    """测试 LLM 响应解析。"""

    def test_parse_comma_separated(self):
        """应解析逗号分隔的 ID 列表。"""
        reranker = LLMReranker()
        ids = reranker._parse_ranked_ids(
            "doc_b, doc_a, doc_c",
            {"doc_a", "doc_b", "doc_c"},
        )
        assert ids == ["doc_b", "doc_a", "doc_c"]

    def test_parse_bracket_format(self):
        """应解析 [id] 格式。"""
        reranker = LLMReranker()
        ids = reranker._parse_ranked_ids(
            "[doc_b], [doc_a], [doc_c]",
            {"doc_a", "doc_b", "doc_c"},
        )
        assert ids == ["doc_b", "doc_a", "doc_c"]

    def test_parse_numbered_list(self):
        """应解析编号列表格式。"""
        reranker = LLMReranker()
        ids = reranker._parse_ranked_ids(
            "1. doc_c\n2. doc_a\n3. doc_b",
            {"doc_a", "doc_b", "doc_c"},
        )
        assert ids == ["doc_c", "doc_a", "doc_b"]

    def test_parse_newline_separated(self):
        """应解析换行分隔的 ID。"""
        reranker = LLMReranker()
        ids = reranker._parse_ranked_ids(
            "doc_a\ndoc_c\ndoc_b",
            {"doc_a", "doc_b", "doc_c"},
        )
        assert ids == ["doc_a", "doc_c", "doc_b"]

    def test_parse_ignores_invalid_ids(self):
        """应忽略不在 valid_ids 中的 ID。"""
        reranker = LLMReranker()
        ids = reranker._parse_ranked_ids(
            "doc_a, doc_unknown, doc_b",
            {"doc_a", "doc_b"},
        )
        assert ids == ["doc_a", "doc_b"]

    def test_parse_empty_response(self):
        """空响应应返回空列表。"""
        reranker = LLMReranker()
        ids = reranker._parse_ranked_ids("", {"doc_a"})
        assert ids == []

    def test_parse_preserves_order(self):
        """解析应保持 ID 出现顺序。"""
        reranker = LLMReranker()
        ids = reranker._parse_ranked_ids(
            "doc_c, doc_a, doc_b",
            {"doc_a", "doc_b", "doc_c"},
        )
        assert ids == ["doc_c", "doc_a", "doc_b"]


@pytest.mark.unit
class TestLLMRerankerErrorHandling:
    """测试 LLMReranker 错误处理。"""

    def test_no_llm_raises_runtime_error(self):
        """未配置 LLM 时应抛出 RuntimeError。"""
        reranker = LLMReranker()
        candidates = [Candidate(id="doc_1", text="内容", score=0.5)]

        with pytest.raises(RuntimeError, match="未配置 LLM"):
            reranker.rerank("查询文本", candidates)

    def test_llm_call_failure_raises_runtime_error(self):
        """LLM 调用失败时应抛出 RuntimeError。"""
        llm = MagicMock(spec=BaseLLM)
        llm.chat_simple.side_effect = ConnectionError("连接超时")
        reranker = LLMReranker(llm=llm)
        candidates = [Candidate(id="doc_1", text="内容", score=0.5)]

        with pytest.raises(RuntimeError, match="LLM Reranker 调用失败"):
            reranker.rerank("查询文本", candidates)

    def test_unparseable_response_raises_runtime_error(self):
        """无法解析响应时应抛出 RuntimeError。"""
        llm = FakeLLM(response="这是一段完全无关的回复，没有任何有效ID")
        reranker = LLMReranker(llm=llm)
        candidates = [Candidate(id="doc_1", text="内容", score=0.5)]

        with pytest.raises(RuntimeError, match="无法从响应中解析"):
            reranker.rerank("查询文本", candidates)


@pytest.mark.unit
class TestLLMRerankerPrompt:
    """测试 prompt 构造。"""

    def test_prompt_file_not_found_raises(self):
        """prompt 文件不存在时应抛出 FileNotFoundError。"""
        reranker = LLMReranker(prompt_path="non_existent_path.txt")
        with pytest.raises(FileNotFoundError, match="prompt 文件不存在"):
            reranker._load_prompt_template()

    def test_loads_prompt_from_file(self):
        """应从文件加载 prompt 模板。"""
        reranker = LLMReranker()  # 使用默认路径 config/prompts/rerank.txt
        template = reranker._load_prompt_template()
        assert "{query}" in template
        assert "{passages}" in template

    def test_format_passages(self):
        """候选文档应格式化为带编号列表。"""
        reranker = LLMReranker()
        candidates = [
            Candidate(id="doc_1", text="内容A", score=0.5),
            Candidate(id="doc_2", text="内容B", score=0.6),
        ]
        formatted = reranker._format_passages(candidates)
        assert "[doc_1] 内容A" in formatted
        assert "[doc_2] 内容B" in formatted

    def test_format_passages_truncates_long_text(self):
        """过长文本应被截断。"""
        reranker = LLMReranker()
        candidates = [Candidate(id="doc_1", text="x" * 1000, score=0.5)]
        formatted = reranker._format_passages(candidates)
        assert "..." in formatted
        assert len(formatted) < 600
