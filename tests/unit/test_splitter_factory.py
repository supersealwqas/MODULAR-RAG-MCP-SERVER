"""测试 Splitter 抽象接口与工厂。"""

import pytest
from typing import List

from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.splitter_factory import SplitterFactory, register_splitter, _SPLITTER_REGISTRY


# --- Fake Splitter 用于测试 ---

class FakeSplitter(BaseSplitter):
    """测试用的假 Splitter 实现，按 chunk_size 固定切分。"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, **kwargs):
        """初始化 FakeSplitter。

        参数:
            chunk_size: 每个文本块的最大字符数
            chunk_overlap: 相邻文本块之间的重叠字符数
            **kwargs: 其他参数
        """
        super().__init__(chunk_size, chunk_overlap, **kwargs)
        self.kwargs = kwargs

    def split_text(self, text: str, **kwargs) -> List[str]:
        """按 chunk_size 固定切分文本，考虑重叠。

        参数:
            text: 待切分的文本
            **kwargs: 其他参数

        返回:
            切分后的文本块列表
        """
        if not text:
            return []

        # 确保 overlap 不超过 chunk_size，防止无限循环
        effective_overlap = min(self.chunk_overlap, self.chunk_size - 1)

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            # 计算下一个起始位置，确保前进
            next_start = end - effective_overlap
            if next_start <= start:
                # 如果没有前进，强制前进到 end
                next_start = end
            start = next_start
            if start >= len(text):
                break
        return chunks


# --- 测试用例 ---

@pytest.mark.unit
class TestBaseSplitter:
    """测试 BaseSplitter 抽象类。"""

    def test_cannot_instantiate_abstract(self):
        """BaseSplitter 是抽象类，不能直接实例化。"""
        with pytest.raises(TypeError):
            BaseSplitter()

    def test_fake_splitter_split_text(self):
        """FakeSplitter 应正确实现 split_text 方法。"""
        splitter = FakeSplitter(chunk_size=10, chunk_overlap=2)
        text = "0123456789abcdef"
        chunks = splitter.split_text(text)
        assert len(chunks) > 1
        assert chunks[0] == "0123456789"

    def test_split_texts(self):
        """split_texts 应批量切分并展平结果。"""
        splitter = FakeSplitter(chunk_size=5, chunk_overlap=0)
        texts = ["hello world", "foo bar baz"]
        chunks = splitter.split_texts(texts)
        assert len(chunks) >= 2
        assert "hello" in chunks[0]

    def test_empty_text(self):
        """空文本应返回空列表。"""
        splitter = FakeSplitter(chunk_size=10)
        chunks = splitter.split_text("")
        assert chunks == []

    def test_short_text(self):
        """短文本应返回单个块。"""
        splitter = FakeSplitter(chunk_size=100)
        chunks = splitter.split_text("short")
        assert chunks == ["short"]

    def test_overlap_larger_than_chunk_size(self):
        """overlap >= chunk_size 时不应死循环。"""
        splitter = FakeSplitter(chunk_size=10, chunk_overlap=15)
        text = "a" * 30
        chunks = splitter.split_text(text)
        assert len(chunks) >= 1
        # 验证不会死循环，能正常返回
        assert all(len(chunk) <= 10 for chunk in chunks)


@pytest.mark.unit
class TestSplitterFactory:
    """测试 SplitterFactory 的路由逻辑。"""

    def setup_method(self):
        """每个测试前注册 fake 策略。"""
        _SPLITTER_REGISTRY["fake"] = FakeSplitter

    def teardown_method(self):
        """每个测试后清理注册表。"""
        _SPLITTER_REGISTRY.pop("fake", None)

    def test_create_registered_strategy(self):
        """应为已注册的策略创建实例。"""
        splitter = SplitterFactory.create(strategy="fake", chunk_size=256, chunk_overlap=20)
        assert isinstance(splitter, FakeSplitter)
        assert splitter.chunk_size == 256
        assert splitter.chunk_overlap == 20

    def test_create_default_params(self):
        """应使用默认参数创建实例。"""
        splitter = SplitterFactory.create(strategy="fake")
        assert splitter.chunk_size == 512
        assert splitter.chunk_overlap == 50

    def test_unknown_strategy_raises(self):
        """未注册的策略应抛出 ValueError。"""
        with pytest.raises(ValueError, match="未知的切分策略.*unknown"):
            SplitterFactory.create(strategy="unknown")

    def test_case_insensitive_strategy(self):
        """策略名称匹配应不区分大小写。"""
        splitter = SplitterFactory.create(strategy="FAKE")
        assert isinstance(splitter, FakeSplitter)

    def test_list_strategies(self):
        """应列出所有已注册的策略。"""
        strategies = SplitterFactory.list_strategies()
        assert "fake" in strategies


@pytest.mark.unit
class TestRegisterSplitterDecorator:
    """测试 @register_splitter 装饰器。"""

    def teardown_method(self):
        """清理注册表。"""
        _SPLITTER_REGISTRY.pop("test_strategy", None)

    def test_register_decorator(self):
        """@register_splitter 应将类添加到注册表。"""
        @register_splitter("test_strategy")
        class TestSplitter(BaseSplitter):
            def split_text(self, text, **kwargs):
                return [text]

        assert "test_strategy" in _SPLITTER_REGISTRY
        assert _SPLITTER_REGISTRY["test_strategy"] is TestSplitter

    def test_register_lowercase(self):
        """策略名称应存储为小写。"""
        @register_splitter("TEST_UPPER")
        class UpperSplitter(BaseSplitter):
            def split_text(self, text, **kwargs):
                return [text]

        assert "test_upper" in _SPLITTER_REGISTRY
        _SPLITTER_REGISTRY.pop("test_upper", None)
