"""测试 RecursiveSplitter 默认实现。"""

import pytest
from typing import List

from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.recursive_splitter import RecursiveSplitter
from src.libs.splitter.splitter_factory import SplitterFactory


@pytest.mark.unit
class TestRecursiveSplitterBasic:
    """测试 RecursiveSplitter 基本功能。"""

    def test_is_subclass_of_base(self):
        """RecursiveSplitter 应继承 BaseSplitter。"""
        splitter = RecursiveSplitter()
        assert isinstance(splitter, BaseSplitter)

    def test_default_params(self):
        """默认参数应正确设置。"""
        splitter = RecursiveSplitter()
        assert splitter.chunk_size == 1000
        assert splitter.chunk_overlap == 200
        assert splitter.keep_code_blocks is True
        assert splitter.keep_headers is True

    def test_custom_params(self):
        """自定义参数应正确传递。"""
        splitter = RecursiveSplitter(
            chunk_size=256,
            chunk_overlap=30,
            keep_code_blocks=False,
            keep_headers=False,
        )
        assert splitter.chunk_size == 256
        assert splitter.chunk_overlap == 30
        assert splitter.keep_code_blocks is False
        assert splitter.keep_headers is False

    def test_empty_text(self):
        """空文本应返回空列表。"""
        splitter = RecursiveSplitter(chunk_size=100)
        assert splitter.split_text("") == []

    def test_short_text(self):
        """短文本应返回单个块。"""
        splitter = RecursiveSplitter(chunk_size=100)
        text = "这是一个短文本。"
        chunks = splitter.split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text


@pytest.mark.unit
class TestRecursiveSplitterSplitting:
    """测试 RecursiveSplitter 切分逻辑。"""

    def test_split_by_paragraph(self):
        """应优先在段落边界处切分。"""
        splitter = RecursiveSplitter(chunk_size=30, chunk_overlap=0)
        text = "第一段内容比较长一些需要超过限制。\n\n第二段内容也比较长需要切分。\n\n第三段内容同样需要处理。"
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2
        # 每个 chunk 不应超过 chunk_size
        for chunk in chunks:
            assert len(chunk) <= 30

    def test_split_by_newline(self):
        """段落内应按换行切分。"""
        splitter = RecursiveSplitter(chunk_size=15, chunk_overlap=0)
        text = "第一行内容比较长\n第二行内容也比较长\n第三行内容\n第四行内容"
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2

    def test_split_respects_chunk_size(self):
        """所有块不应超过 chunk_size。"""
        splitter = RecursiveSplitter(chunk_size=30, chunk_overlap=0)
        text = "这是一段比较长的文本，需要被切分成多个小块来满足 chunk_size 的限制要求。"
        chunks = splitter.split_text(text)
        for chunk in chunks:
            assert len(chunk) <= 30

    def test_split_preserves_all_content(self):
        """切分后所有内容应完整保留（无重叠时）。"""
        splitter = RecursiveSplitter(chunk_size=50, chunk_overlap=0)
        text = "第一部分。第二部分。第三部分。第四部分。第五部分。"
        chunks = splitter.split_text(text)
        reconstructed = "".join(chunks)
        # 验证内容完整性，不仅仅是长度
        assert reconstructed == text

    def test_overlap_creates_repeated_content(self):
        """重叠应导致相邻块有重复内容。"""
        splitter = RecursiveSplitter(chunk_size=20, chunk_overlap=5)
        text = "AAAAAAAAAABBBBBBBBBBCCCCCCCCCC"
        chunks = splitter.split_text(text)
        if len(chunks) >= 2:
            # 第二个块的开头应与第一个块的末尾有重叠
            assert chunks[1][:5] == chunks[0][-5:]

    def test_custom_separators(self):
        """自定义分隔符应被使用。"""
        splitter = RecursiveSplitter(
            chunk_size=10,
            chunk_overlap=0,
            separators=["|", "-"],
        )
        text = "aaa|bbb-ccc"
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2


@pytest.mark.unit
class TestRecursiveSplitterMarkdown:
    """测试 RecursiveSplitter Markdown 结构保护。"""

    def test_code_block_protection(self):
        """代码块内部不应被切碎。"""
        splitter = RecursiveSplitter(chunk_size=100, chunk_overlap=0)
        text = """一些说明文字。

```python
def hello():
    print("hello world")
    return True
```

更多说明文字。"""
        chunks = splitter.split_text(text)
        # 代码块应完整出现在某个 chunk 中
        code_found = False
        for chunk in chunks:
            if "```python" in chunk and "```" in chunk.split("```python")[1]:
                code_found = True
                assert "def hello():" in chunk
                assert "return True" in chunk
                break
        assert code_found, "代码块应完整保留在某个 chunk 中"

    def test_header_protection(self):
        """标题不应与其后内容分离。"""
        splitter = RecursiveSplitter(chunk_size=50, chunk_overlap=0)
        text = """## 第一章

这是第一章的内容。

## 第二章

这是第二章的内容。"""
        chunks = splitter.split_text(text)
        # 标题应与后续内容在同一 chunk 中
        for chunk in chunks:
            if chunk.startswith("## 第一章"):
                assert "第一章的内容" in chunk
            elif chunk.startswith("## 第二章"):
                assert "第二章的内容" in chunk

    def test_multiple_code_blocks(self):
        """多个代码块都应被保护。"""
        splitter = RecursiveSplitter(chunk_size=80, chunk_overlap=0)
        text = """段落一。

```python
x = 1
```

段落二。

```javascript
const y = 2
```

段落三。"""
        chunks = splitter.split_text(text)
        # 两个代码块都应完整
        python_found = any("```python" in c and "x = 1" in c for c in chunks)
        js_found = any("```javascript" in c and "const y = 2" in c for c in chunks)
        assert python_found, "Python 代码块应完整保留"
        assert js_found, "JavaScript 代码块应完整保留"

    def test_disable_code_protection(self):
        """关闭代码块保护时，代码块可被切碎。"""
        splitter = RecursiveSplitter(
            chunk_size=30,
            chunk_overlap=0,
            keep_code_blocks=False,
        )
        text = "一些文字。\n\n```python\nprint('hello very long code line here')\n```\n\n更多文字。"
        chunks = splitter.split_text(text)
        # 不保护时，代码块可能被切分
        assert len(chunks) >= 1


@pytest.mark.unit
class TestRecursiveSplitterChinese:
    """测试 RecursiveSplitter 中文文本处理。"""

    def test_chinese_sentence_split(self):
        """中文句子应在句号处切分。"""
        splitter = RecursiveSplitter(chunk_size=15, chunk_overlap=0)
        text = "这是第一个句子。这是第二个句子。这是第三个句子。"
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2

    def test_chinese_mixed_text(self):
        """中英文混合文本应正确切分。"""
        splitter = RecursiveSplitter(chunk_size=30, chunk_overlap=0)
        text = "RAG 是 Retrieval Augmented Generation 的缩写。它结合了检索和生成两种技术。"
        chunks = splitter.split_text(text)
        for chunk in chunks:
            assert len(chunk) <= 30


@pytest.mark.unit
class TestRecursiveSplitterFactory:
    """测试 RecursiveSplitter 通过工厂创建。"""

    def test_factory_creates_recursive(self):
        """SplitterFactory 应能创建 RecursiveSplitter。"""
        splitter = SplitterFactory.create(strategy="recursive")
        assert isinstance(splitter, RecursiveSplitter)

    def test_factory_case_insensitive(self):
        """工厂应不区分大小写。"""
        splitter = SplitterFactory.create(strategy="Recursive")
        assert isinstance(splitter, RecursiveSplitter)

    def test_factory_with_params(self):
        """工厂应传递参数。"""
        splitter = SplitterFactory.create(
            strategy="recursive",
            chunk_size=256,
            chunk_overlap=20,
        )
        assert isinstance(splitter, RecursiveSplitter)
        assert splitter.chunk_size == 256
        assert splitter.chunk_overlap == 20

    def test_recursive_in_strategies_list(self):
        """recursive 应在可用策略列表中。"""
        strategies = SplitterFactory.list_strategies()
        assert "recursive" in strategies


@pytest.mark.unit
class TestRecursiveSplitterEdgeCases:
    """测试 RecursiveSplitter 边界情况。"""

    def test_single_long_line(self):
        """单行超长文本应被硬切。"""
        splitter = RecursiveSplitter(chunk_size=10, chunk_overlap=0)
        text = "a" * 50
        chunks = splitter.split_text(text)
        assert len(chunks) == 5
        for chunk in chunks:
            assert len(chunk) == 10

    def test_only_whitespace(self):
        """纯空白文本应返回空列表或单个块。"""
        splitter = RecursiveSplitter(chunk_size=100)
        chunks = splitter.split_text("   \n\n   ")
        # 空白文本可能返回空或包含空白的块
        assert len(chunks) <= 1

    def test_exact_chunk_size(self):
        """刚好等于 chunk_size 的文本应返回单个块。"""
        splitter = RecursiveSplitter(chunk_size=10, chunk_overlap=0)
        text = "a" * 10
        chunks = splitter.split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_overlap_larger_than_chunk(self):
        """overlap 大于 chunk_size 时不应死循环。"""
        splitter = RecursiveSplitter(chunk_size=10, chunk_overlap=15)
        text = "a" * 30
        chunks = splitter.split_text(text)
        assert len(chunks) >= 1

    def test_split_texts_batch(self):
        """split_texts 应正确批量切分。"""
        splitter = RecursiveSplitter(chunk_size=20, chunk_overlap=0)
        texts = ["文本一内容比较长需要切分", "文本二内容也比较长需要切分"]
        chunks = splitter.split_texts(texts)
        assert len(chunks) >= 2
