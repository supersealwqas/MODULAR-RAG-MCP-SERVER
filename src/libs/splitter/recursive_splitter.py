"""递归字符文本切分器模块。

基于 LangChain 的 RecursiveCharacterTextSplitter 思路实现，
按分隔符优先级递归切分文本，尽量保持语义完整性。
支持 Markdown 结构（标题、代码块）不被打断。
"""

from __future__ import annotations

import re
from typing import List, Optional

from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.splitter_factory import register_splitter


# 默认分隔符列表，按优先级排序
# 优先按段落、换行、句子、空格切分，最后才按字符切分
DEFAULT_SEPARATORS = [
    "\n\n",  # 段落分隔
    "\n",    # 换行
    "。",    # 中文句号
    "！",    # 中文感叹号
    "？",    # 中文问号
    ". ",    # 英文句号（带空格）
    "! ",    # 英文感叹号
    "? ",    # 英文问号
    "；",    # 中文分号
    "; ",    # 英文分号
    ", ",    # 英文逗号
    "，",    # 中文逗号
    " ",     # 空格
    "",      # 最后按字符切分
]


@register_splitter("recursive")
class RecursiveSplitter(BaseSplitter):
    """递归字符文本切分器。

    按分隔符优先级递归切分文本，尽量在自然边界（段落、句子）处切分。
    支持保护 Markdown 代码块和标题不被切碎。

    特性:
        - 递归切分：优先在大粒度分隔符处切分，不行再降级到小粒度
        - 代码块保护：``` 包裹的代码块内部不切分
        - 标题保护：Markdown 标题（# 开头）不与其后内容分离
        - 重叠支持：相邻 chunk 可配置重叠字符数以保持上下文连贯
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        keep_code_blocks: bool = True,
        keep_headers: bool = True,
        **kwargs,
    ):
        """初始化递归切分器。

        参数:
            chunk_size: 每个文本块的最大字符数（默认 1000）
            chunk_overlap: 相邻文本块之间的重叠字符数（默认 200）
            separators: 自定义分隔符列表（按优先级排序），为 None 时使用默认列表
            keep_code_blocks: 是否保护 Markdown 代码块不被切碎（默认 True）
            keep_headers: 是否保持标题与其后内容不分离（默认 True）
            **kwargs: 其他参数
        """
        super().__init__(chunk_size, chunk_overlap, **kwargs)
        self.separators = separators if separators is not None else DEFAULT_SEPARATORS
        self.keep_code_blocks = keep_code_blocks
        self.keep_headers = keep_headers

    def split_text(self, text: str, **kwargs) -> List[str]:
        """将文本递归切分为多个文本块。

        参数:
            text: 待切分的文本
            **kwargs: 额外参数

        返回:
            切分后的文本块列表
        """
        if not text:
            return []

        # 预处理：提取并保护代码块
        if self.keep_code_blocks:
            chunks = self._split_with_code_protection(text)
        else:
            chunks = self._recursive_split(text, self.separators)

        # 后处理：合并过短的块，添加重叠
        chunks = self._merge_short_chunks(chunks)
        chunks = self._add_overlap(chunks)

        return chunks

    def _split_with_code_protection(self, text: str) -> List[str]:
        """保护 Markdown 代码块不被切碎。

        将代码块整体提取出来作为独立块，对非代码部分递归切分。

        参数:
            text: 原始文本

        返回:
            切分后的文本块列表
        """
        # 匹配 ``` 包裹的代码块
        code_block_pattern = re.compile(r"```[\s\S]*?```", re.MULTILINE)
        parts = []
        last_end = 0

        for match in code_block_pattern.finditer(text):
            # 代码块前的普通文本
            before = text[last_end:match.start()]
            if before.strip():
                parts.extend(self._recursive_split(before, self.separators))
            # 代码块整体作为一个块（如果太长则内部切分）
            code_block = match.group()
            if len(code_block) <= self.chunk_size:
                parts.append(code_block)
            else:
                parts.extend(self._split_long_code_block(code_block))
            last_end = match.end()

        # 最后一段普通文本
        after = text[last_end:]
        if after.strip():
            parts.extend(self._recursive_split(after, self.separators))

        return parts

    def _split_long_code_block(self, code_block: str) -> List[str]:
        """对超长代码块进行内部切分。

        保留开头的 ``` 标记和结尾的 ``` 标记。

        参数:
            code_block: 代码块文本

        返回:
            切分后的代码块片段列表
        """
        lines = code_block.split("\n")
        chunks = []
        current_chunk = []
        current_len = 0

        for line in lines:
            line_len = len(line) + 1  # +1 for newline
            if current_len + line_len > self.chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_len = 0
            current_chunk.append(line)
            current_len += line_len

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """递归切分文本。

        按分隔符优先级依次尝试切分，直到所有块都不超过 chunk_size。

        参数:
            text: 待切分文本
            separators: 当前可用的分隔符列表

        返回:
            切分后的文本块列表
        """
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            # 没有更多分隔符，硬切
            return self._hard_split(text)

        sep = separators[0]
        remaining_separators = separators[1:]

        if sep == "":
            # 空分隔符 = 按字符切分
            return self._hard_split(text)

        # 按当前分隔符切分
        parts = text.split(sep)

        if len(parts) == 1:
            # 分隔符不存在，尝试下一个
            return self._recursive_split(text, remaining_separators)

        # 合并小块，使其接近 chunk_size
        chunks = []
        current_chunk = ""

        for i, part in enumerate(parts):
            candidate = current_chunk + (sep if current_chunk else "") + part

            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # 如果单个 part 就超了，递归用更小的分隔符切
                if len(part) > self.chunk_size:
                    sub_chunks = self._recursive_split(part, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _hard_split(self, text: str) -> List[str]:
        """硬切分：按 chunk_size 强制切分。

        参数:
            text: 待切分文本

        返回:
            切分后的文本块列表
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end
        return chunks

    def _merge_short_chunks(self, chunks: List[str]) -> List[str]:
        """合并过短的文本块。

        如果相邻块合并后不超过 chunk_size，则合并它们。

        参数:
            chunks: 原始文本块列表

        返回:
            合并后的文本块列表
        """
        if not chunks:
            return []

        # 保护标题：标题行与其后内容不分离
        if self.keep_headers:
            chunks = self._protect_headers(chunks)

        merged = [chunks[0]]
        for chunk in chunks[1:]:
            if len(merged[-1]) + len(chunk) + 1 <= self.chunk_size:
                merged[-1] = merged[-1] + "\n" + chunk
            else:
                merged.append(chunk)
        return merged

    def _protect_headers(self, chunks: List[str]) -> List[str]:
        """保护 Markdown 标题不与后续内容分离。

        如果一个块只包含标题行（# 开头），则与下一个块合并。

        参数:
            chunks: 文本块列表

        返回:
            处理后的文本块列表
        """
        if len(chunks) <= 1:
            return chunks

        result = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            # 检查是否是纯标题块
            lines = chunk.strip().split("\n")
            is_header_only = (
                len(lines) == 1
                and lines[0].startswith("#")
                and i + 1 < len(chunks)
            )

            if is_header_only:
                # 合并标题与下一个块
                next_chunk = chunks[i + 1]
                if len(chunk) + len(next_chunk) + 1 <= self.chunk_size:
                    result.append(chunk + "\n" + next_chunk)
                    i += 2
                    continue

            result.append(chunk)
            i += 1

        return result

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """为相邻文本块添加重叠。

        参数:
            chunks: 原始文本块列表

        返回:
            添加重叠后的文本块列表
        """
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            # 从前一个块的末尾取 overlap 个字符
            prev = chunks[i - 1]
            overlap_text = prev[-self.chunk_overlap:]
            # 尝试在自然边界处截断重叠文本
            overlap_text = self._find_natural_boundary(overlap_text)
            new_chunk = overlap_text + chunks[i]
            # 确保不超过 chunk_size
            if len(new_chunk) > self.chunk_size:
                new_chunk = new_chunk[:self.chunk_size]
            result.append(new_chunk)

        return result

    def _find_natural_boundary(self, text: str) -> str:
        """在自然边界处截断重叠文本。

        优先在句子、换行处截断，避免截断在词中间。

        参数:
            text: 重叠文本

        返回:
            截断后的文本
        """
        # 尝试在换行处截断
        idx = text.find("\n")
        if idx >= 0:
            return text[idx + 1:]

        # 尝试在空格处截断
        idx = text.find(" ")
        if idx >= 0:
            return text[idx + 1:]

        # 无法找到自然边界，直接返回
        return text
