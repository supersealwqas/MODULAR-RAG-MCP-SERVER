"""递归字符文本切分器模块。

基于 LangChain 的 RecursiveCharacterTextSplitter 思路实现，
按分隔符优先级递归切分文本，尽量保持语义完整性。
支持 Markdown 结构（标题、代码块）不被打断。

改进点（相比初版）：
- overlap 机制：从后处理改为内嵌合并（LangChain _merge_splits 风格），
  overlap 在分隔符边界截断，不会切断词语。
- 新增 length_function 参数，支持按 token 计数等自定义长度函数。
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

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
        - 内嵌合并：小块合并 + overlap 在递归过程中一步完成，
          overlap 在分隔符边界截断（LangChain _merge_splits 风格）
        - length_function 可配置：支持按 token 计数等自定义长度函数
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        keep_code_blocks: bool = True,
        keep_headers: bool = True,
        length_function: Optional[Callable[[str], int]] = None,
        **kwargs,
    ):
        """初始化递归切分器。

        参数:
            chunk_size: 每个文本块的最大字符数（默认 1000）
            chunk_overlap: 相邻文本块之间的重叠字符数（默认 200）
            separators: 自定义分隔符列表（按优先级排序），为 None 时使用默认列表
            keep_code_blocks: 是否保护 Markdown 代码块不被切碎（默认 True）
            keep_headers: 是否保持标题与其后内容不分离（默认 True）
            length_function: 自定义长度函数（默认 len，可传入 token 计数器）
            **kwargs: 其他参数
        """
        super().__init__(chunk_size, chunk_overlap, **kwargs)
        self.separators = separators if separators is not None else DEFAULT_SEPARATORS
        self.keep_code_blocks = keep_code_blocks
        self.keep_headers = keep_headers
        self.length_function = length_function or len

    def _len(self, text: str) -> int:
        """计算文本长度（使用 length_function）。"""
        return self.length_function(text)

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

        # 后处理：保护标题不被孤立
        if self.keep_headers:
            chunks = self._protect_headers(chunks)

        # 过滤空白块
        chunks = [c for c in chunks if c.strip()]

        return chunks

    def _split_with_code_protection(self, text: str) -> List[str]:
        """保护 Markdown 代码块不被切碎。

        使用字符扫描提取代码块，避免正则回溯风险。
        对于未闭合的代码块，将其整体视为代码块处理。

        参数:
            text: 原始文本

        返回:
            切分后的文本块列表
        """
        code_blocks = self._scan_code_blocks(text)
        parts = []
        last_end = 0

        for start, end, lang in code_blocks:
            # 代码块前的普通文本
            before = text[last_end:start]
            if before.strip():
                parts.extend(self._recursive_split(before, self.separators))
            # 代码块整体作为一个块（如果太长则内部切分）
            code_block = text[start:end]
            if self._len(code_block) <= self.chunk_size:
                parts.append(code_block)
            else:
                parts.extend(self._split_long_code_block(code_block, lang))
            last_end = end

        # 最后一段普通文本
        after = text[last_end:]
        if after.strip():
            parts.extend(self._recursive_split(after, self.separators))

        return parts

    def _scan_code_blocks(self, text: str) -> List[Tuple[int, int, str]]:
        """字符扫描提取 Markdown 代码块位置，避免正则回溯。

        扫描策略：
        - 找到行首的 ``` 作为代码块起点
        - 从起点之后持续向后查找，直到找到行首的 ``` 作为终点
        - 未闭合的代码块延伸到文本末尾

        参数:
            text: 原始文本

        返回:
            [(start, end, lang), ...] 代码块的起止位置和语言标记列表
        """
        blocks = []
        i = 0
        text_len = len(text)

        while i < text_len:
            # 查找 ```
            idx = text.find("```", i)
            if idx == -1:
                break

            # 确认 ``` 在行首（或文本开头）
            if idx > 0 and text[idx - 1] not in ("\n", "\r"):
                i = idx + 3
                continue

            # 提取语言标记（``` 后到行尾的内容）
            lang_start = idx + 3
            lang_end = text.find("\n", lang_start)
            if lang_end == -1:
                lang_end = text_len
            lang = text[lang_start:lang_end].strip()

            # 持续向后查找行首的闭合 ```
            search_from = lang_end
            close_idx = -1
            while True:
                pos = text.find("```", search_from)
                if pos == -1:
                    break
                # 判断是否为行首的合法闭合标记
                if pos == 0 or text[pos - 1] in ("\n", "\r"):
                    close_idx = pos
                    break
                # 不是行首，跳过这个假标记继续往后找
                search_from = pos + 3

            # 未闭合：延伸到文本末尾
            if close_idx == -1:
                blocks.append((idx, text_len, lang))
                break

            # 计算结束位置（包含 ``` 本身 + 行尾）
            end_pos = close_idx + 3
            if end_pos < text_len and text[end_pos] in ("\n", "\r"):
                # Windows \r\n 需要吞掉两个字符
                if end_pos + 1 < text_len and text[end_pos:end_pos + 2] == "\r\n":
                    end_pos += 2
                else:
                    end_pos += 1

            blocks.append((idx, end_pos, lang))
            i = end_pos

        return blocks

    def _split_long_code_block(self, code_block: str, lang: str = "") -> List[str]:
        """对超长代码块进行内部切分。

        保留开头的 ``` 标记和结尾的 ``` 标记。
        每个子块都是语法上完整的闭合代码块：
        - 非首块：补上 ```lang 前缀
        - 非末块：补上 ``` 闭合后缀

        参数:
            code_block: 代码块文本
            lang: 代码语言标记（如 "python"）

        返回:
            切分后的代码块片段列表
        """
        lines = code_block.split("\n")
        chunks = []
        current_chunk = []
        current_len = 0

        for line in lines:
            line_len = self._len(line) + 1  # +1 for newline
            if current_len + line_len > self.chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_len = 0
            current_chunk.append(line)
            current_len += line_len

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        # 确保每个子块都是语法上完整的闭合代码块
        if len(chunks) > 1:
            lang_prefix = f"```{lang}\n"
            close_suffix = "\n```"

            for i in range(len(chunks)):
                # 非首块：补上语言标记前缀
                if i > 0 and not chunks[i].startswith("```"):
                    chunks[i] = lang_prefix + chunks[i]
                # 非末块：补上闭合标记后缀
                if i < len(chunks) - 1 and not chunks[i].endswith("```"):
                    chunks[i] = chunks[i] + close_suffix

        return chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """递归切分文本，内嵌合并与 overlap。

        按分隔符优先级依次尝试切分，同时完成小块合并和 overlap 处理。
        分隔符作为独立元素保留在切分结果中，确保重建文本时内容不丢失。

        参数:
            text: 待切分文本
            separators: 当前可用的分隔符列表

        返回:
            切分后的文本块列表
        """
        if self._len(text) <= self.chunk_size:
            return [text]

        # 确定切分策略
        if not separators:
            # 无更多分隔符，硬切 + 合并
            parts = self._hard_split(text)
            return self._merge_splits(parts, "")

        sep = separators[0]
        remaining_separators = separators[1:]

        if sep == "":
            # 按字符硬切 + 合并
            parts = self._hard_split(text)
            return self._merge_splits(parts, "")

        # 按当前分隔符切分，保留分隔符作为独立元素
        parts = self._split_keep_separator(text, sep)

        if len(parts) == 1:
            # 分隔符不存在，尝试下一个
            return self._recursive_split(text, remaining_separators)

        # 递归处理超大块
        refined_parts = []
        for part in parts:
            # 跳过分隔符元素（长度短且与分隔符相同）
            if part == sep:
                refined_parts.append(part)
            elif self._len(part) > self.chunk_size:
                # 超大块递归切分
                sub_chunks = self._recursive_split(part, remaining_separators)
                refined_parts.extend(sub_chunks)
            else:
                refined_parts.append(part)

        # 合并小块 + 内嵌 overlap（LangChain _merge_splits 风格）
        return self._merge_splits(refined_parts, sep)

    def _split_keep_separator(self, text: str, separator: str) -> List[str]:
        """切分文本并保留分隔符作为独立元素。

        例如：split_keep_separator("aaa|bbb|ccc", "|")
        → ["aaa", "|", "bbb", "|", "ccc"]

        参数:
            text: 待切分文本
            separator: 分隔符

        返回:
            切分结果列表，分隔符作为独立元素
        """
        parts = text.split(separator)
        result = []
        for i, part in enumerate(parts):
            result.append(part)
            if i < len(parts) - 1:
                result.append(separator)
        return result

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """合并小块并处理 overlap（LangChain _merge_splits 风格）。

        核心逻辑：
        - 小块累积到 current_doc，总长度超过 chunk_size 时 flush
        - flush 后 pop 前端直到剩余长度 ≈ chunk_overlap
        - overlap 在分隔符边界截断，不会切断词语

        参数:
            splits: 切分后的片段列表（包含分隔符元素）
            separator: 当前层级的分隔符（用于长度计算）

        返回:
            合并后的文本块列表
        """
        separator_len = self._len(separator)
        current_doc: List[str] = []
        total = 0
        docs: List[str] = []

        for d in splits:
            d_len = self._len(d)

            # 加入 d 后是否超过 chunk_size
            sep_cost = separator_len if current_doc else 0
            if total + d_len + sep_cost > self.chunk_size:
                # flush 当前累积块
                if current_doc:
                    doc = self._join_docs(current_doc)
                    if doc:
                        docs.append(doc)

                    # pop 前端直到剩余长度 ≈ chunk_overlap
                    while total > self.chunk_overlap or (
                        total + d_len + sep_cost > self.chunk_size and total > 0
                    ):
                        if not current_doc:
                            break
                        removed = current_doc.pop(0)
                        removed_len = self._len(removed)
                        if current_doc:
                            total -= removed_len + separator_len
                        else:
                            total -= removed_len
                        if total <= 0:
                            total = 0
                            break

            # 加入当前片段
            current_doc.append(d)
            total += d_len + (separator_len if len(current_doc) > 1 else 0)

        # flush 最后一块
        if current_doc:
            doc = self._join_docs(current_doc)
            if doc:
                docs.append(doc)

        return docs

    def _join_docs(self, docs: List[str]) -> str:
        """连接文本片段。

        因为分隔符已经作为独立元素保留在列表中，直接拼接即可。

        参数:
            docs: 文本片段列表

        返回:
            连接后的文本
        """
        text = "".join(docs)
        text = text.strip()
        return text if text else ""

    def _hard_split(self, text: str) -> List[str]:
        """硬切分：按 step 步长切分为小片段，供 _merge_splits 处理 overlap。

        当 length_function 不是 len 时（如 token 计数），通过采样估算
        字符/token 比例，按比例缩放字符切片范围，避免切出过小的块。
        下一个 chunk 的起点基于实际截断位置往回退 overlap，确保无数据丢失。

        参数:
            text: 待切分文本

        返回:
            切分后的文本片段列表
        """
        text_char_len = len(text)
        text_token_len = self._len(text)

        # 估算字符/token 比例：如果 length_function 不是 len，
        # 用采样方式估算每个 token 对应多少字符
        if text_token_len > 0 and self.length_function is not len:
            ratio = text_char_len / text_token_len
        else:
            ratio = 1.0

        # 按比例缩放 chunk_size 和 overlap 到字符维度
        char_chunk_size = max(1, int(self.chunk_size * ratio))
        char_overlap = max(0, int(self.chunk_overlap * ratio))

        chunks = []
        start = 0
        while start < text_char_len:
            end = min(start + char_chunk_size, text_char_len)
            # 尝试在词边界（空格、换行）处截断，避免切断词语
            if end < text_char_len:
                end = self._find_word_boundary(text, end)

            # 防御：_find_word_boundary 可能将 end 退回到 start 之前
            if end <= start:
                end = min(start + char_chunk_size, text_char_len)

            chunks.append(text[start:end])

            # 下一个起点基于实际截断的 end 往回退 overlap，确保无数据丢失
            next_start = end - char_overlap
            # 必须确保 start 在前进，防止 overlap 过大导致死循环
            start = max(start + 1, next_start)

        return chunks

    def _find_word_boundary(self, text: str, pos: int, search_range: int = 50) -> int:
        """在 pos 附近寻找词边界（空格、换行、标点）。

        优先在 pos 之前找到最近的边界，找不到则在 pos 之后查找。

        参数:
            text: 原始文本
            pos: 目标位置
            search_range: 搜索范围（字符数）

        返回:
            调整后的位置
        """
        boundaries = {
            " ", "\n", "\r", "\t",
            # 中文标点
            "。", "！", "？", "；", "，", "：", "、",
            # 中文括号/引号
            "（", "）", "【", "】", "「", "」", """, """, "'", "'",
            # 英文标点
            ".", "!", "?", ";", ",",
            # 英文括号/冒号
            ":", ")", "]", "}",
        }

        # 向前搜索
        for offset in range(search_range):
            check = pos - offset
            if check > 0 and text[check - 1] in boundaries:
                return check

        # 向后搜索
        for offset in range(1, search_range):
            check = pos + offset
            if check < len(text) and text[check - 1] in boundaries:
                return check

        return pos

    def _protect_headers(self, chunks: List[str]) -> List[str]:
        """保护 Markdown 标题不与后续内容分离。

        两种情况触发合并：
        1. 纯标题块（仅一行 # 开头）→ 与下一个块合并
        2. 标题首行块（首行 # 开头，内容过短）→ 与下一个块合并

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
            lines = chunk.strip().split("\n")
            first_line = lines[0].strip()

            # 判断是否为需要保护的标题块
            is_header = first_line.startswith("#") and i + 1 < len(chunks)
            if is_header:
                # 情况1：纯标题块（只有一行）
                # 情况2：首行是标题，但内容很短（不足 chunk_size 的 1/4）
                is_pure_header = len(lines) == 1
                is_short_header = (
                    len(lines) > 1
                    and self._len(chunk.strip()) < self.chunk_size // 4
                )

                if is_pure_header or is_short_header:
                    next_chunk = chunks[i + 1]
                    if self._len(chunk) + self._len(next_chunk) + 1 <= self.chunk_size:
                        result.append(chunk + "\n" + next_chunk)
                        i += 2
                        continue

            result.append(chunk)
            i += 1

        return result
