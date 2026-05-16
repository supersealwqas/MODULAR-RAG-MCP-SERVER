"""文本切分器模块。

提供可插拔的文本切分策略：
- RecursiveSplitter: 递归字符切分（默认）
"""

from src.libs.splitter.recursive_splitter import RecursiveSplitter

__all__ = ["RecursiveSplitter"]
