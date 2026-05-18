"""QueryProcessor 模块。

对用户查询进行预处理：关键词提取（jieba/正则分词 + 停用词过滤），
解析通用 filters 结构。输出 ProcessedQuery 供下游 Dense/Sparse 检索使用。
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import ProcessedQuery

logger = logging.getLogger(__name__)

# 默认停用词集合（中英文混合，与 SparseEncoder 保持一致）
_DEFAULT_STOPWORDS: Set[str] = {
    # 英文停用词
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "to",
    "for", "with", "on", "at", "from", "by", "and", "or", "not", "it",
    "this", "that", "as", "but", "if", "then", "so", "no", "than",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    # 中文停用词
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
    "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
    "会", "着", "没有", "看", "好", "自己", "这", "他", "她", "它",
    "们", "那", "些", "什么", "怎么", "如何", "可以", "能", "对",
    "与", "及", "等", "之", "其", "中", "将", "把", "被", "让",
    "给", "向", "从", "以", "用", "为", "所", "而", "但", "却",
    "又", "只", "已", "正", "则", "并", "或", "如", "更", "再",
    "最", "非", "无", "未", "每", "各", "某", "此", "这些", "那些",
}


class QueryProcessor:
    """查询预处理器：提取关键词、解析过滤条件。

    属性:
        stopwords: 停用词集合
        min_keyword_length: 最小关键词长度
    """

    def __init__(
        self,
        settings: Settings,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        stopwords: Optional[Set[str]] = None,
        min_keyword_length: int = 2,
    ) -> None:
        """初始化 QueryProcessor。

        参数:
            settings: 全局配置对象
            tokenizer: 自定义分词函数（可选，默认使用 jieba 或正则）
            stopwords: 停用词集合（可选，默认使用内置停用词）
            min_keyword_length: 最小关键词长度（默认 2）
        """
        self._settings = settings
        self._tokenizer = tokenizer
        self.stopwords = stopwords if stopwords is not None else _DEFAULT_STOPWORDS
        self.min_keyword_length = min_keyword_length
        self._jieba = None
        self._jieba_tried = False

    def _try_load_jieba(self) -> Optional[object]:
        """尝试加载 jieba 分词库（延迟加载）。

        返回:
            jieba 模块对象，加载失败返回 None
        """
        if self._jieba_tried:
            return self._jieba
        self._jieba_tried = True
        try:
            import jieba
            self._jieba = jieba
            logger.info("QueryProcessor 使用 jieba 分词器")
        except ImportError:
            logger.info("jieba 未安装，QueryProcessor 使用正则分词器")
            self._jieba = None
        return self._jieba

    def _default_tokenizer(self, text: str) -> List[str]:
        """默认分词器：jieba 优先，降级到正则。

        参数:
            text: 待分词的文本

        返回:
            token 列表
        """
        jieba = self._try_load_jieba()
        if jieba is not None:
            return list(jieba.cut(text))

        # 降级：正则分词（中文逐字 + 英文单词 + 数字）
        tokens = re.findall(r'[一-鿿]|[a-zA-Z]+|\d+', text)
        return tokens

    def _tokenize(self, text: str) -> List[str]:
        """对文本进行分词并过滤停用词。

        参数:
            text: 待分词的文本

        返回:
            过滤后的 token 列表
        """
        if self._tokenizer is not None:
            raw_tokens = self._tokenizer(text)
        else:
            raw_tokens = self._default_tokenizer(text)

        # 过滤停用词和短词，统一小写
        filtered = []
        for token in raw_tokens:
            token_lower = token.lower().strip()
            if (
                token_lower
                and len(token_lower) >= self.min_keyword_length
                and token_lower not in self.stopwords
            ):
                filtered.append(token_lower)
        return filtered

    def extract_keywords(
        self,
        query: str,
        trace: Optional[TraceContext] = None,
    ) -> List[str]:
        """从查询文本中提取关键词。

        参数:
            query: 用户查询文本
            trace: 可选的追踪上下文

        返回:
            去重后的关键词列表（保持首次出现顺序）
        """
        if not query or not query.strip():
            return []

        start_time = time.time()
        tokens = self._tokenize(query)

        # 去重，保持顺序
        seen: Set[str] = set()
        keywords: List[str] = []
        for token in tokens:
            if token not in seen:
                seen.add(token)
                keywords.append(token)

        if trace:
            elapsed = (time.time() - start_time) * 1000
            trace.record_stage(
                "query_processing",
                method="jieba" if self._jieba else "regex",
                original_query=query,
                keyword_count=len(keywords),
                elapsed_ms=round(elapsed, 2),
            )

        return keywords

    def parse_filters(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """解析并规范化过滤条件。

        参数:
            filters: 原始过滤条件字典（可为 None 或空 dict）

        返回:
            规范化后的过滤条件字典
        """
        if not filters:
            return {}

        # 规范化：移除 None 值和空字符串
        parsed: Dict[str, Any] = {}
        for key, value in filters.items():
            if value is not None and value != "":
                parsed[key] = value
        return parsed

    def process(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[TraceContext] = None,
    ) -> ProcessedQuery:
        """处理查询，输出 ProcessedQuery。

        参数:
            query: 用户查询文本
            filters: 可选的过滤条件
            trace: 可选的追踪上下文

        返回:
            ProcessedQuery 对象，包含 original、keywords、filters
        """
        keywords = self.extract_keywords(query, trace)
        parsed_filters = self.parse_filters(filters)

        return ProcessedQuery(
            original=query,
            keywords=keywords,
            filters=parsed_filters,
        )
