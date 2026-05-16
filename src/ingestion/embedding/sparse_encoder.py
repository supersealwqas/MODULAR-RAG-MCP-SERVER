"""SparseEncoder 模块。

对 Chunk 文本进行分词和词频统计，生成稀疏向量（term weights），
填充 ChunkRecord.sparse_vector，供下游 BM25Indexer 使用。
支持两种后端：
- jieba：本地分词 + TF 计算（轻量，无需 GPU）
- bge：BGE-M3 模型的 lexical_weights（质量更高，需要模型文件）
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from typing import Callable, Dict, List, Optional, Set

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk, ChunkRecord

logger = logging.getLogger(__name__)

# 默认停用词集合（中英文混合）
_DEFAULT_STOPWORDS: Set[str] = {
    # 英文停用词
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "to",
    "for", "with", "on", "at", "from", "by", "and", "or", "not", "it",
    "this", "that", "as", "but", "if", "then", "so", "no", "than",
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


class SparseEncoder:
    """稀疏向量编码器：对 Chunk 文本分词并计算 term weights。

    支持两种后端：
    - jieba：jieba 分词 + 对数归一化 TF（轻量，无需 GPU）
    - bge：BGE-M3 模型的 lexical_weights（质量更高，需要模型文件）

    属性:
        backend: 稀疏向量后端（"jieba" 或 "bge"）
        min_token_length: 最小 token 长度（低于此长度的 token 被过滤）
        stopwords: 停用词集合
    """

    def __init__(
        self,
        settings: Settings,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        stopwords: Optional[Set[str]] = None,
        min_token_length: int = 1,
        backend: Optional[str] = None,
    ) -> None:
        """初始化 SparseEncoder。

        参数:
            settings: 全局配置对象
            tokenizer: 自定义分词函数（可选，默认使用 jieba 或正则）
            stopwords: 停用词集合（可选，默认使用内置停用词）
            min_token_length: 最小 token 长度（默认 1）
            backend: 稀疏向量后端（"jieba" 或 "bge"，默认从配置读取）
        """
        self._settings = settings
        self._tokenizer = tokenizer
        self.stopwords = stopwords if stopwords is not None else _DEFAULT_STOPWORDS
        self.min_token_length = min_token_length
        self.backend = backend or getattr(settings.embedding, "sparse_backend", "jieba")

        # jieba 后端状态
        self._jieba = None
        self._jieba_tried = False

        # BGE 后端状态
        self._bge_model = None
        self._bge_tokenizer = None
        self._bge_tried = False

    # ============================================================
    # jieba 后端
    # ============================================================

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
            logger.info("使用 jieba 分词器")
        except ImportError:
            logger.info("jieba 未安装，使用正则分词器")
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
        """对文本进行分词（jieba 后端）。

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
                and len(token_lower) >= self.min_token_length
                and token_lower not in self.stopwords
            ):
                filtered.append(token_lower)
        return filtered

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """计算归一化词频（TF）。

        使用对数归一化: TF(t) = 1 + log(count(t))，若 count > 0，否则 0。
        这是 BM25 变体中常用的 TF 归一化方式。

        参数:
            tokens: token 列表

        返回:
            term → weight 字典
        """
        if not tokens:
            return {}

        counter = Counter(tokens)
        tf: Dict[str, float] = {}
        for term, count in counter.items():
            # 对数归一化 TF
            tf[term] = 1.0 + (0.0 if count <= 0 else math.log(count))
        return tf

    def _encode_with_jieba(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None,
    ) -> List[ChunkRecord]:
        """使用 jieba 后端编码 Chunk 列表。

        参数:
            chunks: 待编码的 Chunk 列表
            trace: 可选的追踪上下文

        返回:
            ChunkRecord 列表，每个记录包含 sparse_vector
        """
        start_time = time.time()
        records: List[ChunkRecord] = []

        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            tf = self._compute_tf(tokens)
            record = self._chunk_to_record(chunk)
            record.sparse_vector = tf
            records.append(record)

        elapsed_ms = (time.time() - start_time) * 1000

        if trace:
            total_terms = sum(len(r.sparse_vector) for r in records if r.sparse_vector)
            trace.record_stage(
                "sparse_encode",
                method="jieba",
                chunk_count=len(chunks),
                total_terms=total_terms,
                elapsed_ms=round(elapsed_ms, 2),
            )

        return records

    # ============================================================
    # BGE-M3 后端
    # ============================================================

    def _try_load_bge(self) -> bool:
        """尝试加载 BGE-M3 模型（延迟加载）。

        返回:
            是否加载成功
        """
        if self._bge_tried:
            return self._bge_model is not None
        self._bge_tried = True

        try:
            from src.libs.embedding.bge_embedding import BGEEmbedding

            self._bge_model = BGEEmbedding(
                model=self._settings.embedding.model,
                model_path=self._settings.embedding.model_path,
                use_fp16=True,
            )
            # 触发模型加载以获取 tokenizer
            self._bge_model._load_model()
            self._bge_tokenizer = self._bge_model._embedding_model.model.tokenizer
            logger.info("使用 BGE-M3 稀疏编码后端")
            return True
        except Exception as e:
            logger.warning("BGE-M3 模型加载失败，降级到 jieba: %s", e)
            self._bge_model = None
            return False

    def _token_ids_to_tokens(self, token_weights: Dict) -> Dict[str, float]:
        """将 BGE-M3 的 lexical_weights 转换为 token_string 权重。

        BGE-M3 的 lexical_weights 返回 {str(token_id): weight}，
        需要将 token_id decode 为实际的 token 字符串。

        参数:
            token_weights: {token_id_str: weight} 字典

        返回:
            {token_string: weight} 字典，过滤停用词和短词
        """
        result: Dict[str, float] = {}
        for token_id, weight in token_weights.items():
            # 跳过零权重
            if weight <= 0:
                continue

            # 将 token_id 转为 int 后 decode 为 token 字符串
            try:
                tid = int(token_id)
                token_str = self._bge_tokenizer.decode([tid]).strip().lower()
            except (ValueError, TypeError):
                # 无法转换为 int，直接使用原始字符串
                token_str = str(token_id).strip().lower()

            # 过滤：空串、短词、停用词、特殊 token（如 [CLS], [SEP], <s> 等）
            if (
                not token_str
                or len(token_str) < self.min_token_length
                or token_str in self.stopwords
                or token_str.startswith("[") and token_str.endswith("]")
                or token_str.startswith("<") and token_str.endswith(">")
            ):
                continue

            # 同一 token 可能出现多次（不同 token_id 映射到同一字符串），取最大权重
            if token_str not in result or weight > result[token_str]:
                result[token_str] = weight

        return result

    def _encode_with_bge(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None,
    ) -> List[ChunkRecord]:
        """使用 BGE-M3 后端编码 Chunk 列表。

        调用 BGE-M3 的 embed_with_sparse() 获取 lexical_weights，
        然后将 token_id 转换为 token_string，保持 Dict[str, float] 格式。

        参数:
            chunks: 待编码的 Chunk 列表
            trace: 可选的追踪上下文

        返回:
            ChunkRecord 列表，每个记录包含 sparse_vector
        """
        if not self._try_load_bge():
            # BGE 加载失败，降级到 jieba
            logger.info("BGE 不可用，降级到 jieba 后端")
            return self._encode_with_jieba(chunks, trace)

        start_time = time.time()
        texts = [chunk.text for chunk in chunks]

        # 调用 BGE-M3 获取 dense + sparse
        _, sparse_vecs = self._bge_model.embed_with_sparse(texts)

        records: List[ChunkRecord] = []
        for chunk, sparse_raw in zip(chunks, sparse_vecs):
            # 将 {token_id: weight} 转为 {token_string: weight}
            sparse_str = self._token_ids_to_tokens(sparse_raw)
            record = self._chunk_to_record(chunk)
            record.sparse_vector = sparse_str
            records.append(record)

        elapsed_ms = (time.time() - start_time) * 1000

        if trace:
            total_terms = sum(len(r.sparse_vector) for r in records if r.sparse_vector)
            trace.record_stage(
                "sparse_encode",
                method="bge",
                chunk_count=len(chunks),
                total_terms=total_terms,
                elapsed_ms=round(elapsed_ms, 2),
            )

        return records

    # ============================================================
    # 公共接口
    # ============================================================

    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None,
    ) -> List[ChunkRecord]:
        """将 Chunk 列表编码为 ChunkRecord 列表（填充 sparse_vector）。

        根据 self.backend 分派到对应的编码方法：
        - "jieba"：jieba 分词 + TF 计算
        - "bge"：BGE-M3 lexical_weights

        参数:
            chunks: 待编码的 Chunk 列表
            trace: 可选的追踪上下文

        返回:
            ChunkRecord 列表，每个记录包含 sparse_vector
        """
        if not chunks:
            return []

        if self.backend == "bge":
            return self._encode_with_bge(chunks, trace)
        else:
            return self._encode_with_jieba(chunks, trace)

    def _chunk_to_record(self, chunk: Chunk) -> ChunkRecord:
        """将 Chunk 转换为 ChunkRecord（不填充向量）。

        参数:
            chunk: 源 Chunk 对象

        返回:
            ChunkRecord 对象，sparse_vector 为 None
        """
        return ChunkRecord(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata.copy(),
        )
