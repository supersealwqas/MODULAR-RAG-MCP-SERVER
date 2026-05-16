"""DenseEncoder 模块。

将 Chunk 文本批量转换为稠密向量，填充 ChunkRecord.dense_vector。
依赖 libs.embedding 的 BaseEmbedding 抽象接口，通过 EmbeddingFactory 创建实例。
支持批量处理、Trace 记录和错误降级。
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk, ChunkRecord
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory

logger = logging.getLogger(__name__)


class DenseEncoder:
    """稠密向量编码器：将 Chunk 文本批量送入 Embedding 模型生成向量。

    处理流程：
    1. 从 Chunk 列表中提取文本
    2. 调用 BaseEmbedding.embed() 批量生成向量
    3. 将向量填充到 ChunkRecord.dense_vector
    4. 记录 Trace 阶段数据

    属性:
        settings: 全局配置对象
        batch_size: 每批处理的 chunk 数量
    """

    def __init__(
        self,
        settings: Settings,
        embedding: Optional[BaseEmbedding] = None,
        batch_size: int = 64,
    ) -> None:
        """初始化 DenseEncoder。

        参数:
            settings: 全局配置对象
            embedding: Embedding 实例（可选，不传时根据 settings 自动创建）
            batch_size: 每批处理的 chunk 数量（默认 64）
        """
        self._settings = settings
        self._embedding = embedding
        self.batch_size = batch_size

    def _get_embedding(self) -> Optional[BaseEmbedding]:
        """获取 Embedding 实例（延迟创建）。

        返回:
            BaseEmbedding 实例，创建失败时返回 None
        """
        if self._embedding is None:
            try:
                self._embedding = EmbeddingFactory.create(self._settings.embedding)
            except Exception as e:
                logger.warning("Embedding 创建失败: %s", e)
                return None
        return self._embedding

    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None,
    ) -> List[ChunkRecord]:
        """将 Chunk 列表编码为 ChunkRecord 列表（填充 dense_vector）。

        参数:
            chunks: 待编码的 Chunk 列表
            trace: 可选的追踪上下文

        返回:
            ChunkRecord 列表，每个记录包含 dense_vector
        """
        if not chunks:
            return []

        embedding = self._get_embedding()
        if embedding is None:
            logger.error("Embedding 不可用，无法生成稠密向量")
            return [self._chunk_to_record(chunk) for chunk in chunks]

        # 提取文本列表
        texts = [chunk.text for chunk in chunks]

        # 分批编码
        all_vectors: List[List[float]] = []
        start_time = time.time()

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            try:
                batch_vectors = embedding.embed(batch_texts)
                all_vectors.extend(batch_vectors)
            except Exception as e:
                logger.warning("批次 %d 编码失败: %s", i // self.batch_size, e)
                # 用零向量填充失败批次
                dim = embedding.dimensions
                all_vectors.extend([[0.0] * dim for _ in batch_texts])

        elapsed_ms = (time.time() - start_time) * 1000

        # 组装 ChunkRecord
        records: List[ChunkRecord] = []
        for chunk, vector in zip(chunks, all_vectors):
            record = self._chunk_to_record(chunk)
            record.dense_vector = vector
            records.append(record)

        # 记录 Trace
        if trace:
            trace.record_stage(
                "dense_encode",
                method="embedding",
                chunk_count=len(chunks),
                vector_dim=embedding.dimensions,
                elapsed_ms=round(elapsed_ms, 2),
            )

        return records

    def _chunk_to_record(self, chunk: Chunk) -> ChunkRecord:
        """将 Chunk 转换为 ChunkRecord（不填充向量）。

        参数:
            chunk: 源 Chunk 对象

        返回:
            ChunkRecord 对象，dense_vector 为 None
        """
        return ChunkRecord(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata.copy(),
        )
