"""BatchProcessor 模块。

将 Chunk 列表分批，驱动 DenseEncoder 和 SparseEncoder 进行编码，
记录每批次耗时，输出完整的 ChunkRecord 列表（含 dense_vector 和 sparse_vector）。
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk, ChunkRecord
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder

logger = logging.getLogger(__name__)


class BatchProcessor:
    """批处理编排器：分批驱动 Dense/Sparse 编码。

    处理流程：
    1. 将 Chunk 列表按 batch_size 分批
    2. 对每批依次调用 SparseEncoder 和 DenseEncoder
    3. 合并两个编码器的输出（dense_vector + sparse_vector）
    4. 记录每批次和总体的 Trace 数据

    属性:
        batch_size: 每批处理的 chunk 数量
    """

    def __init__(
        self,
        settings: Settings,
        dense_encoder: Optional[DenseEncoder] = None,
        sparse_encoder: Optional[SparseEncoder] = None,
        batch_size: int = 64,
    ) -> None:
        """初始化 BatchProcessor。

        参数:
            settings: 全局配置对象
            dense_encoder: DenseEncoder 实例（可选，不传时自动创建）
            sparse_encoder: SparseEncoder 实例（可选，不传时自动创建）
            batch_size: 每批处理的 chunk 数量（默认 64）
        """
        self._settings = settings
        self._dense_encoder = dense_encoder
        self._sparse_encoder = sparse_encoder
        self.batch_size = batch_size

    def _get_dense_encoder(self) -> DenseEncoder:
        """获取 DenseEncoder 实例（延迟创建）。

        返回:
            DenseEncoder 实例
        """
        if self._dense_encoder is None:
            self._dense_encoder = DenseEncoder(
                settings=self._settings,
                batch_size=self.batch_size,
            )
        return self._dense_encoder

    def _get_sparse_encoder(self) -> SparseEncoder:
        """获取 SparseEncoder 实例（延迟创建）。

        返回:
            SparseEncoder 实例
        """
        if self._sparse_encoder is None:
            self._sparse_encoder = SparseEncoder(settings=self._settings)
        return self._sparse_encoder

    def process(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None,
    ) -> List[ChunkRecord]:
        """将 Chunk 列表分批编码，输出完整的 ChunkRecord 列表。

        参数:
            chunks: 待编码的 Chunk 列表
            trace: 可选的追踪上下文

        返回:
            ChunkRecord 列表，包含 dense_vector 和 sparse_vector
        """
        if not chunks:
            return []

        dense_encoder = self._get_dense_encoder()
        sparse_encoder = self._get_sparse_encoder()
        start_time = time.time()

        # 分批处理
        all_records: List[ChunkRecord] = []
        batch_count = 0

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            batch_start = time.time()
            batch_count += 1

            # 先做稀疏编码，再做稠密编码
            sparse_records = sparse_encoder.encode(batch)
            dense_records = dense_encoder.encode(batch)

            # 合并：以 sparse_records 为基础，补充 dense_vector
            merged = self._merge_records(sparse_records, dense_records)
            all_records.extend(merged)

            batch_elapsed = (time.time() - batch_start) * 1000

            # 记录批次 Trace
            if trace:
                trace.record_stage(
                    f"batch_{batch_count}",
                    chunk_count=len(batch),
                    batch_index=batch_count,
                    elapsed_ms=round(batch_elapsed, 2),
                )

        total_elapsed = (time.time() - start_time) * 1000

        # 记录总体 Trace
        if trace:
            trace.record_stage(
                "batch_process",
                method="dense+sparse",
                chunk_count=len(chunks),
                batch_count=batch_count,
                batch_size=self.batch_size,
                elapsed_ms=round(total_elapsed, 2),
            )

        return all_records

    def _merge_records(
        self,
        sparse_records: List[ChunkRecord],
        dense_records: List[ChunkRecord],
    ) -> List[ChunkRecord]:
        """合并稀疏和稠密编码结果。

        以 sparse_records 为基础，从 dense_records 中补充 dense_vector。

        参数:
            sparse_records: 稀疏编码结果
            dense_records: 稠密编码结果

        返回:
            合并后的 ChunkRecord 列表
        """
        # 建立 id → dense_vector 映射
        dense_map = {r.id: r.dense_vector for r in dense_records}

        for record in sparse_records:
            record.dense_vector = dense_map.get(record.id)

        return sparse_records

    def split_into_batches(self, chunks: List[Chunk]) -> List[List[Chunk]]:
        """将 Chunk 列表按 batch_size 分批（辅助方法）。

        参数:
            chunks: Chunk 列表

        返回:
            分批后的二维列表
        """
        batches = []
        for i in range(0, len(chunks), self.batch_size):
            batches.append(chunks[i : i + self.batch_size])
        return batches
