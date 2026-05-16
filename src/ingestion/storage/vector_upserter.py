"""VectorUpserter 模块。

接收 BatchProcessor 输出的 ChunkRecord 列表（含 dense_vector），
生成稳定的 chunk_id，并调用 VectorStore 进行幂等写入。
保证同一内容重复写入不产生重复记录。
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import ChunkRecord
from src.libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

logger = logging.getLogger(__name__)


class VectorUpserter:
    """向量存储写入器：将 ChunkRecord 写入 VectorStore。

    核心职责：
    1. 生成确定性 chunk_id（基于 source_path + chunk_index + content_hash）
    2. 将 ChunkRecord 转换为 VectorRecord
    3. 调用 VectorStore.upsert() 进行幂等写入
    4. 记录 Trace 阶段数据

    属性:
        settings: 全局配置对象
    """

    def __init__(
        self,
        settings: Settings,
        vector_store: Optional[BaseVectorStore] = None,
    ) -> None:
        """初始化 VectorUpserter。

        参数:
            settings: 全局配置对象
            vector_store: VectorStore 实例（可选，不传时根据 settings 自动创建）
        """
        self._settings = settings
        self._vector_store = vector_store

    def _get_vector_store(self) -> BaseVectorStore:
        """获取 VectorStore 实例（延迟创建）。

        返回:
            BaseVectorStore 实例
        """
        if self._vector_store is None:
            self._vector_store = VectorStoreFactory.create(self._settings.vector_store)
        return self._vector_store

    @staticmethod
    def generate_stable_id(
        source_path: str,
        chunk_index: int,
        content_hash: str,
    ) -> str:
        """生成确定性 chunk_id。

        公式: sha256(source_path + chunk_index + content_hash[:8])[:16]

        参数:
            source_path: 文档来源路径
            chunk_index: chunk 在文档中的序号
            content_hash: 内容的 SHA256 哈希值

        返回:
            16 字符的确定性 ID
        """
        raw = f"{source_path}:{chunk_index}:{content_hash[:8]}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _compute_content_hash(text: str) -> str:
        """计算文本内容的 SHA256 哈希。

        参数:
            text: 文本内容

        返回:
            SHA256 哈希字符串
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _record_to_vector_record(self, record: ChunkRecord) -> VectorRecord:
        """将 ChunkRecord 转换为 VectorRecord。

        使用 ChunkRecord.id 作为 VectorRecord.id（由 DocumentChunker 生成的确定性 ID）。
        如果 ChunkRecord 没有 dense_vector，使用零向量。

        参数:
            record: 源 ChunkRecord

        返回:
            VectorRecord 对象
        """
        vector = record.dense_vector or []
        return VectorRecord(
            id=record.id,
            vector=vector,
            text=record.text,
            metadata=record.metadata.copy(),
        )

    def upsert(
        self,
        records: List[ChunkRecord],
        trace: Optional[TraceContext] = None,
    ) -> int:
        """将 ChunkRecord 列表写入 VectorStore。

        幂等性保证：同一 ChunkRecord.id 的重复写入会被 VectorStore 的
        upsert 语义覆盖，不会产生重复记录。

        参数:
            records: 含 dense_vector 的 ChunkRecord 列表
            trace: 可选的追踪上下文

        返回:
            成功写入的记录数
        """
        if not records:
            logger.warning("空记录列表，跳过 upsert")
            return 0

        # 过滤没有 dense_vector 的记录
        valid_records = [r for r in records if r.dense_vector is not None]
        if not valid_records:
            logger.warning("所有记录均无 dense_vector，跳过 upsert")
            return 0

        skipped = len(records) - len(valid_records)
        if skipped > 0:
            logger.info("跳过 %d 条无向量记录", skipped)

        start_time = time.time()

        # 转换为 VectorRecord
        vector_records = [self._record_to_vector_record(r) for r in valid_records]

        # 调用 VectorStore upsert
        store = self._get_vector_store()
        upserted = store.upsert(vector_records)

        elapsed_ms = (time.time() - start_time) * 1000

        # 记录 Trace
        if trace:
            trace.record_stage(
                "vector_upsert",
                method=self._settings.vector_store.provider,
                total_records=len(records),
                valid_records=len(valid_records),
                skipped_records=skipped,
                upserted_count=upserted,
                elapsed_ms=round(elapsed_ms, 2),
            )

        logger.info(
            "VectorStore upsert 完成: %d/%d 条记录写入（耗时 %.1fms）",
            upserted,
            len(records),
            elapsed_ms,
        )

        return upserted

    def delete(
        self,
        chunk_ids: List[str],
        trace: Optional[TraceContext] = None,
    ) -> int:
        """从 VectorStore 中删除指定 chunk。

        参数:
            chunk_ids: 待删除的 chunk ID 列表
            trace: 可选的追踪上下文

        返回:
            成功删除的记录数
        """
        if not chunk_ids:
            return 0

        start_time = time.time()
        store = self._get_vector_store()

        try:
            deleted = store.delete(chunk_ids)
        except NotImplementedError:
            logger.warning("VectorStore 不支持 delete 操作")
            return 0

        elapsed_ms = (time.time() - start_time) * 1000

        if trace:
            trace.record_stage(
                "vector_delete",
                requested_count=len(chunk_ids),
                deleted_count=deleted,
                elapsed_ms=round(elapsed_ms, 2),
            )

        logger.info("VectorStore 删除完成: %d 条记录", deleted)
        return deleted
