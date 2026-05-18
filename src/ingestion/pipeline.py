"""Ingestion Pipeline 编排模块。

串行执行完整摄取流程：
integrity → load → split → transform → encode → store

每个阶段独立异常处理，失败时标记 integrity 并抛出 PipelineError。
支持 TraceContext 全链路追踪和 on_progress 进度回调。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk, ChunkRecord, Document
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.ingestion.embedding.batch_processor import BatchProcessor
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.ingestion.transform.base_transform import BaseTransform
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.ingestion.transform.image_captioner import ImageCaptioner
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.file_integrity import FileIntegrityChecker, SQLiteIntegrityChecker
from src.libs.loader.pdf_loader import PdfLoader

logger = logging.getLogger(__name__)


# ============================================================
# 异常与结果
# ============================================================


class PipelineError(Exception):
    """Pipeline 阶段失败异常。

    属性:
        stage: 失败阶段名称
        original_error: 原始异常
    """

    def __init__(self, stage: str, original_error: Exception) -> None:
        self.stage = stage
        self.original_error = original_error
        super().__init__(f"Pipeline 阶段 [{stage}] 失败: {original_error}")


@dataclass
class PipelineResult:
    """Pipeline 执行结果。

    属性:
        file_path: 源文件路径
        collection: 集合名称
        file_hash: 文件 SHA256 哈希
        doc_id: 文档 ID
        chunk_count: 产出 chunk 数量
        record_count: 写入向量库的记录数量
        skipped: 是否因增量跳过
        elapsed_ms: 总耗时（毫秒）
        stage_times: 各阶段耗时（毫秒）
    """
    file_path: str
    collection: str
    file_hash: str
    doc_id: str
    chunk_count: int
    record_count: int
    skipped: bool
    elapsed_ms: float
    stage_times: Dict[str, float] = field(default_factory=dict)


# ============================================================
# Pipeline 编排器
# ============================================================

# 进度回调签名: on_progress(stage_name, current_step, total_steps)
ProgressCallback = Callable[[str, int, int], None]

# Pipeline 总阶段数
_TOTAL_STAGES = 6


class IngestionPipeline:
    """Ingestion Pipeline 编排器。

    串行执行 6 个阶段：
    1. integrity — SHA256 增量检查
    2. load — 文件加载为 Document
    3. split — Document 切分为 Chunks
    4. transform — 去噪 + 图片描述
    5. encode — Dense + Sparse 编码
    6. store — 向量写入 + BM25 索引

    所有组件支持依赖注入（测试用），未注入时从 Settings 懒创建。
    """

    def __init__(
        self,
        settings: Settings,
        *,
        integrity_checker: Optional[FileIntegrityChecker] = None,
        loader: Optional[BaseLoader] = None,
        chunker: Optional[DocumentChunker] = None,
        refiner: Optional[BaseTransform] = None,
        captioner: Optional[BaseTransform] = None,
        enricher: Optional[BaseTransform] = None,
        batch_processor: Optional[BatchProcessor] = None,
        vector_upserter: Optional[VectorUpserter] = None,
        bm25_indexer: Optional[BM25Indexer] = None,
        compute_hash: Optional[Callable[[str], str]] = None,
    ) -> None:
        """初始化 Pipeline。

        参数:
            settings: 全局配置
            integrity_checker: 文件完整性检查器（可选，默认 SQLiteIntegrityChecker）
            loader: 文件加载器（可选，默认 PdfLoader）
            chunker: 文档切分器（可选，默认 DocumentChunker）
            refiner: Chunk 去噪器（可选，默认 ChunkRefiner 规则模式）
            captioner: 图片描述器（可选，默认 ImageCaptioner）
            enricher: 元数据增强器（可选，默认 MetadataEnricher 规则模式）
            batch_processor: 批量编码器（可选，默认 BatchProcessor）
            vector_upserter: 向量写入器（可选，默认 VectorUpserter）
            bm25_indexer: BM25 索引器（可选，默认 BM25Indexer）
            compute_hash: 文件哈希计算函数（可选，默认 FileIntegrityChecker.compute_sha256）
        """
        self._settings = settings
        self._integrity_checker = integrity_checker
        self._loader = loader
        self._chunker = chunker
        self._refiner = refiner
        self._captioner = captioner
        self._enricher = enricher
        self._batch_processor = batch_processor
        self._vector_upserter = vector_upserter
        self._bm25_indexer = bm25_indexer
        self._compute_hash = compute_hash or FileIntegrityChecker.compute_sha256

    # ----------------------------------------------------------
    # 懒创建方法
    # ----------------------------------------------------------

    def _get_integrity_checker(self) -> FileIntegrityChecker:
        """获取文件完整性检查器。"""
        if self._integrity_checker is None:
            self._integrity_checker = SQLiteIntegrityChecker()
        return self._integrity_checker

    def _get_loader(self) -> BaseLoader:
        """获取文件加载器。"""
        if self._loader is None:
            self._loader = PdfLoader()
        return self._loader

    def _get_chunker(self) -> DocumentChunker:
        """获取文档切分器。"""
        if self._chunker is None:
            self._chunker = DocumentChunker(self._settings)
        return self._chunker

    def _get_refiner(self) -> BaseTransform:
        """获取 Chunk 去噪器。"""
        if self._refiner is None:
            use_llm = getattr(self._settings.pipeline, "use_llm_refiner", False)
            self._refiner = ChunkRefiner(self._settings, use_llm=use_llm)
        return self._refiner

    def _get_captioner(self) -> BaseTransform:
        """获取图片描述器。"""
        if self._captioner is None:
            use_vision = getattr(self._settings.pipeline, "use_vision_llm", False)
            self._captioner = ImageCaptioner(self._settings, use_vision_llm=use_vision)
        return self._captioner

    def _get_enricher(self) -> BaseTransform:
        """获取元数据增强器。"""
        if self._enricher is None:
            use_llm = getattr(self._settings.pipeline, "use_llm_enricher", False)
            self._enricher = MetadataEnricher(self._settings, use_llm=use_llm)
        return self._enricher

    def _get_batch_processor(self) -> BatchProcessor:
        """获取批量编码器。"""
        if self._batch_processor is None:
            self._batch_processor = BatchProcessor(self._settings)
        return self._batch_processor

    def _get_vector_upserter(self) -> VectorUpserter:
        """获取向量写入器。"""
        if self._vector_upserter is None:
            self._vector_upserter = VectorUpserter(self._settings)
        return self._vector_upserter

    def _get_bm25_indexer(self) -> BM25Indexer:
        """获取 BM25 索引器。"""
        if self._bm25_indexer is None:
            self._bm25_indexer = BM25Indexer()
        return self._bm25_indexer

    # ----------------------------------------------------------
    # 主入口
    # ----------------------------------------------------------

    def run(
        self,
        file_path: str,
        collection: str = "default",
        force: bool = False,
        trace: Optional[TraceContext] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> PipelineResult:
        """执行完整摄取流程。

        参数:
            file_path: 源文件路径
            collection: 集合名称
            force: 是否强制重新处理（忽略增量检查）
            trace: 追踪上下文（可选）
            on_progress: 进度回调（可选）

        返回:
            PipelineResult 执行结果

        异常:
            FileNotFoundError: 文件不存在
            PipelineError: 阶段执行失败
        """
        start_time = time.time()
        stage_times: Dict[str, float] = {}

        # 阶段 1: 完整性检查
        self._report_progress(on_progress, "integrity", 1, _TOTAL_STAGES)
        file_hash = self._stage_integrity(file_path, force, trace, stage_times)

        if file_hash is None:
            # 增量跳过
            elapsed = (time.time() - start_time) * 1000
            return PipelineResult(
                file_path=file_path,
                collection=collection,
                file_hash="",
                doc_id="",
                chunk_count=0,
                record_count=0,
                skipped=True,
                elapsed_ms=elapsed,
                stage_times=stage_times,
            )

        try:
            # 阶段 2: 加载
            self._report_progress(on_progress, "load", 2, _TOTAL_STAGES)
            document = self._stage_load(file_path, collection, trace, stage_times)

            # 阶段 3: 切分
            self._report_progress(on_progress, "split", 3, _TOTAL_STAGES)
            chunks = self._stage_split(document, trace, stage_times)

            # 阶段 4: 变换
            self._report_progress(on_progress, "transform", 4, _TOTAL_STAGES)
            chunks = self._stage_transform(chunks, trace, stage_times)

            # 阶段 5: 编码
            self._report_progress(on_progress, "encode", 5, _TOTAL_STAGES)
            records = self._stage_encode(chunks, trace, stage_times)

            # 阶段 6: 存储
            self._report_progress(on_progress, "store", 6, _TOTAL_STAGES)
            stored_count = self._stage_store(records, trace, stage_times)

            # 标记成功
            integrity = self._get_integrity_checker()
            integrity.mark_success(
                file_hash=file_hash,
                file_path=file_path,
                chunk_count=len(chunks),
            )

            elapsed = (time.time() - start_time) * 1000
            logger.info(
                "Pipeline 完成: %s, %d chunks, %d 条记录, 耗时 %.1fms",
                file_path, len(chunks), stored_count, elapsed,
            )

            return PipelineResult(
                file_path=file_path,
                collection=collection,
                file_hash=file_hash,
                doc_id=document.id,
                chunk_count=len(chunks),
                record_count=stored_count,
                skipped=False,
                elapsed_ms=elapsed,
                stage_times=stage_times,
            )

        except PipelineError:
            raise
        except Exception as e:
            # 标记失败并包装为 PipelineError
            integrity = self._get_integrity_checker()
            integrity.mark_failed(file_hash, file_path, str(e))
            raise PipelineError("unknown", e) from e

    # ----------------------------------------------------------
    # 各阶段实现
    # ----------------------------------------------------------

    def _stage_integrity(
        self,
        file_path: str,
        force: bool,
        trace: Optional[TraceContext],
        stage_times: Dict[str, float],
    ) -> Optional[str]:
        """阶段 1: 完整性检查。返回 file_hash 或 None（跳过）。"""
        stage_start = time.time()
        try:
            file_hash = self._compute_hash(file_path)
            integrity = self._get_integrity_checker()

            if not force and integrity.should_skip(file_hash):
                logger.info("文件已处理，跳过: %s (hash=%s)", file_path, file_hash[:16])
                if trace:
                    elapsed = (time.time() - stage_start) * 1000
                    trace.record_stage(
                        "integrity",
                        action="skip",
                        file_hash=file_hash,
                        elapsed_ms=round(elapsed, 2),
                    )
                return None

            if trace:
                elapsed = (time.time() - stage_start) * 1000
                trace.record_stage(
                    "integrity",
                    action="process",
                    file_hash=file_hash,
                    elapsed_ms=round(elapsed, 2),
                )
            return file_hash

        except Exception as e:
            raise PipelineError("integrity", e) from e
        finally:
            stage_times["integrity"] = (time.time() - stage_start) * 1000

    def _stage_load(
        self,
        file_path: str,
        collection: str,
        trace: Optional[TraceContext],
        stage_times: Dict[str, float],
    ) -> Document:
        """阶段 2: 加载文件为 Document。"""
        stage_start = time.time()
        try:
            loader = self._get_loader()
            document = loader.load(file_path, collection=collection)
            logger.info("加载完成: %s, %d 字符", file_path, len(document.text))
            if trace:
                elapsed = (time.time() - stage_start) * 1000
                trace.record_stage(
                    "load",
                    method=type(loader).__name__,
                    text_length=len(document.text),
                    image_count=len(document.get_images()),
                    elapsed_ms=round(elapsed, 2),
                )
            return document
        except Exception as e:
            raise PipelineError("load", e) from e
        finally:
            stage_times["load"] = (time.time() - stage_start) * 1000

    def _stage_split(
        self,
        document: Document,
        trace: Optional[TraceContext],
        stage_times: Dict[str, float],
    ) -> List[Chunk]:
        """阶段 3: Document 切分为 Chunks。"""
        stage_start = time.time()
        try:
            chunker = self._get_chunker()
            chunks = chunker.split_document(document)
            logger.info("切分完成: %d chunks", len(chunks))
            if trace:
                elapsed = (time.time() - stage_start) * 1000
                trace.record_stage(
                    "split",
                    method=type(chunker).__name__,
                    chunk_count=len(chunks),
                    elapsed_ms=round(elapsed, 2),
                )
            return chunks
        except Exception as e:
            raise PipelineError("split", e) from e
        finally:
            stage_times["split"] = (time.time() - stage_start) * 1000

    def _stage_transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext],
        stage_times: Dict[str, float],
    ) -> List[Chunk]:
        """阶段 4: 变换（去噪 + 图片描述）。"""
        stage_start = time.time()
        try:
            # 4a: ChunkRefiner 去噪
            refiner = self._get_refiner()
            chunks = refiner.transform(chunks, trace=trace)
            logger.info("去噪完成: %d chunks", len(chunks))

            # 4b: ImageCaptioner 图片描述
            captioner = self._get_captioner()
            chunks = captioner.transform(chunks, trace=trace)
            logger.info("图片描述完成: %d chunks", len(chunks))

            # 4c: MetadataEnricher 元数据增强
            enricher = self._get_enricher()
            chunks = enricher.transform(chunks, trace=trace)
            logger.info("元数据增强完成: %d chunks", len(chunks))

            if trace:
                elapsed = (time.time() - stage_start) * 1000
                method = "+".join([
                    type(refiner).__name__,
                    type(captioner).__name__,
                    type(enricher).__name__,
                ])
                trace.record_stage(
                    "transform",
                    method=method,
                    chunk_count=len(chunks),
                    elapsed_ms=round(elapsed, 2),
                )
            return chunks
        except Exception as e:
            raise PipelineError("transform", e) from e
        finally:
            stage_times["transform"] = (time.time() - stage_start) * 1000

    def _stage_encode(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext],
        stage_times: Dict[str, float],
    ) -> List[ChunkRecord]:
        """阶段 5: Dense + Sparse 编码。"""
        stage_start = time.time()
        try:
            processor = self._get_batch_processor()
            records = processor.process(chunks, trace=trace)
            logger.info("编码完成: %d records", len(records))
            if trace:
                elapsed = (time.time() - stage_start) * 1000
                trace.record_stage(
                    "embed",
                    method=type(processor).__name__,
                    record_count=len(records),
                    elapsed_ms=round(elapsed, 2),
                )
            return records
        except Exception as e:
            raise PipelineError("encode", e) from e
        finally:
            stage_times["encode"] = (time.time() - stage_start) * 1000

    def _stage_store(
        self,
        records: List[ChunkRecord],
        trace: Optional[TraceContext],
        stage_times: Dict[str, float],
    ) -> int:
        """阶段 6: 向量写入 + BM25 索引。返回写入记录数。"""
        stage_start = time.time()
        try:
            # 6a: 向量写入
            upserter = self._get_vector_upserter()
            stored = upserter.upsert(records, trace=trace)
            logger.info("向量写入完成: %d 条", stored)

            # 6b: BM25 索引
            indexer = self._get_bm25_indexer()
            indexer.build(records, trace=trace)
            indexer.save()
            logger.info("BM25 索引构建完成")

            if trace:
                elapsed = (time.time() - stage_start) * 1000
                method = "+".join([
                    type(upserter).__name__,
                    type(indexer).__name__,
                ])
                trace.record_stage(
                    "upsert",
                    method=method,
                    vector_count=stored,
                    vocabulary_size=indexer.get_vocabulary_size(),
                    elapsed_ms=round(elapsed, 2),
                )
            return stored
        except Exception as e:
            raise PipelineError("store", e) from e
        finally:
            stage_times["store"] = (time.time() - stage_start) * 1000

    # ----------------------------------------------------------
    # 工具方法
    # ----------------------------------------------------------

    @staticmethod
    def _report_progress(
        on_progress: Optional[ProgressCallback],
        stage_name: str,
        current: int,
        total: int,
    ) -> None:
        """报告进度。"""
        if on_progress is not None:
            on_progress(stage_name, current, total)
