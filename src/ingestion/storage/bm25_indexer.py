"""BM25Indexer 模块。

接收 SparseEncoder 的 term weights 输出，计算 IDF，构建倒排索引，
并持久化到文件系统（pickle 格式）。
支持索引构建、加载、查询（返回 top-k chunk_ids）和增量更新。
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.core.trace.trace_context import TraceContext
from src.core.types import ChunkRecord

logger = logging.getLogger(__name__)

# 默认索引持久化目录
_DEFAULT_INDEX_DIR = os.path.join("data", "db", "bm25")

# 索引文件名
_INDEX_FILE = "bm25_index.pkl"


class BM25Indexer:
    """BM25 倒排索引构建与查询。

    数据结构：
    - inverted_index: `{term: {idf, postings: [{chunk_id, tf, doc_length}]}}`
    - doc_lengths: `{chunk_id: doc_length}`（文档长度，用于查询时归一化）
    - corpus_size: 语料库文档总数

    属性:
        index_dir: 索引持久化目录
        k1: BM25 参数 k1（控制 TF 饱和度，默认 1.5）
        b: BM25 参数 b（控制文档长度归一化，默认 0.75）
    """

    def __init__(
        self,
        index_dir: str = _DEFAULT_INDEX_DIR,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """初始化 BM25Indexer。

        参数:
            index_dir: 索引持久化目录
            k1: BM25 参数 k1（默认 1.5）
            b: BM25 参数 b（默认 0.75）
        """
        self.index_dir = index_dir
        self.k1 = k1
        self.b = b
        self.inverted_index: Dict[str, Dict] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.corpus_size: int = 0
        self.avg_doc_length: float = 0.0

    def build(
        self,
        records: List[ChunkRecord],
        trace: Optional[TraceContext] = None,
    ) -> None:
        """从 ChunkRecord 列表构建倒排索引。

        参数:
            records: 含 sparse_vector 的 ChunkRecord 列表
            trace: 可选的追踪上下文
        """
        if not records:
            logger.warning("空记录列表，跳过索引构建")
            return

        start_time = time.time()

        # 清空旧索引
        self.inverted_index = {}
        self.doc_lengths = {}
        self.corpus_size = len(records)

        # 第一遍：收集文档长度和 term 文档频率
        term_doc_freq: Dict[str, int] = {}
        total_length = 0

        for record in records:
            sv = record.sparse_vector or {}
            doc_length = sum(sv.values())
            self.doc_lengths[record.id] = int(doc_length) if doc_length > 0 else len(sv)
            total_length += self.doc_lengths[record.id]

            # 统计每个 term 出现在多少文档中
            for term in sv:
                term_doc_freq[term] = term_doc_freq.get(term, 0) + 1

        self.avg_doc_length = total_length / self.corpus_size if self.corpus_size > 0 else 0

        # 第二遍：计算 IDF 并构建倒排索引
        for record in records:
            sv = record.sparse_vector or {}
            doc_length = self.doc_lengths[record.id]

            for term, tf in sv.items():
                df = term_doc_freq.get(term, 0)
                idf = self._compute_idf(df)

                if term not in self.inverted_index:
                    self.inverted_index[term] = {
                        "idf": idf,
                        "postings": [],
                    }

                self.inverted_index[term]["postings"].append({
                    "chunk_id": record.id,
                    "tf": tf,
                    "doc_length": doc_length,
                })

        elapsed_ms = (time.time() - start_time) * 1000

        # 记录 Trace
        if trace:
            trace.record_stage(
                "bm25_build",
                corpus_size=self.corpus_size,
                vocabulary_size=len(self.inverted_index),
                avg_doc_length=round(self.avg_doc_length, 2),
                elapsed_ms=round(elapsed_ms, 2),
            )

        logger.info(
            "BM25 索引构建完成: %d 文档, %d 词条, 平均文档长度 %.1f",
            self.corpus_size,
            len(self.inverted_index),
            self.avg_doc_length,
        )

    def _compute_idf(self, df: int) -> float:
        """计算 IDF (Inverse Document Frequency)。

        公式: IDF(term) = log((N - df + 0.5) / (df + 0.5))

        参数:
            df: 包含该 term 的文档数

        返回:
            IDF 值
        """
        n = self.corpus_size
        return math.log((n - df + 0.5) / (df + 0.5) + 1e-10)

    def query(
        self,
        terms: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """查询 BM25 索引，返回 top-k (chunk_id, score) 列表。

        BM25 评分公式:
        score(q, d) = Σ IDF(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |d| / avgdl))

        参数:
            terms: 查询 term 列表
            top_k: 返回结果数量

        返回:
            [(chunk_id, score), ...] 按 score 降序排列
        """
        if not terms or not self.inverted_index:
            return []

        # 累加每个 chunk 的 BM25 得分
        scores: Dict[str, float] = {}

        for term in terms:
            if term not in self.inverted_index:
                continue

            idf = self.inverted_index[term]["idf"]
            postings = self.inverted_index[term]["postings"]

            for posting in postings:
                chunk_id = posting["chunk_id"]
                tf = posting["tf"]
                doc_length = posting["doc_length"]

                # BM25 公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1)
                )
                score = idf * numerator / denominator

                scores[chunk_id] = scores.get(chunk_id, 0.0) + score

        # 排序并返回 top-k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def save(self, path: Optional[str] = None) -> str:
        """持久化索引到文件。

        参数:
            path: 保存路径（可选，默认 index_dir/bm25_index.pkl）

        返回:
            实际保存路径
        """
        save_path = path or os.path.join(self.index_dir, _INDEX_FILE)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "inverted_index": self.inverted_index,
            "doc_lengths": self.doc_lengths,
            "corpus_size": self.corpus_size,
            "avg_doc_length": self.avg_doc_length,
            "k1": self.k1,
            "b": self.b,
        }

        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        logger.info("BM25 索引已保存: %s", save_path)
        return save_path

    def load(self, path: Optional[str] = None) -> None:
        """从文件加载索引。

        参数:
            path: 加载路径（可选，默认 index_dir/bm25_index.pkl）

        异常:
            FileNotFoundError: 索引文件不存在
        """
        load_path = path or os.path.join(self.index_dir, _INDEX_FILE)

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"BM25 索引文件不存在: {load_path}")

        with open(load_path, "rb") as f:
            data = pickle.load(f)

        self.inverted_index = data["inverted_index"]
        self.doc_lengths = data["doc_lengths"]
        self.corpus_size = data["corpus_size"]
        self.avg_doc_length = data["avg_doc_length"]
        self.k1 = data.get("k1", 1.5)
        self.b = data.get("b", 0.75)

        logger.info(
            "BM25 索引已加载: %d 文档, %d 词条",
            self.corpus_size,
            len(self.inverted_index),
        )

    def add_documents(
        self,
        records: List[ChunkRecord],
        trace: Optional[TraceContext] = None,
    ) -> None:
        """增量添加文档到已有索引。

        参数:
            records: 含 sparse_vector 的 ChunkRecord 列表
            trace: 可选的追踪上下文
        """
        if not records:
            return

        start_time = time.time()
        added_count = 0

        for record in records:
            # 跳过已存在的文档
            if record.id in self.doc_lengths:
                continue

            sv = record.sparse_vector or {}
            doc_length = sum(sv.values())
            self.doc_lengths[record.id] = int(doc_length) if doc_length > 0 else len(sv)
            self.corpus_size += 1
            added_count += 1

            for term, tf in sv.items():
                if term not in self.inverted_index:
                    # 新 term：需要重新计算所有 term 的 IDF
                    self.inverted_index[term] = {
                        "idf": 0.0,  # 稍后更新
                        "postings": [],
                    }

                self.inverted_index[term]["postings"].append({
                    "chunk_id": record.id,
                    "tf": tf,
                    "doc_length": self.doc_lengths[record.id],
                })

        # 更新平均文档长度
        total_length = sum(self.doc_lengths.values())
        self.avg_doc_length = total_length / self.corpus_size if self.corpus_size > 0 else 0

        # 重新计算所有 IDF（因为 N 和 df 都可能变化）
        self._recompute_idf()

        elapsed_ms = (time.time() - start_time) * 1000

        if trace:
            trace.record_stage(
                "bm25_incremental",
                added_count=added_count,
                corpus_size=self.corpus_size,
                vocabulary_size=len(self.inverted_index),
                elapsed_ms=round(elapsed_ms, 2),
            )

        logger.info("BM25 索引增量更新: 新增 %d 文档, 总计 %d", added_count, self.corpus_size)

    def _recompute_idf(self) -> None:
        """重新计算所有 term 的 IDF。"""
        for term, entry in self.inverted_index.items():
            df = len(entry["postings"])
            entry["idf"] = self._compute_idf(df)

    def remove_document(self, chunk_id: str) -> bool:
        """从索引中移除一个文档。

        参数:
            chunk_id: 要移除的文档 ID

        返回:
            是否成功移除
        """
        if chunk_id not in self.doc_lengths:
            return False

        # 从 postings 中移除
        terms_to_remove = []
        for term, entry in self.inverted_index.items():
            entry["postings"] = [
                p for p in entry["postings"] if p["chunk_id"] != chunk_id
            ]
            if not entry["postings"]:
                terms_to_remove.append(term)

        # 清理空 term
        for term in terms_to_remove:
            del self.inverted_index[term]

        # 更新元数据
        del self.doc_lengths[chunk_id]
        self.corpus_size -= 1
        total_length = sum(self.doc_lengths.values())
        self.avg_doc_length = total_length / self.corpus_size if self.corpus_size > 0 else 0

        # 重新计算 IDF
        self._recompute_idf()

        return True

    def get_vocabulary_size(self) -> int:
        """获取词汇表大小。

        返回:
            索引中的唯一 term 数量
        """
        return len(self.inverted_index)
