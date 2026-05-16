"""C11 BM25Indexer 单元测试。

覆盖索引构建、IDF 计算、查询排序、持久化 roundtrip、增量更新、
文档移除、空输入处理等。
验收标准：build 后能 load 并返回稳定 top ids；IDF 计算准确。
"""

from __future__ import annotations

import math
import os
import tempfile

import pytest

from src.core.types import ChunkRecord
from src.ingestion.storage.bm25_indexer import BM25Indexer


# ============================================================
# 测试辅助函数
# ============================================================


def _make_record(
    chunk_id: str,
    sparse_vector: dict,
    text: str = "test text",
) -> ChunkRecord:
    """创建含 sparse_vector 的 ChunkRecord。"""
    return ChunkRecord(
        id=chunk_id,
        text=text,
        metadata={"source_path": "/test.pdf"},
        sparse_vector=sparse_vector,
    )


def _make_corpus() -> list:
    """创建测试语料：5 个文档，确保查询 term 的 IDF 为正。"""
    return [
        _make_record("doc1", {"rag": 2.0, "检索": 2.0, "生成": 1.0}),
        _make_record("doc2", {"rag": 2.0, "向量": 1.0, "数据库": 1.0}),
        _make_record("doc3", {"bm25": 1.0, "检索": 1.0, "算法": 1.0}),
        _make_record("doc4", {"transformer": 1.0, "注意力": 1.0, "模型": 1.0}),
        _make_record("doc5", {"向量": 1.0, "数据库": 1.0, "索引": 1.0}),
    ]


# ============================================================
# 测试：索引构建
# ============================================================

class TestBuild:
    """索引构建测试。"""

    def test_build_populates_inverted_index(self):
        """build 后 inverted_index 非空。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())

        assert len(indexer.inverted_index) > 0

    def test_build_correct_corpus_size(self):
        """build 后 corpus_size 正确。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())

        assert indexer.corpus_size == 5

    def test_build_correct_vocabulary(self):
        """build 后词汇表包含所有唯一 term。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())

        terms = set(indexer.inverted_index.keys())
        assert "rag" in terms
        assert "检索" in terms
        assert "bm25" in terms
        assert "向量" in terms

    def test_build_correct_postings_count(self):
        """每个 term 的 postings 数量正确。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())

        # "rag" 出现在 doc1 和 doc2
        assert len(indexer.inverted_index["rag"]["postings"]) == 2
        # "检索" 出现在 doc1 和 doc3
        assert len(indexer.inverted_index["检索"]["postings"]) == 2
        # "bm25" 只出现在 doc3
        assert len(indexer.inverted_index["bm25"]["postings"]) == 1

    def test_build_empty_records_no_error(self):
        """空记录列表不报错。"""
        indexer = BM25Indexer()
        indexer.build([])

        assert indexer.corpus_size == 0
        assert len(indexer.inverted_index) == 0

    def test_build_clears_old_index(self):
        """重新 build 清空旧索引。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())
        old_size = len(indexer.inverted_index)

        # 用不同语料重新构建
        new_corpus = [_make_record("new1", {"new_term": 1.0})]
        indexer.build(new_corpus)

        assert len(indexer.inverted_index) != old_size
        assert "new_term" in indexer.inverted_index


# ============================================================
# 测试：IDF 计算
# ============================================================

class TestIDF:
    """IDF 计算测试。"""

    def test_idf_formula_correct(self):
        """IDF 公式: log((N - df + 0.5) / (df + 0.5) + 1e-10)。"""
        indexer = BM25Indexer()
        indexer.corpus_size = 3
        # df=2 的 term
        idf = indexer._compute_idf(2)
        expected = math.log((3 - 2 + 0.5) / (2 + 0.5) + 1e-10)
        assert abs(idf - expected) < 0.001

    def test_rare_term_higher_idf(self):
        """罕见 term 的 IDF 高于常见 term。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())

        idf_rag = indexer.inverted_index["rag"]["idf"]      # df=2
        idf_bm25 = indexer.inverted_index["bm25"]["idf"]    # df=1

        assert idf_bm25 > idf_rag

    def test_idf_rare_terms_positive(self):
        """罕见 term（df < N/2）的 IDF 为正数。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())

        # bm25 只出现在 1/3 文档中，IDF 应为正
        assert indexer.inverted_index["bm25"]["idf"] > 0

    def test_idf_single_doc_corpus(self):
        """单文档语料库的 IDF 计算。"""
        indexer = BM25Indexer()
        corpus = [_make_record("doc1", {"term": 1.0})]
        indexer.build(corpus)

        # df=1, N=1 → log((1-1+0.5)/(1+0.5)+1e-10) ≈ log(0.333) ≈ -1.099
        idf = indexer.inverted_index["term"]["idf"]
        assert isinstance(idf, float)


# ============================================================
# 测试：查询
# ============================================================

class TestQuery:
    """查询测试。"""

    def test_query_returns_top_k(self):
        """query 返回最多 top_k 个结果。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())

        results = indexer.query(["rag"], top_k=2)
        assert len(results) <= 2

    def test_query_relevant_docs_ranked_higher(self):
        """包含查询 term 的文档排名更高。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())

        results = indexer.query(["bm25"])
        # doc3 包含 "bm25"，应排在最前
        assert results[0][0] == "doc3"

    def test_query_multiple_terms(self):
        """多 term 查询累加得分。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())

        # "rag" + "检索"：doc1 同时包含两者，应排最前
        results = indexer.query(["rag", "检索"])
        assert results[0][0] == "doc1"

    def test_query_unknown_term_returns_empty(self):
        """未知 term 查询返回空结果。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())

        results = indexer.query(["nonexistent_term"])
        assert results == []

    def test_query_empty_terms_returns_empty(self):
        """空 term 列表返回空结果。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())

        results = indexer.query([])
        assert results == []

    def test_query_empty_index_returns_empty(self):
        """空索引返回空结果。"""
        indexer = BM25Indexer()
        results = indexer.query(["test"])
        assert results == []

    def test_query_stable_ordering(self):
        """相同查询返回稳定排序。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())

        results1 = indexer.query(["rag", "检索"])
        results2 = indexer.query(["rag", "检索"])

        assert results1 == results2

    def test_query_scores_are_floats(self):
        """所有查询得分为浮点数。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())

        results = indexer.query(["rag", "检索", "bm25"])
        for _, score in results:
            assert isinstance(score, float)


# ============================================================
# 测试：持久化 roundtrip
# ============================================================

class TestPersistence:
    """持久化 roundtrip 测试。"""

    def test_save_load_roundtrip(self):
        """save 后 load 恢复完整索引。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bm25_index.pkl")
            indexer.save(path)

            loaded = BM25Indexer()
            loaded.load(path)

            assert loaded.corpus_size == indexer.corpus_size
            assert len(loaded.inverted_index) == len(indexer.inverted_index)
            assert loaded.avg_doc_length == indexer.avg_doc_length

    def test_load_query_stable(self):
        """load 后查询结果与 save 前一致。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())
        original_results = indexer.query(["rag", "检索"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bm25_index.pkl")
            indexer.save(path)

            loaded = BM25Indexer()
            loaded.load(path)
            loaded_results = loaded.query(["rag", "检索"])

        assert original_results == loaded_results

    def test_load_preserves_idf(self):
        """load 后 IDF 值与 save 前一致。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bm25_index.pkl")
            indexer.save(path)

            loaded = BM25Indexer()
            loaded.load(path)

            for term in indexer.inverted_index:
                assert abs(
                    loaded.inverted_index[term]["idf"]
                    - indexer.inverted_index[term]["idf"]
                ) < 1e-10

    def test_save_creates_directory(self):
        """save 自动创建目录。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "bm25_index.pkl")
            indexer.save(path)

            assert os.path.exists(path)

    def test_load_nonexistent_raises(self):
        """加载不存在的文件抛 FileNotFoundError。"""
        indexer = BM25Indexer()
        with pytest.raises(FileNotFoundError):
            indexer.load("/nonexistent/path/index.pkl")


# ============================================================
# 测试：增量更新
# ============================================================

class TestIncrementalUpdate:
    """增量更新测试。"""

    def test_add_documents_increases_corpus_size(self):
        """add_documents 增加 corpus_size。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())
        old_size = indexer.corpus_size

        new_records = [_make_record("doc_new", {"新词": 1.0})]
        indexer.add_documents(new_records)

        assert indexer.corpus_size == old_size + 1

    def test_add_documents_updates_index(self):
        """add_documents 更新倒排索引。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())

        new_records = [_make_record("doc_new", {"新词": 1.0})]
        indexer.add_documents(new_records)

        assert "新词" in indexer.inverted_index
        assert len(indexer.inverted_index["新词"]["postings"]) == 1

    def test_add_documents_skips_existing(self):
        """add_documents 跳过已存在的文档。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())

        # 重复添加 doc1
        existing = [_make_record("doc1", {"rag": 1.0})]
        indexer.add_documents(existing)

        # corpus_size 不变
        assert indexer.corpus_size == 5

    def test_add_documents_recomputes_idf(self):
        """add_documents 重新计算 IDF。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())
        old_idf = indexer.inverted_index["rag"]["idf"]

        # 添加更多包含 "rag" 的文档
        new_records = [_make_record(f"new_{i}", {"rag": 1.0}) for i in range(5)]
        indexer.add_documents(new_records)

        # df 增加，IDF 应下降
        new_idf = indexer.inverted_index["rag"]["idf"]
        assert new_idf < old_idf

    def test_incremental_then_query(self):
        """增量更新后查询正常工作。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())

        new_records = [_make_record("doc_new", {"新词": 1.0, "rag": 1.0})]
        indexer.add_documents(new_records)

        results = indexer.query(["新词"])
        assert len(results) == 1
        assert results[0][0] == "doc_new"


# ============================================================
# 测试：文档移除
# ============================================================

class TestRemoveDocument:
    """文档移除测试。"""

    def test_remove_existing_document(self):
        """移除已存在的文档返回 True。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())
        old_size = indexer.corpus_size

        assert indexer.remove_document("doc1") is True
        assert indexer.corpus_size == old_size - 1

    def test_remove_nonexistent_returns_false(self):
        """移除不存在的文档返回 False。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())

        assert indexer.remove_document("nonexistent") is False

    def test_remove_cleans_postings(self):
        """移除文档后 postings 被清理。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())

        # "bm25" 只在 doc3 中
        indexer.remove_document("doc3")
        assert "bm25" not in indexer.inverted_index

    def test_remove_updates_idf(self):
        """移除文档后 IDF 更新。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())
        old_idf = indexer.inverted_index["rag"]["idf"]

        # 移除包含 "rag" 的 doc1
        indexer.remove_document("doc1")

        # N 减少，df 减少，IDF 应变化
        new_idf = indexer.inverted_index["rag"]["idf"]
        assert new_idf != old_idf

    def test_remove_then_query(self):
        """移除文档后查询不再返回该文档。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())

        indexer.remove_document("doc3")
        results = indexer.query(["bm25"])

        chunk_ids = [r[0] for r in results]
        assert "doc3" not in chunk_ids


# ============================================================
# 测试：辅助方法
# ============================================================

class TestHelperMethods:
    """辅助方法测试。"""

    def test_get_vocabulary_size(self):
        """get_vocabulary_size 返回正确数量。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())

        assert indexer.get_vocabulary_size() == len(indexer.inverted_index)

    def test_avg_doc_length(self):
        """avg_doc_length 计算正确。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())

        total = sum(indexer.doc_lengths.values())
        expected = total / indexer.corpus_size
        assert abs(indexer.avg_doc_length - expected) < 0.01


# ============================================================
# 测试：输出契约（可用于 SparseRetriever）
# ============================================================

class TestOutputContract:
    """输出契约测试。"""

    def test_query_returns_list_of_tuples(self):
        """query 返回 [(chunk_id, score), ...]。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())

        results = indexer.query(["rag"])
        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)

    def test_inverted_index_structure(self):
        """inverted_index 结构正确。"""
        indexer = BM25Indexer()
        indexer.build(_make_corpus())

        for term, entry in indexer.inverted_index.items():
            assert "idf" in entry
            assert "postings" in entry
            assert isinstance(entry["idf"], float)
            assert isinstance(entry["postings"], list)
            for posting in entry["postings"]:
                assert "chunk_id" in posting
                assert "tf" in posting
                assert "doc_length" in posting
