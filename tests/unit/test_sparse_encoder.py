"""C9 SparseEncoder 单元测试。

使用 Mock 和自定义分词器隔离测试，覆盖分词、TF 计算、停用词过滤、
空文本处理、Trace 记录等。
验收标准：输出结构可用于 bm25_indexer；对空文本有明确行为。
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import pytest

from src.core.settings import EmbeddingConfig, Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk, ChunkRecord
from src.ingestion.embedding.sparse_encoder import SparseEncoder


# ============================================================
# 测试辅助函数
# ============================================================


def _make_chunk(text: str, chunk_id: str = "test_0000_abcd1234") -> Chunk:
    """创建测试用 Chunk。"""
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "/test.pdf", "chunk_index": 0},
        source_ref="doc_001",
    )


def _make_settings_stub(**kwargs) -> Settings:
    """创建测试用 Settings stub。"""
    return Settings(
        llm=MagicMock(),
        vision_llm=MagicMock(),
        ollama=MagicMock(),
        embedding=MagicMock(sparse_backend=kwargs.get("sparse_backend", "jieba")),
        splitter=MagicMock(),
        vector_store=MagicMock(),
        retrieval=MagicMock(),
        rerank=MagicMock(),
        evaluation=MagicMock(),
        observability=MagicMock(),
        pipeline=MagicMock(),
    )


def _simple_tokenizer(text: str) -> List[str]:
    """简单分词器：按空格切分。"""
    return text.lower().split()


# ============================================================
# 测试：基本编码功能
# ============================================================

class TestEncodeBasic:
    """基本编码功能测试。"""

    def test_encode_returns_chunk_records(self):
        """encode 返回 ChunkRecord 列表。"""
        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=_simple_tokenizer)
        chunks = [_make_chunk("hello world", "c1"), _make_chunk("foo bar", "c2")]
        records = encoder.encode(chunks)

        assert len(records) == 2
        assert all(isinstance(r, ChunkRecord) for r in records)

    def test_encode_fills_sparse_vector(self):
        """encode 填充 sparse_vector 字段。"""
        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=_simple_tokenizer)
        chunks = [_make_chunk("hello world hello", "c1")]
        records = encoder.encode(chunks)

        assert records[0].sparse_vector is not None
        assert isinstance(records[0].sparse_vector, dict)

    def test_encode_vector_count_matches_chunks(self):
        """输出记录数量与 chunks 数量一致。"""
        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=_simple_tokenizer)
        chunks = [_make_chunk(f"text_{i}", f"c{i}") for i in range(5)]
        records = encoder.encode(chunks)

        assert len(records) == 5
        assert all(r.sparse_vector is not None for r in records)

    def test_encode_preserves_chunk_id(self):
        """encode 保持 chunk ID 不变。"""
        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=_simple_tokenizer)
        chunks = [_make_chunk("text_1", "id_001"), _make_chunk("text_2", "id_002")]
        records = encoder.encode(chunks)

        assert records[0].id == "id_001"
        assert records[1].id == "id_002"

    def test_encode_preserves_chunk_text(self):
        """encode 保持 chunk 文本不变。"""
        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=_simple_tokenizer)
        chunks = [_make_chunk("原始文本", "c1")]
        records = encoder.encode(chunks)

        assert records[0].text == "原始文本"

    def test_encode_preserves_metadata(self):
        """encode 保持原有 metadata 字段。"""
        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=_simple_tokenizer)
        chunk = _make_chunk("text", "c1")
        chunk.metadata["custom_field"] = 42
        records = encoder.encode([chunk])

        assert records[0].metadata["custom_field"] == 42
        assert records[0].metadata["source_path"] == "/test.pdf"


# ============================================================
# 测试：TF 计算
# ============================================================

class TestTFComputation:
    """TF 计算测试。"""

    def test_tf_counts_term_frequency(self):
        """TF 正确统计词频。"""
        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=_simple_tokenizer)
        chunks = [_make_chunk("hello world hello foo", "c1")]
        records = encoder.encode(chunks)
        tf = records[0].sparse_vector

        # "hello" 出现 2 次，其他出现 1 次
        assert tf["hello"] > tf["world"]
        assert tf["hello"] > tf["foo"]

    def test_tf_log_normalization(self):
        """TF 使用对数归一化：1 + log(count)。"""
        import math

        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=_simple_tokenizer)
        # "hello" 出现 3 次
        chunks = [_make_chunk("hello hello hello world", "c1")]
        records = encoder.encode(chunks)
        tf = records[0].sparse_vector

        # TF("hello") = 1 + log(3) ≈ 2.099
        expected = 1.0 + math.log(3)
        assert abs(tf["hello"] - expected) < 0.001

    def test_tf_single_occurrence(self):
        """出现 1 次的 term，TF = 1.0。"""
        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=_simple_tokenizer)
        chunks = [_make_chunk("unique word", "c1")]
        records = encoder.encode(chunks)
        tf = records[0].sparse_vector

        assert tf["unique"] == 1.0
        assert tf["word"] == 1.0

    def test_tf_deterministic(self):
        """相同输入产生相同的 TF 输出。"""
        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=_simple_tokenizer)
        chunk = _make_chunk("hello world hello", "c1")
        records1 = encoder.encode([chunk])
        records2 = encoder.encode([chunk])

        assert records1[0].sparse_vector == records2[0].sparse_vector


# ============================================================
# 测试：停用词过滤
# ============================================================

class TestStopwords:
    """停用词过滤测试。"""

    def test_default_stopwords_filtered(self):
        """默认停用词被过滤。"""
        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=_simple_tokenizer)
        # "the" 和 "is" 是英文停用词
        chunks = [_make_chunk("the cat is cute", "c1")]
        records = encoder.encode(chunks)
        tf = records[0].sparse_vector

        assert "the" not in tf
        assert "is" not in tf
        assert "cat" in tf
        assert "cute" in tf

    def test_chinese_stopwords_filtered(self):
        """中文停用词被过滤。"""
        def cn_tokenizer(text: str) -> List[str]:
            return list(text)
        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=cn_tokenizer)
        # "的"、"是"、"在" 是中文停用词
        chunks = [_make_chunk("好的是的在", "c1")]
        records = encoder.encode(chunks)
        tf = records[0].sparse_vector

        assert "的" not in tf
        assert "是" not in tf
        assert "在" not in tf

    def test_custom_stopwords(self):
        """自定义停用词生效。"""
        custom_stopwords = {"cat", "dog"}
        encoder = SparseEncoder(
            settings=_make_settings_stub(),
            tokenizer=_simple_tokenizer,
            stopwords=custom_stopwords,
        )
        chunks = [_make_chunk("cat dog bird fish", "c1")]
        records = encoder.encode(chunks)
        tf = records[0].sparse_vector

        assert "cat" not in tf
        assert "dog" not in tf
        assert "bird" in tf
        assert "fish" in tf

    def test_empty_stopwords_keeps_all(self):
        """空停用词集合保留所有 token。"""
        encoder = SparseEncoder(
            settings=_make_settings_stub(),
            tokenizer=_simple_tokenizer,
            stopwords=set(),
        )
        chunks = [_make_chunk("the is are", "c1")]
        records = encoder.encode(chunks)
        tf = records[0].sparse_vector

        assert "the" in tf
        assert "is" in tf
        assert "are" in tf


# ============================================================
# 测试：空文本处理
# ============================================================

class TestEmptyInput:
    """空输入测试。"""

    def test_empty_chunks_returns_empty(self):
        """空列表输入返回空列表。"""
        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=_simple_tokenizer)
        records = encoder.encode([])

        assert records == []

    def test_empty_text_returns_empty_sparse_vector(self):
        """空文本产生空 sparse_vector。"""
        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=_simple_tokenizer)
        chunks = [_make_chunk("", "c1")]
        records = encoder.encode(chunks)

        assert records[0].sparse_vector == {}

    def test_whitespace_only_returns_empty(self):
        """纯空白文本产生空 sparse_vector。"""
        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=_simple_tokenizer)
        chunks = [_make_chunk("   \n\t  ", "c1")]
        records = encoder.encode(chunks)

        assert records[0].sparse_vector == {}

    def test_all_stopwords_returns_empty(self):
        """全部是停用词时产生空 sparse_vector。"""
        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=_simple_tokenizer)
        # "the is are" 全部是停用词
        chunks = [_make_chunk("the is are", "c1")]
        records = encoder.encode(chunks)

        assert records[0].sparse_vector == {}


# ============================================================
# 测试：最小 token 长度
# ============================================================

class TestMinTokenLength:
    """最小 token 长度过滤测试。"""

    def test_short_tokens_filtered(self):
        """低于 min_token_length 的 token 被过滤。"""
        encoder = SparseEncoder(
            settings=_make_settings_stub(),
            tokenizer=_simple_tokenizer,
            min_token_length=3,
            stopwords=set(),  # 清空停用词以隔离测试
        )
        chunks = [_make_chunk("a bb ccc dddd", "c1")]
        records = encoder.encode(chunks)
        tf = records[0].sparse_vector

        assert "a" not in tf
        assert "bb" not in tf
        assert "ccc" in tf
        assert "dddd" in tf


# ============================================================
# 测试：自定义分词器
# ============================================================

class TestCustomTokenizer:
    """自定义分词器测试。"""

    def test_custom_tokenizer_used(self):
        """自定义分词器被正确调用。"""
        call_log = []

        def logging_tokenizer(text: str) -> List[str]:
            call_log.append(text)
            return text.split()

        encoder = SparseEncoder(
            settings=_make_settings_stub(),
            tokenizer=logging_tokenizer,
        )
        chunks = [_make_chunk("hello world", "c1")]
        encoder.encode(chunks)

        assert call_log == ["hello world"]

    def test_tokenizer_case_insensitive(self):
        """分词结果统一小写。"""
        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=_simple_tokenizer)
        chunks = [_make_chunk("Hello HELLO hello", "c1")]
        records = encoder.encode(chunks)
        tf = records[0].sparse_vector

        # 所有变体应合并为 "hello"
        assert "Hello" not in tf
        assert "HELLO" not in tf
        assert "hello" in tf
        # TF = 1 + log(3)
        import math
        assert abs(tf["hello"] - (1.0 + math.log(3))) < 0.001


# ============================================================
# 测试：异常处理
# ============================================================

class TestErrorHandling:
    """异常处理测试。"""

    def test_tokenizer_exception_does_not_crash(self):
        """分词器抛异常时不应崩溃。"""
        def bad_tokenizer(text: str) -> List[str]:
            raise RuntimeError("tokenizer error")

        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=bad_tokenizer)
        chunks = [_make_chunk("text", "c1")]

        # 应抛出异常（单 chunk 级别没有 try/except）
        with pytest.raises(RuntimeError, match="tokenizer error"):
            encoder.encode(chunks)

    def test_metadata_not_shared_between_records(self):
        """不同 ChunkRecord 的 metadata 应独立。"""
        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=_simple_tokenizer)
        chunks = [_make_chunk("text_1", "c1"), _make_chunk("text_2", "c2")]
        records = encoder.encode(chunks)

        records[0].metadata["extra"] = "value"
        assert "extra" not in records[1].metadata


# ============================================================
# 测试：Trace 记录
# ============================================================

class TestTraceRecording:
    """Trace 阶段记录测试。"""

    def test_trace_records_sparse_encode_stage(self):
        """encode 成功时 trace 记录 sparse_encode 阶段。"""
        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=_simple_tokenizer)
        trace = TraceContext()
        encoder.encode([_make_chunk("hello world", "c1")], trace=trace)

        stage_names = [s["name"] for s in trace.stages]
        assert "sparse_encode" in stage_names

    def test_trace_records_chunk_count(self):
        """trace 记录 chunk 数量。"""
        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=_simple_tokenizer)
        trace = TraceContext()
        chunks = [_make_chunk(f"text_{i}", f"c{i}") for i in range(3)]
        encoder.encode(chunks, trace=trace)

        sparse_stage = [s for s in trace.stages if s["name"] == "sparse_encode"][0]
        assert sparse_stage["chunk_count"] == 3

    def test_trace_records_total_terms(self):
        """trace 记录总 term 数量。"""
        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=_simple_tokenizer)
        trace = TraceContext()
        # "hello world" → 2 个 term（无停用词时）
        encoder.encode([_make_chunk("hello world", "c1")], trace=trace)

        sparse_stage = [s for s in trace.stages if s["name"] == "sparse_encode"][0]
        assert sparse_stage["total_terms"] == 2

    def test_trace_records_elapsed_ms(self):
        """trace 记录耗时。"""
        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=_simple_tokenizer)
        trace = TraceContext()
        encoder.encode([_make_chunk("text", "c1")], trace=trace)

        sparse_stage = [s for s in trace.stages if s["name"] == "sparse_encode"][0]
        assert "elapsed_ms" in sparse_stage
        assert sparse_stage["elapsed_ms"] >= 0

    def test_no_trace_no_error(self):
        """不传 trace 时不报错。"""
        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=_simple_tokenizer)
        records = encoder.encode([_make_chunk("text", "c1")])
        assert len(records) == 1


# ============================================================
# 测试：输出契约（可用于 BM25Indexer）
# ============================================================

class TestOutputContract:
    """输出契约测试：验证 sparse_vector 格式可被 BM25Indexer 使用。"""

    def test_sparse_vector_is_dict_str_float(self):
        """sparse_vector 类型为 Dict[str, float]。"""
        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=_simple_tokenizer)
        records = encoder.encode([_make_chunk("hello world", "c1")])
        sv = records[0].sparse_vector

        assert isinstance(sv, dict)
        for key, value in sv.items():
            assert isinstance(key, str)
            assert isinstance(value, float)

    def test_sparse_vector_all_positive_weights(self):
        """所有 term weights 为正数。"""
        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=_simple_tokenizer)
        records = encoder.encode([_make_chunk("hello world test", "c1")])
        sv = records[0].sparse_vector

        assert all(w > 0 for w in sv.values())

    def test_sparse_vector_keys_are_lowercase(self):
        """所有 term keys 为小写。"""
        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=_simple_tokenizer)
        records = encoder.encode([_make_chunk("Hello WORLD Test", "c1")])
        sv = records[0].sparse_vector

        assert all(k == k.lower() for k in sv.keys())

    def test_chunk_record_serializable(self):
        """含 sparse_vector 的 ChunkRecord 可序列化。"""
        encoder = SparseEncoder(settings=_make_settings_stub(), tokenizer=_simple_tokenizer)
        records = encoder.encode([_make_chunk("hello world", "c1")])
        record = records[0]

        d = record.to_dict()
        assert "sparse_vector" in d
        assert isinstance(d["sparse_vector"], dict)

        restored = ChunkRecord.from_dict(d)
        assert restored.sparse_vector == record.sparse_vector


# ============================================================
# 测试：BGE-M3 后端
# ============================================================

class TestBGEBackend:
    """BGE-M3 后端测试（使用 Mock）。"""

    def _make_bge_encoder(self) -> SparseEncoder:
        """创建 BGE 后端的 SparseEncoder（Mock 模型）。"""
        settings = _make_settings_stub(sparse_backend="bge")
        settings.embedding.model = "bge-m3"
        settings.embedding.model_path = "models/bge-m3"
        encoder = SparseEncoder(settings=settings, backend="bge")

        # Mock BGE 模型和 tokenizer
        mock_tokenizer = MagicMock()
        # token_id → token_string 映射
        mock_tokenizer.decode.side_effect = lambda ids: {
            100: "rag",
            200: "检索",
            300: "生成",
            400: "向量",
            500: "数据库",
            600: "[CLS]",
            700: "的",
        }.get(ids[0], f"token_{ids[0]}")

        mock_bge_model = MagicMock()
        # embed_with_sparse 返回 (dense, sparse)
        # BGE-M3 实际返回 str key，如 {"100": 1.5}
        mock_bge_model.embed_with_sparse.return_value = (
            [[0.1, 0.2, 0.3]],  # dense
            [{"100": 1.5, "200": 0.8, "300": 0.3}],  # sparse: {str(token_id): weight}
        )

        encoder._bge_model = mock_bge_model
        encoder._bge_tokenizer = mock_tokenizer
        encoder._bge_tried = True
        return encoder

    def test_bge_encode_returns_chunk_records(self):
        """BGE 后端 encode 返回 ChunkRecord 列表。"""
        encoder = self._make_bge_encoder()
        chunks = [_make_chunk("RAG 检索增强生成", "c1")]
        records = encoder.encode(chunks)

        assert len(records) == 1
        assert isinstance(records[0], ChunkRecord)

    def test_bge_encode_converts_token_ids_to_strings(self):
        """BGE 后端将 token_id 转换为 token_string。"""
        encoder = self._make_bge_encoder()
        chunks = [_make_chunk("RAG 检索生成", "c1")]
        records = encoder.encode(chunks)
        sv = records[0].sparse_vector

        # token_id 100→"rag", 200→"检索", 300→"生成"
        assert "rag" in sv
        assert "检索" in sv
        assert "生成" in sv
        # 值应保留原始权重
        assert sv["rag"] == 1.5
        assert sv["检索"] == 0.8

    def test_bge_encode_filters_stopwords(self):
        """BGE 后端过滤停用词。"""
        encoder = self._make_bge_encoder()
        # Mock tokenizer 将 700→"的"（停用词）
        encoder._bge_model.embed_with_sparse.return_value = (
            [[0.1]],
            [{"100": 1.5, "700": 0.5}],  # 700→"的" 是停用词
        )
        chunks = [_make_chunk("RAG 的", "c1")]
        records = encoder.encode(chunks)
        sv = records[0].sparse_vector

        assert "rag" in sv
        assert "的" not in sv

    def test_bge_encode_filters_special_tokens(self):
        """BGE 后端过滤特殊 token（[CLS] 等）。"""
        encoder = self._make_bge_encoder()
        encoder._bge_model.embed_with_sparse.return_value = (
            [[0.1]],
            [{"100": 1.5, "600": 0.5}],  # 600→"[CLS]" 是特殊 token
        )
        chunks = [_make_chunk("RAG [CLS]", "c1")]
        records = encoder.encode(chunks)
        sv = records[0].sparse_vector

        assert "rag" in sv
        assert "[CLS]" not in sv

    def test_bge_encode_filters_zero_weight(self):
        """BGE 后端过滤零权重 token。"""
        encoder = self._make_bge_encoder()
        encoder._bge_model.embed_with_sparse.return_value = (
            [[0.1]],
            [{"100": 1.5, "200": 0.0}],  # 200 权重为 0
        )
        chunks = [_make_chunk("RAG 检索", "c1")]
        records = encoder.encode(chunks)
        sv = records[0].sparse_vector

        assert "rag" in sv
        assert "检索" not in sv

    def test_bge_encode_preserves_metadata(self):
        """BGE 后端保持 metadata 不变。"""
        encoder = self._make_bge_encoder()
        chunk = _make_chunk("RAG", "c1")
        chunk.metadata["custom"] = 42
        records = encoder.encode([chunk])

        assert records[0].metadata["custom"] == 42

    def test_bge_encode_empty_chunks(self):
        """BGE 后端空列表返回空。"""
        encoder = self._make_bge_encoder()
        records = encoder.encode([])

        assert records == []

    def test_bge_encode_trace_recording(self):
        """BGE 后端记录 Trace。"""
        encoder = self._make_bge_encoder()
        trace = TraceContext()
        encoder.encode([_make_chunk("RAG", "c1")], trace=trace)

        stage_names = [s["name"] for s in trace.stages]
        assert "sparse_encode" in stage_names

        sparse_stage = [s for s in trace.stages if s["name"] == "sparse_encode"][0]
        assert sparse_stage["method"] == "bge"

    def test_bge_fallback_to_jieba_on_load_failure(self):
        """BGE 加载失败时降级到 jieba。"""
        settings = _make_settings_stub(sparse_backend="bge")
        settings.embedding.model = "bge-m3"
        settings.embedding.model_path = "models/bge-m3"
        encoder = SparseEncoder(settings=settings, backend="bge")

        # Mock BGE 加载失败
        encoder._bge_tried = True
        encoder._bge_model = None

        # 应降级到 jieba（使用自定义 tokenizer 避免 jieba 依赖）
        encoder._tokenizer = _simple_tokenizer
        chunks = [_make_chunk("hello world", "c1")]
        records = encoder.encode(chunks)

        # 应该返回结果（jieba 路径）
        assert len(records) == 1
        assert "hello" in records[0].sparse_vector

    def test_bge_output_format_compatible_with_bm25(self):
        """BGE 输出格式兼容 BM25Indexer（Dict[str, float]）。"""
        encoder = self._make_bge_encoder()
        records = encoder.encode([_make_chunk("RAG 检索", "c1")])
        sv = records[0].sparse_vector

        assert isinstance(sv, dict)
        for key, value in sv.items():
            assert isinstance(key, str)
            assert isinstance(value, float)
            assert value > 0
