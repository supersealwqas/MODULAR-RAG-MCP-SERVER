"""ingest.py 脚本端到端测试。

测试 CLI 参数解析、文件收集、Pipeline 调用和输出行为。
使用 Fake 组件和临时目录隔离外部依赖。
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

# 确保项目根目录在 path 中
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.ingest import (
    _collect_files,
    _format_stage_times,
    _print_progress,
    _print_result,
    main,
    parse_args,
)
from src.core.settings import (
    EmbeddingConfig,
    EvaluationConfig,
    LLMConfig,
    ObservabilityConfig,
    OllamaConfig,
    RetrievalConfig,
    RerankConfig,
    Settings,
    SplitterConfig,
    VectorStoreConfig,
    VisionLLMConfig,
)
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk, ChunkRecord, Document
from src.ingestion.pipeline import IngestionPipeline, PipelineError, PipelineResult
from src.libs.loader.base_loader import BaseLoader


# ============================================================
# Fake 组件（与 C14 集成测试共享模式）
# ============================================================


class FakeLoader(BaseLoader):
    """Fake 文件加载器，返回预构建的 Document。"""

    def __init__(self, document: Document) -> None:
        self._document = document
        self.call_count = 0

    def load(self, path: str, collection: str = "default") -> Document:
        self.call_count += 1
        return self._document


class FakeBatchProcessor:
    """Fake 批量编码器。"""

    def __init__(self) -> None:
        self.call_count = 0

    def process(self, chunks: List[Chunk], trace=None) -> List[ChunkRecord]:
        self.call_count += 1
        return [
            ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=chunk.metadata.copy(),
                dense_vector=[0.1, 0.2, 0.3],
                sparse_vector={"test": 1.0},
            )
            for chunk in chunks
        ]


class FakeVectorUpserter:
    """Fake 向量写入器。"""

    def __init__(self) -> None:
        self.call_count = 0

    def upsert(self, records: List[ChunkRecord], trace=None) -> int:
        self.call_count += 1
        return len(records)

    def delete(self, chunk_ids: List[str], trace=None) -> int:
        return len(chunk_ids)


class FakeBM25Indexer:
    """Fake BM25 索引器。"""

    def __init__(self) -> None:
        self.build_count = 0
        self.save_count = 0

    def build(self, records: List[ChunkRecord], trace=None) -> None:
        self.build_count += 1

    def save(self, path=None) -> str:
        self.save_count += 1
        return path or "fake"

    def get_vocabulary_size(self) -> int:
        return 10


class FakeIntegrityChecker:
    """Fake 文件完整性检查器。"""

    def __init__(self) -> None:
        self._processed = {}
        self._hash_cache = {}

    def compute_hash(self, file_path: str) -> str:
        if file_path not in self._hash_cache:
            self._hash_cache[file_path] = f"{hash(file_path) % (10**64):064d}"
        return self._hash_cache[file_path]

    def should_skip(self, file_hash: str) -> bool:
        return self._processed.get(file_hash) == "success"

    def mark_success(self, file_hash: str, file_path: str, **kwargs) -> None:
        self._processed[file_hash] = "success"

    def mark_failed(self, file_hash: str, file_path: str, error_msg: str) -> None:
        self._processed[file_hash] = "failed"

    def get_status(self, file_hash: str) -> Optional[str]:
        return self._processed.get(file_hash)


# ============================================================
# 辅助工具
# ============================================================


def _make_settings() -> Settings:
    """创建最小测试配置。"""
    return Settings(
        llm=LLMConfig(provider="fake", model="fake"),
        vision_llm=VisionLLMConfig(),
        ollama=OllamaConfig(),
        embedding=EmbeddingConfig(provider="fake", model="fake", dimensions=3),
        splitter=SplitterConfig(strategy="recursive", chunk_size=500, chunk_overlap=50),
        vector_store=VectorStoreConfig(provider="fake"),
        retrieval=RetrievalConfig(),
        rerank=RerankConfig(),
        evaluation=EvaluationConfig(),
        observability=ObservabilityConfig(),
    )


def _make_document(
    text: str = "这是测试文档。" * 20,
    source_path: str = "test.pdf",
) -> Document:
    """创建测试 Document。"""
    return Document(
        id="test_doc",
        text=text,
        metadata={"source_path": source_path, "doc_type": "pdf"},
    )


def _create_temp_pdf(directory: Path, name: str = "test.pdf") -> Path:
    """在临时目录创建一个最小 PDF 文件。

    使用 reportlab 或直接写入最小 PDF 结构。
    """
    pdf_path = directory / name
    # 最小 PDF 结构（可被 PdfLoader 解析）
    minimal_pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 12 Tf 100 700 Td "
        b"(Hello World) Tj ET\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000364 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
        b"startxref\n445\n%%EOF\n"
    )
    pdf_path.write_bytes(minimal_pdf)
    return pdf_path


def _create_temp_text(directory: Path, name: str, content: str = "") -> Path:
    """在临时目录创建文本文件。"""
    text_path = directory / name
    text_path.write_text(content or f"这是 {name} 的测试内容。" * 10, encoding="utf-8")
    return text_path


# ============================================================
# 测试用例
# ============================================================


class TestParseArgs:
    """命令行参数解析测试。"""

    def test_parse_args_required_path(self):
        """--path 是必填参数。"""
        args = parse_args(["--path", "test.pdf"])
        assert args.path == "test.pdf"
        assert args.collection == "default"
        assert args.force is False
        assert args.config == "config/settings.yaml"

    def test_parse_args_all_options(self):
        """所有参数均可正确解析。"""
        args = parse_args([
            "--path", "/data/docs/",
            "--collection", "my_collection",
            "--force",
            "--config", "custom.yaml",
        ])
        assert args.path == "/data/docs/"
        assert args.collection == "my_collection"
        assert args.force is True
        assert args.config == "custom.yaml"

    def test_parse_args_defaults(self):
        """未指定可选参数时使用默认值。"""
        args = parse_args(["--path", "x"])
        assert args.collection == "default"
        assert args.force is False
        assert args.config == "config/settings.yaml"

    def test_parse_args_missing_path_raises(self):
        """缺少 --path 时应报错。"""
        with pytest.raises(SystemExit):
            parse_args([])


class TestCollectFiles:
    """文件收集测试。"""

    def test_collect_single_file(self):
        """单文件路径应返回包含该文件的列表。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf = _create_temp_pdf(Path(tmpdir), "test.pdf")
            result = _collect_files(pdf)
            assert len(result) == 1
            assert result[0] == pdf

    def test_collect_directory(self):
        """目录模式应递归收集支持格式的文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _create_temp_pdf(tmppath, "a.pdf")
            _create_temp_text(tmppath, "b.md", "# 测试")
            _create_temp_text(tmppath, "c.txt", "文本")
            # 不支持的格式应被忽略
            (tmppath / "d.xyz").write_bytes(b"unknown")

            result = _collect_files(tmppath)
            names = [f.name for f in result]
            assert "a.pdf" in names
            assert "b.md" in names
            assert "c.txt" in names
            assert "d.xyz" not in names

    def test_collect_not_found_raises(self):
        """路径不存在时应抛出 FileNotFoundError。"""
        # 使用临时目录中确定不存在的路径
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "definitely_not_exist" / "nope.pdf"
            with pytest.raises(FileNotFoundError):
                _collect_files(nonexistent)

    def test_collect_empty_directory_raises(self):
        """空目录（无支持格式文件）应抛出 ValueError。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="未找到支持格式"):
                _collect_files(Path(tmpdir))

    def test_collect_subdirectory_recursive(self):
        """应递归搜索子目录。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            subdir = tmppath / "sub"
            subdir.mkdir()
            _create_temp_pdf(subdir, "nested.pdf")

            result = _collect_files(tmppath)
            assert len(result) == 1
            assert result[0].name == "nested.pdf"

    def test_collect_sorted_output(self):
        """返回的文件列表应按路径排序。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _create_temp_pdf(tmppath, "z.pdf")
            _create_temp_pdf(tmppath, "a.pdf")
            _create_temp_text(tmppath, "m.md", "test")

            result = _collect_files(tmppath)
            names = [f.name for f in result]
            assert names == sorted(names)


class TestFormatStageTimes:
    """阶段耗时格式化测试。"""

    def test_format_empty(self):
        """空字典返回空字符串。"""
        assert _format_stage_times({}) == ""

    def test_format_single_stage(self):
        """单阶段格式化。"""
        result = _format_stage_times({"load": 123.4})
        assert "load=123ms" in result

    def test_format_multiple_stages(self):
        """多阶段格式化。"""
        result = _format_stage_times({"load": 100, "split": 200, "encode": 300})
        assert "load=100ms" in result
        assert "split=200ms" in result
        assert "encode=300ms" in result


class TestPrintResult:
    """结果打印测试。"""

    def test_print_skipped_result(self, capsys):
        """跳过的结果应显示跳过标记。"""
        result = PipelineResult(
            file_path="test.pdf",
            collection="default",
            file_hash="",
            doc_id="",
            chunk_count=0,
            record_count=0,
            skipped=True,
            elapsed_ms=0,
        )
        _print_result(result, 1, 1)
        captured = capsys.readouterr()
        assert "已跳过" in captured.err
        assert "test.pdf" in captured.err

    def test_print_success_result(self, capsys):
        """成功的结果应显示完成标记和统计。"""
        result = PipelineResult(
            file_path="test.pdf",
            collection="default",
            file_hash="abc123",
            doc_id="doc1",
            chunk_count=5,
            record_count=5,
            skipped=False,
            elapsed_ms=1234.5,
        )
        _print_result(result, 2, 3)
        captured = capsys.readouterr()
        assert "完成" in captured.err
        assert "[2/3]" in captured.err
        assert "5 chunks" in captured.err
        assert "5 records" in captured.err


class TestMainWithMockedPipeline:
    """使用 mock Pipeline 测试 main() 入口。"""

    def test_main_single_file_success(self, capsys):
        """单文件成功摄取。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf = _create_temp_pdf(Path(tmpdir), "test.pdf")

            mock_result = PipelineResult(
                file_path=str(pdf),
                collection="default",
                file_hash="abc",
                doc_id="doc1",
                chunk_count=3,
                record_count=3,
                skipped=False,
                elapsed_ms=100.0,
                stage_times={"load": 10, "split": 20},
            )

            with patch("scripts.ingest.load_settings") as mock_ls, \
                 patch("scripts.ingest.IngestionPipeline") as mock_pipe_cls:
                mock_ls.return_value = _make_settings()
                mock_pipe = MagicMock()
                mock_pipe.run.return_value = mock_result
                mock_pipe_cls.return_value = mock_pipe

                exit_code = main(["--path", str(pdf)])

                assert exit_code == 0
                mock_pipe.run.assert_called_once()
                call_kwargs = mock_pipe.run.call_args
                assert call_kwargs[1]["collection"] == "default"
                assert call_kwargs[1]["force"] is False

    def test_main_directory_success(self, capsys):
        """目录模式应处理所有文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _create_temp_pdf(tmppath, "a.pdf")
            _create_temp_text(tmppath, "b.md", "content")

            mock_result = PipelineResult(
                file_path="",
                collection="test_col",
                file_hash="abc",
                doc_id="doc1",
                chunk_count=2,
                record_count=2,
                skipped=False,
                elapsed_ms=50.0,
            )

            with patch("scripts.ingest.load_settings") as mock_ls, \
                 patch("scripts.ingest.IngestionPipeline") as mock_pipe_cls:
                mock_ls.return_value = _make_settings()
                mock_pipe = MagicMock()
                mock_pipe.run.return_value = mock_result
                mock_pipe_cls.return_value = mock_pipe

                exit_code = main(["--path", str(tmppath), "--collection", "test_col"])

                assert exit_code == 0
                assert mock_pipe.run.call_count == 2

    def test_main_skip_behavior(self, capsys):
        """第二次运行应跳过已处理文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf = _create_temp_pdf(Path(tmpdir), "test.pdf")

            first_result = PipelineResult(
                file_path=str(pdf), collection="default", file_hash="abc",
                doc_id="doc1", chunk_count=3, record_count=3,
                skipped=False, elapsed_ms=100.0,
            )
            skip_result = PipelineResult(
                file_path=str(pdf), collection="default", file_hash="",
                doc_id="", chunk_count=0, record_count=0,
                skipped=True, elapsed_ms=1.0,
            )

            with patch("scripts.ingest.load_settings") as mock_ls, \
                 patch("scripts.ingest.IngestionPipeline") as mock_pipe_cls:
                mock_ls.return_value = _make_settings()
                mock_pipe = MagicMock()
                mock_pipe.run.side_effect = [first_result, skip_result]
                mock_pipe_cls.return_value = mock_pipe

                # 第一次
                code1 = main(["--path", str(pdf)])
                # 第二次
                code2 = main(["--path", str(pdf)])

                assert code1 == 0
                assert code2 == 0

    def test_main_force_flag(self, capsys):
        """--force 应传递给 Pipeline。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf = _create_temp_pdf(Path(tmpdir), "test.pdf")

            mock_result = PipelineResult(
                file_path=str(pdf), collection="default", file_hash="abc",
                doc_id="doc1", chunk_count=1, record_count=1,
                skipped=False, elapsed_ms=50.0,
            )

            with patch("scripts.ingest.load_settings") as mock_ls, \
                 patch("scripts.ingest.IngestionPipeline") as mock_pipe_cls:
                mock_ls.return_value = _make_settings()
                mock_pipe = MagicMock()
                mock_pipe.run.return_value = mock_result
                mock_pipe_cls.return_value = mock_pipe

                exit_code = main(["--path", str(pdf), "--force"])

                assert exit_code == 0
                call_kwargs = mock_pipe.run.call_args[1]
                assert call_kwargs["force"] is True

    def test_main_collection_passed(self):
        """--collection 参数应传递给 Pipeline。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf = _create_temp_pdf(Path(tmpdir), "test.pdf")

            with patch("scripts.ingest.load_settings") as mock_ls, \
                 patch("scripts.ingest.IngestionPipeline") as mock_pipe_cls:
                mock_ls.return_value = _make_settings()
                mock_pipe = MagicMock()
                mock_pipe.run.return_value = PipelineResult(
                    file_path=str(pdf), collection="custom", file_hash="h",
                    doc_id="d", chunk_count=1, record_count=1,
                    skipped=False, elapsed_ms=10.0,
                )
                mock_pipe_cls.return_value = mock_pipe

                main(["--path", str(pdf), "--collection", "custom"])

                call_kwargs = mock_pipe.run.call_args[1]
                assert call_kwargs["collection"] == "custom"

    def test_main_file_not_found_exits_1(self, capsys):
        """路径不存在时应返回退出码 1。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = str(Path(tmpdir) / "no_such_dir" / "file.pdf")
            with patch("scripts.ingest.load_settings") as mock_ls:
                mock_ls.return_value = _make_settings()
                exit_code = main(["--path", nonexistent])
                assert exit_code == 1

    def test_main_pipeline_error_exits_1(self):
        """Pipeline 失败时应返回退出码 1。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf = _create_temp_pdf(Path(tmpdir), "test.pdf")

            with patch("scripts.ingest.load_settings") as mock_ls, \
                 patch("scripts.ingest.IngestionPipeline") as mock_pipe_cls:
                mock_ls.return_value = _make_settings()
                mock_pipe = MagicMock()
                mock_pipe.run.side_effect = PipelineError("load", RuntimeError("加载失败"))
                mock_pipe_cls.return_value = mock_pipe

                exit_code = main(["--path", str(pdf)])
                assert exit_code == 1

    def test_main_partial_failure(self):
        """部分文件失败时应返回退出码 1，但继续处理其余文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _create_temp_pdf(tmppath, "good.pdf")
            _create_temp_pdf(tmppath, "bad.pdf")

            success_result = PipelineResult(
                file_path="good.pdf", collection="default", file_hash="h",
                doc_id="d", chunk_count=1, record_count=1,
                skipped=False, elapsed_ms=10.0,
            )

            with patch("scripts.ingest.load_settings") as mock_ls, \
                 patch("scripts.ingest.IngestionPipeline") as mock_pipe_cls:
                mock_ls.return_value = _make_settings()
                mock_pipe = MagicMock()
                # _collect_files 按字母排序: bad.pdf 先于 good.pdf
                mock_pipe.run.side_effect = [
                    PipelineError("load", RuntimeError("损坏")),
                    success_result,
                ]
                mock_pipe_cls.return_value = mock_pipe

                exit_code = main(["--path", str(tmppath)])

                assert exit_code == 1
                assert mock_pipe.run.call_count == 2


class TestMainProgressCallback:
    """进度回调测试。"""

    def test_progress_callback_called(self, capsys):
        """进度回调应输出到 stderr。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf = _create_temp_pdf(Path(tmpdir), "test.pdf")

            mock_result = PipelineResult(
                file_path=str(pdf), collection="default", file_hash="h",
                doc_id="d", chunk_count=1, record_count=1,
                skipped=False, elapsed_ms=50.0,
            )

            with patch("scripts.ingest.load_settings") as mock_ls, \
                 patch("scripts.ingest.IngestionPipeline") as mock_pipe_cls:
                mock_ls.return_value = _make_settings()
                mock_pipe = MagicMock()

                # 捕获 on_progress 回调并模拟调用
                def fake_run(*args, **kwargs):
                    on_progress = kwargs.get("on_progress")
                    if on_progress:
                        on_progress("load", 2, 6)
                    return mock_result

                mock_pipe.run.side_effect = fake_run
                mock_pipe_cls.return_value = mock_pipe

                main(["--path", str(pdf)])

                captured = capsys.readouterr()
                assert "load" in captured.err
                assert "[2/6]" in captured.err


class TestMainSummaryOutput:
    """汇总输出测试。"""

    def test_summary_shows_counts(self, capsys):
        """汇总应显示成功/跳过/失败计数。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf = _create_temp_pdf(Path(tmpdir), "test.pdf")

            with patch("scripts.ingest.load_settings") as mock_ls, \
                 patch("scripts.ingest.IngestionPipeline") as mock_pipe_cls:
                mock_ls.return_value = _make_settings()
                mock_pipe = MagicMock()
                mock_pipe.run.return_value = PipelineResult(
                    file_path=str(pdf), collection="default", file_hash="h",
                    doc_id="d", chunk_count=3, record_count=3,
                    skipped=False, elapsed_ms=100.0,
                )
                mock_pipe_cls.return_value = mock_pipe

                main(["--path", str(pdf)])

                captured = capsys.readouterr()
                assert "成功:" in captured.err
                assert "跳过:" in captured.err
                assert "失败:" in captured.err
                assert "总耗时:" in captured.err
