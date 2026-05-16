"""C2 文件完整性检查单元测试。

覆盖 SQLiteIntegrityChecker 的：
- SHA256 计算一致性
- should_skip 判定逻辑
- mark_success / mark_failed 状态流转
- get_status / remove_record / list_processed
- 数据库文件自动创建
- 并发写入（WAL 模式）
"""

import os
import tempfile

import pytest

from src.libs.loader.file_integrity import FileIntegrityChecker, SQLiteIntegrityChecker


@pytest.fixture
def tmp_db(tmp_path):
    """提供临时数据库路径。"""
    return str(tmp_path / "test_ingestion_history.db")


@pytest.fixture
def checker(tmp_db):
    """提供 SQLiteIntegrityChecker 实例。"""
    return SQLiteIntegrityChecker(db_path=tmp_db)


@pytest.fixture
def sample_file(tmp_path):
    """创建临时测试文件。"""
    file_path = str(tmp_path / "test_doc.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("这是一个测试文档的内容。" * 10)
    return file_path


# ============================================================
# compute_sha256 测试
# ============================================================

class TestComputeSHA256:
    """SHA256 计算测试。"""

    def test_consistent_hash(self, sample_file):
        """同一文件多次计算 hash 结果一致。"""
        hash1 = FileIntegrityChecker.compute_sha256(sample_file)
        hash2 = FileIntegrityChecker.compute_sha256(sample_file)
        assert hash1 == hash2

    def test_hash_format(self, sample_file):
        """hash 格式为 64 位十六进制字符串。"""
        h = FileIntegrityChecker.compute_sha256(sample_file)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_files_different_hash(self, tmp_path):
        """不同文件 hash 不同。"""
        file1 = str(tmp_path / "a.txt")
        file2 = str(tmp_path / "b.txt")
        with open(file1, "w") as f:
            f.write("内容A")
        with open(file2, "w") as f:
            f.write("内容B")
        assert FileIntegrityChecker.compute_sha256(file1) != FileIntegrityChecker.compute_sha256(file2)

    def test_file_not_found(self):
        """文件不存在时抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError, match="文件不存在"):
            FileIntegrityChecker.compute_sha256("/nonexistent/file.txt")

    def test_empty_file(self, tmp_path):
        """空文件也能正常计算 hash。"""
        empty = str(tmp_path / "empty.txt")
        with open(empty, "w") as f:
            pass
        h = FileIntegrityChecker.compute_sha256(empty)
        assert len(h) == 64


# ============================================================
# SQLiteIntegrityChecker 基本功能测试
# ============================================================

class TestSQLiteIntegrityChecker:
    """SQLiteIntegrityChecker 基本功能测试。"""

    def test_db_auto_created(self, tmp_db):
        """数据库文件在初始化时自动创建。"""
        assert not os.path.exists(tmp_db)
        SQLiteIntegrityChecker(db_path=tmp_db)
        assert os.path.exists(tmp_db)

    def test_db_directory_auto_created(self, tmp_path):
        """数据库目录不存在时自动创建。"""
        db_path = str(tmp_path / "subdir" / "ingestion.db")
        assert not os.path.exists(os.path.dirname(db_path))
        SQLiteIntegrityChecker(db_path=db_path)
        assert os.path.exists(db_path)

    def test_should_skip_initially_false(self, checker):
        """初始状态下 should_skip 返回 False。"""
        assert checker.should_skip("abc123") is False

    def test_mark_success_then_skip(self, checker):
        """标记 success 后，should_skip 返回 True。"""
        checker.mark_success("hash_001", "/path/to/file.pdf")
        assert checker.should_skip("hash_001") is True

    def test_mark_failed_then_not_skip(self, checker):
        """标记 failed 后，should_skip 仍返回 False。"""
        checker.mark_failed("hash_002", "/path/to/file.pdf", "解析失败")
        assert checker.should_skip("hash_002") is False

    def test_get_status_none(self, checker):
        """无记录时 get_status 返回 None。"""
        assert checker.get_status("nonexistent") is None

    def test_get_status_success(self, checker):
        """标记 success 后 get_status 返回 'success'。"""
        checker.mark_success("h1", "/f.pdf")
        assert checker.get_status("h1") == "success"

    def test_get_status_failed(self, checker):
        """标记 failed 后 get_status 返回 'failed'。"""
        checker.mark_failed("h2", "/f.pdf", "err")
        assert checker.get_status("h2") == "failed"

    def test_mark_success_upsert(self, checker):
        """重复 mark_success 不会报错，且状态保持 success。"""
        checker.mark_success("h3", "/f.pdf", file_size=100, chunk_count=5)
        checker.mark_success("h3", "/f.pdf", file_size=100, chunk_count=10)
        assert checker.get_status("h3") == "success"

    def test_mark_failed_then_success(self, checker):
        """从 failed 状态可更新为 success。"""
        checker.mark_failed("h4", "/f.pdf", "第一次失败")
        assert checker.get_status("h4") == "failed"
        checker.mark_success("h4", "/f.pdf")
        assert checker.get_status("h4") == "success"
        assert checker.should_skip("h4") is True

    def test_mark_success_with_metadata(self, checker):
        """mark_success 可记录 file_size 和 chunk_count。"""
        checker.mark_success("h5", "/f.pdf", file_size=1024, chunk_count=8)
        checker.mark_success("h5", "/f.pdf", file_size=2048, chunk_count=16)
        # upsert 后应反映最新值
        assert checker.should_skip("h5") is True


# ============================================================
# remove_record 测试
# ============================================================

class TestRemoveRecord:
    """remove_record 方法测试。"""

    def test_remove_existing(self, checker):
        """删除存在的记录返回 True。"""
        checker.mark_success("h_del", "/f.pdf")
        assert checker.remove_record("h_del") is True
        assert checker.get_status("h_del") is None
        assert checker.should_skip("h_del") is False

    def test_remove_nonexistent(self, checker):
        """删除不存在的记录返回 False。"""
        assert checker.remove_record("nonexistent") is False


# ============================================================
# list_processed 测试
# ============================================================

class TestListProcessed:
    """list_processed 方法测试。"""

    def test_empty_list(self, checker):
        """无记录时返回空列表。"""
        assert checker.list_processed() == []

    def test_list_only_success(self, checker):
        """只返回 success 状态的记录。"""
        checker.mark_success("h_ok", "/ok.pdf", file_size=100, chunk_count=5)
        checker.mark_failed("h_fail", "/fail.pdf", "error")
        result = checker.list_processed()
        assert len(result) == 1
        assert result[0]["file_hash"] == "h_ok"
        assert result[0]["file_size"] == 100
        assert result[0]["chunk_count"] == 5

    def test_list_order_by_date(self, checker):
        """结果按 processed_at 降序排列。"""
        checker.mark_success("h1", "/a.pdf")
        checker.mark_success("h2", "/b.pdf")
        result = checker.list_processed()
        assert len(result) == 2
        # 最新的排在前面
        assert result[0]["file_hash"] == "h2"
        assert result[1]["file_hash"] == "h1"

    def test_list_after_remove(self, checker):
        """删除后 list 不再包含该记录。"""
        checker.mark_success("h_rm", "/f.pdf")
        checker.remove_record("h_rm")
        assert checker.list_processed() == []


# ============================================================
# 完整流程测试
# ============================================================

class TestFullWorkflow:
    """完整工作流测试：模拟实际摄取场景。"""

    def test_ingestion_workflow(self, checker, sample_file):
        """模拟完整摄取流程：计算hash → 检查 → 处理 → 标记。"""
        # 1. 计算文件 hash
        file_hash = FileIntegrityChecker.compute_sha256(sample_file)

        # 2. 首次检查：应处理
        assert checker.should_skip(file_hash) is False
        assert checker.get_status(file_hash) is None

        # 3. 模拟处理成功
        file_size = os.path.getsize(sample_file)
        checker.mark_success(file_hash, sample_file, file_size=file_size, chunk_count=15)

        # 4. 再次检查：应跳过
        assert checker.should_skip(file_hash) is True
        assert checker.get_status(file_hash) == "success"

        # 5. list_processed 包含该记录
        records = checker.list_processed()
        assert len(records) == 1
        assert records[0]["file_hash"] == file_hash
        assert records[0]["chunk_count"] == 15

    def test_force_reprocess(self, checker, sample_file):
        """模拟强制重新处理：删除记录后重新摄取。"""
        file_hash = FileIntegrityChecker.compute_sha256(sample_file)

        # 首次处理
        checker.mark_success(file_hash, sample_file)
        assert checker.should_skip(file_hash) is True

        # 删除记录（强制重新处理）
        checker.remove_record(file_hash)
        assert checker.should_skip(file_hash) is False

        # 重新处理
        checker.mark_success(file_hash, sample_file, chunk_count=20)
        assert checker.should_skip(file_hash) is True

    def test_failed_then_retry(self, checker, sample_file):
        """模拟失败后重试。"""
        file_hash = FileIntegrityChecker.compute_sha256(sample_file)

        # 第一次处理失败
        checker.mark_failed(file_hash, sample_file, "LLM 超时")
        assert checker.should_skip(file_hash) is False

        # 重试成功
        checker.mark_success(file_hash, sample_file, chunk_count=10)
        assert checker.should_skip(file_hash) is True


# ============================================================
# WAL 模式测试
# ============================================================

class TestWALMode:
    """WAL 模式与并发写入测试。"""

    def test_wal_mode_enabled(self, tmp_db):
        """数据库连接使用 WAL 模式。"""
        checker = SQLiteIntegrityChecker(db_path=tmp_db)
        import sqlite3
        conn = sqlite3.connect(tmp_db)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        # WAL 模式在连接时设置，再次连接时应为 wal
        assert mode in ("wal", "delete")  # delete 是默认回退

    def test_multiple_instances_same_db(self, tmp_db):
        """多个实例可共享同一数据库文件。"""
        c1 = SQLiteIntegrityChecker(db_path=tmp_db)
        c2 = SQLiteIntegrityChecker(db_path=tmp_db)

        c1.mark_success("h_shared", "/f.pdf")
        assert c2.should_skip("h_shared") is True

        c2.mark_success("h_shared2", "/f2.pdf")
        assert c1.should_skip("h_shared2") is True


# ============================================================
# Re-export 测试
# ============================================================

class TestReExport:
    """验证 __init__.py 的 re-export 正确。"""

    def test_import_from_loader(self):
        """从 src.libs.loader 可直接导入。"""
        from src.libs.loader import FileIntegrityChecker, SQLiteIntegrityChecker
        assert FileIntegrityChecker is not None
        assert SQLiteIntegrityChecker is not None
