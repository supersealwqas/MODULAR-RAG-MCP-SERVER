"""文件完整性检查模块。

通过 SHA256 哈希判断文件是否已处理，实现增量摄取的零成本跳过。
默认使用 SQLite 作为持久化存储，支持替换为 Redis/PostgreSQL。
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional


# ============================================================
# 抽象接口
# ============================================================

class FileIntegrityChecker(ABC):
    """文件完整性检查器的抽象基类。

    子类必须实现 should_skip / mark_success / mark_failed 方法。
    """

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """判断指定 hash 的文件是否应跳过（已成功处理）。

        参数:
            file_hash: 文件的 SHA256 哈希值

        返回:
            True 表示应跳过（已处理），False 表示需处理
        """
        ...

    @abstractmethod
    def mark_success(
        self,
        file_hash: str,
        file_path: str,
        file_size: int = 0,
        chunk_count: int = 0,
    ) -> None:
        """标记文件处理成功。

        参数:
            file_hash: 文件的 SHA256 哈希值
            file_path: 文件路径
            file_size: 文件大小（字节）
            chunk_count: 产出的 chunk 数量
        """
        ...

    @abstractmethod
    def mark_failed(self, file_hash: str, file_path: str, error_msg: str) -> None:
        """标记文件处理失败。

        参数:
            file_hash: 文件的 SHA256 哈希值
            file_path: 文件路径
            error_msg: 错误信息
        """
        ...

    @abstractmethod
    def get_status(self, file_hash: str) -> Optional[str]:
        """获取指定 hash 的处理状态。

        参数:
            file_hash: 文件的 SHA256 哈希值

        返回:
            状态字符串 ('success'/'failed'/'processing') 或 None（无记录）
        """
        ...

    @abstractmethod
    def remove_record(self, file_hash: str) -> bool:
        """删除指定 hash 的记录（供 DocumentManager 使用）。

        参数:
            file_hash: 文件的 SHA256 哈希值

        返回:
            True 表示删除成功，False 表示记录不存在
        """
        ...

    @abstractmethod
    def list_processed(self) -> list[dict]:
        """列出所有已成功处理的记录（供 DocumentManager 使用）。

        返回:
            记录字典列表，每条包含 file_hash/file_path/file_size/processed_at/chunk_count
        """
        ...

    @staticmethod
    def compute_sha256(path: str) -> str:
        """计算文件的 SHA256 哈希值。

        参数:
            path: 文件路径

        返回:
            十六进制哈希字符串

        异常:
            FileNotFoundError: 文件不存在
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"文件不存在: {path}")

        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


# ============================================================
# SQLite 默认实现
# ============================================================

class SQLiteIntegrityChecker(FileIntegrityChecker):
    """基于 SQLite 的文件完整性检查器。

    使用 WAL 模式支持并发写入，数据存储在 data/db/ingestion_history.db。
    """

    def __init__(self, db_path: str = "data/db/ingestion_history.db"):
        """初始化 SQLite 检查器。

        参数:
            db_path: SQLite 数据库文件路径
        """
        self.db_path = db_path
        # 确保目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接（WAL 模式）。"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """初始化数据库表结构。"""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_history (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_size INTEGER DEFAULT 0,
                    status TEXT NOT NULL CHECK(status IN ('success', 'failed', 'processing')),
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    error_msg TEXT,
                    chunk_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON ingestion_history(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_processed_at ON ingestion_history(processed_at)
            """)

    def should_skip(self, file_hash: str) -> bool:
        """判断文件是否应跳过（已成功处理）。"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT status FROM ingestion_history WHERE file_hash = ? AND status = 'success'",
                (file_hash,),
            ).fetchone()
            return row is not None

    def mark_success(
        self,
        file_hash: str,
        file_path: str,
        file_size: int = 0,
        chunk_count: int = 0,
    ) -> None:
        """标记文件处理成功（upsert 语义）。"""
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_history (file_hash, file_path, file_size, status, processed_at, chunk_count)
                VALUES (?, ?, ?, 'success', ?, ?)
                ON CONFLICT(file_hash) DO UPDATE SET
                    file_path = excluded.file_path,
                    file_size = excluded.file_size,
                    status = 'success',
                    processed_at = excluded.processed_at,
                    error_msg = NULL,
                    chunk_count = excluded.chunk_count
                """,
                (file_hash, file_path, file_size, now, chunk_count),
            )

    def mark_failed(self, file_hash: str, file_path: str, error_msg: str) -> None:
        """标记文件处理失败（upsert 语义）。"""
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_history (file_hash, file_path, file_size, status, processed_at, error_msg)
                VALUES (?, ?, 0, 'failed', ?, ?)
                ON CONFLICT(file_hash) DO UPDATE SET
                    file_path = excluded.file_path,
                    status = 'failed',
                    processed_at = excluded.processed_at,
                    error_msg = excluded.error_msg
                """,
                (file_hash, file_path, now, error_msg),
            )

    def get_status(self, file_hash: str) -> Optional[str]:
        """获取文件处理状态。"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT status FROM ingestion_history WHERE file_hash = ?",
                (file_hash,),
            ).fetchone()
            return row["status"] if row else None

    def remove_record(self, file_hash: str) -> bool:
        """删除指定记录。"""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM ingestion_history WHERE file_hash = ?",
                (file_hash,),
            )
            return cursor.rowcount > 0

    def list_processed(self) -> list[dict]:
        """列出所有已成功处理的记录。"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT file_hash, file_path, file_size, processed_at, chunk_count "
                "FROM ingestion_history WHERE status = 'success' ORDER BY processed_at DESC"
            ).fetchall()
            return [dict(row) for row in rows]
