"""ImageStorage 模块。

保存图片文件到 data/images/{collection}/ 目录，
并使用 SQLite 记录 image_id → file_path 映射关系。
支持按 collection 批量查询、文档级清理等操作。
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 默认路径
_DEFAULT_DB_PATH = os.path.join("data", "db", "image_index.db")
_DEFAULT_IMAGE_DIR = os.path.join("data", "images")


class ImageStorage:
    """图片文件存储与 SQLite 索引管理。

    核心职责：
    1. 保存图片文件到 data/images/{collection}/
    2. 在 SQLite 中记录 image_id → file_path 映射
    3. 支持按 collection / doc_hash 批量查询
    4. 支持图片删除（文件 + 索引）

    属性:
        db_path: SQLite 数据库路径
        image_dir: 图片根目录
    """

    def __init__(
        self,
        db_path: str = _DEFAULT_DB_PATH,
        image_dir: str = _DEFAULT_IMAGE_DIR,
    ) -> None:
        """初始化 ImageStorage。

        参数:
            db_path: SQLite 数据库文件路径
            image_dir: 图片根目录路径
        """
        self.db_path = db_path
        self.image_dir = image_dir

        # 确保目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)

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
                CREATE TABLE IF NOT EXISTS image_index (
                    image_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    collection TEXT,
                    doc_hash TEXT,
                    page_num INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_collection
                ON image_index(collection)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_hash
                ON image_index(doc_hash)
            """)

    def save_image(
        self,
        image_id: str,
        image_data: bytes,
        collection: str = "default",
        doc_hash: str = "",
        page_num: int = 0,
    ) -> str:
        """保存图片文件并记录索引。

        参数:
            image_id: 图片唯一标识符
            image_data: 图片二进制数据
            collection: 集合名称（用于目录组织）
            doc_hash: 来源文档哈希（可选）
            page_num: 图片所在页码（可选）

        返回:
            图片保存后的文件路径
        """
        # 构建保存路径: data/images/{collection}/{image_id}.png
        collection_dir = os.path.join(self.image_dir, collection)
        os.makedirs(collection_dir, exist_ok=True)

        file_path = os.path.join(collection_dir, f"{image_id}.png")

        # 写入文件
        with open(file_path, "wb") as f:
            f.write(image_data)

        # 写入 SQLite 索引
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO image_index (image_id, file_path, collection, doc_hash, page_num, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_id) DO UPDATE SET
                    file_path = excluded.file_path,
                    collection = excluded.collection,
                    doc_hash = excluded.doc_hash,
                    page_num = excluded.page_num,
                    created_at = excluded.created_at
                """,
                (image_id, file_path, collection, doc_hash, page_num, now),
            )

        logger.debug("图片已保存: %s -> %s", image_id, file_path)
        return file_path

    def get_image_path(self, image_id: str) -> Optional[str]:
        """根据 image_id 查找图片文件路径。

        参数:
            image_id: 图片唯一标识符

        返回:
            文件路径，不存在时返回 None
        """
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT file_path FROM image_index WHERE image_id = ?",
                (image_id,),
            ).fetchone()
            if row is None:
                return None

            file_path = row["file_path"]
            # 验证文件实际存在
            if not os.path.isfile(file_path):
                logger.warning("索引存在但文件缺失: %s", file_path)
                return None
            return file_path

    def get_image_info(self, image_id: str) -> Optional[Dict[str, Any]]:
        """获取图片完整信息。

        参数:
            image_id: 图片唯一标识符

        返回:
            图片信息字典，不存在时返回 None
        """
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM image_index WHERE image_id = ?",
                (image_id,),
            ).fetchone()
            return dict(row) if row else None

    def list_by_collection(self, collection: str) -> List[Dict[str, Any]]:
        """列出指定集合下的所有图片。

        参数:
            collection: 集合名称

        返回:
            图片信息字典列表
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM image_index WHERE collection = ? ORDER BY created_at",
                (collection,),
            ).fetchall()
            return [dict(row) for row in rows]

    def list_by_doc_hash(self, doc_hash: str) -> List[Dict[str, Any]]:
        """列出指定文档的所有图片。

        参数:
            doc_hash: 文档哈希值

        返回:
            图片信息字典列表
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM image_index WHERE doc_hash = ? ORDER BY page_num",
                (doc_hash,),
            ).fetchall()
            return [dict(row) for row in rows]

    def delete_image(self, image_id: str) -> bool:
        """删除图片（文件 + 索引）。

        参数:
            image_id: 图片唯一标识符

        返回:
            是否成功删除
        """
        # 先查路径
        file_path = self.get_image_path(image_id)

        # 删除索引
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM image_index WHERE image_id = ?",
                (image_id,),
            )
            deleted = cursor.rowcount > 0

        # 删除文件
        if file_path and os.path.isfile(file_path):
            try:
                os.remove(file_path)
                logger.debug("图片文件已删除: %s", file_path)
            except OSError as e:
                logger.warning("图片文件删除失败: %s, 错误: %s", file_path, e)

        return deleted

    def delete_by_doc_hash(self, doc_hash: str) -> int:
        """删除指定文档的所有图片。

        参数:
            doc_hash: 文档哈希值

        返回:
            成功删除的记录数
        """
        # 先查出所有文件路径
        images = self.list_by_doc_hash(doc_hash)
        file_paths = [img["file_path"] for img in images]

        # 删除索引
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM image_index WHERE doc_hash = ?",
                (doc_hash,),
            )
            deleted = cursor.rowcount

        # 删除文件
        for fp in file_paths:
            if os.path.isfile(fp):
                try:
                    os.remove(fp)
                except OSError as e:
                    logger.warning("图片文件删除失败: %s, 错误: %s", fp, e)

        logger.info("已删除文档 %s 的 %d 张图片", doc_hash, deleted)
        return deleted

    def count(self, collection: Optional[str] = None) -> int:
        """统计图片数量。

        参数:
            collection: 集合名称（可选，不传则统计全部）

        返回:
            图片数量
        """
        with self._get_conn() as conn:
            if collection:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM image_index WHERE collection = ?",
                    (collection,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM image_index"
                ).fetchone()
            return row["cnt"]

    def list_collections(self) -> List[str]:
        """列出所有集合名称。

        返回:
            集合名称列表（去重、排序）
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT collection FROM image_index ORDER BY collection"
            ).fetchall()
            return [row["collection"] for row in rows]
