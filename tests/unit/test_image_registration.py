"""Pipeline 图片注册测试。

验证 Pipeline 在 store 阶段后将图片注册到 ImageStorage SQLite 索引。
"""

import os
import tempfile

import pytest

from src.ingestion.storage.image_storage import ImageStorage


class TestImageStorageRegister:
    """ImageStorage.register_image 方法测试。"""

    def setup_method(self):
        """创建临时目录和 ImageStorage。"""
        self._tmp_dir = tempfile.mkdtemp()
        self._storage = ImageStorage(
            db_path=os.path.join(self._tmp_dir, "test_image.db"),
            image_dir=self._tmp_dir,
        )

    def teardown_method(self):
        """清理临时文件。"""
        import shutil
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def test_register_image_basic(self):
        """基本注册功能：注册后可通过 doc_hash 查询到。"""
        self._storage.register_image(
            image_id="img_001",
            file_path="/tmp/img_001.png",
            collection="default",
            doc_hash="abc123",
            page_num=1,
        )
        images = self._storage.list_by_doc_hash("abc123")
        assert len(images) == 1
        assert images[0]["image_id"] == "img_001"
        assert images[0]["file_path"] == "/tmp/img_001.png"
        assert images[0]["page_num"] == 1

    def test_register_image_upsert(self):
        """重复注册更新路径。"""
        self._storage.register_image(
            image_id="img_001",
            file_path="/tmp/old.png",
            collection="default",
            doc_hash="abc123",
        )
        self._storage.register_image(
            image_id="img_001",
            file_path="/tmp/new.png",
            collection="default",
            doc_hash="abc123",
        )
        images = self._storage.list_by_doc_hash("abc123")
        assert len(images) == 1
        assert images[0]["file_path"] == "/tmp/new.png"

    def test_register_image_list_by_doc_hash(self):
        """注册后可通过 doc_hash 查询。"""
        self._storage.register_image(
            image_id="img_001",
            file_path="/tmp/img_001.png",
            collection="default",
            doc_hash="hash_a",
            page_num=1,
        )
        self._storage.register_image(
            image_id="img_002",
            file_path="/tmp/img_002.png",
            collection="default",
            doc_hash="hash_a",
            page_num=2,
        )
        self._storage.register_image(
            image_id="img_003",
            file_path="/tmp/img_003.png",
            collection="other",
            doc_hash="hash_b",
        )

        images = self._storage.list_by_doc_hash("hash_a")
        assert len(images) == 2

    def test_register_multiple_collections(self):
        """不同集合的图片独立统计。"""
        self._storage.register_image(
            image_id="img_001",
            file_path="/tmp/img_001.png",
            collection="col_a",
            doc_hash="hash_a",
        )
        self._storage.register_image(
            image_id="img_002",
            file_path="/tmp/img_002.png",
            collection="col_b",
            doc_hash="hash_b",
        )

        assert len(self._storage.list_by_doc_hash("hash_a")) == 1
        assert len(self._storage.list_by_doc_hash("hash_b")) == 1
