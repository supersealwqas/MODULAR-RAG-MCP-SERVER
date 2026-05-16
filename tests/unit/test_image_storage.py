"""ImageStorage 单元测试。

测试图片文件保存、SQLite 索引、查询、删除等功能。
使用临时目录隔离测试环境。
"""

from __future__ import annotations

import os
import tempfile

import pytest

from src.ingestion.storage.image_storage import ImageStorage


# ============================================================
# 辅助工具
# ============================================================


def _make_image_data(seed: int = 0) -> bytes:
    """生成模拟图片数据（不同 seed 产生不同内容）。"""
    return bytes([seed % 256] * 100)


def _create_storage(tmp_path) -> ImageStorage:
    """在临时目录中创建 ImageStorage 实例。"""
    db_path = os.path.join(str(tmp_path), "db", "image_index.db")
    image_dir = os.path.join(str(tmp_path), "images")
    return ImageStorage(db_path=db_path, image_dir=image_dir)


# ============================================================
# 测试用例
# ============================================================


class TestImageStorageSave:
    """图片保存功能测试。"""

    def test_save_creates_file(self, tmp_path):
        """save_image 应创建图片文件。"""
        storage = _create_storage(tmp_path)
        data = _make_image_data(1)

        file_path = storage.save_image("img_001", data, collection="test")

        assert os.path.isfile(file_path)
        assert file_path.endswith("img_001.png")

    def test_save_creates_collection_dir(self, tmp_path):
        """save_image 应自动创建集合目录。"""
        storage = _create_storage(tmp_path)
        data = _make_image_data(1)

        file_path = storage.save_image("img_001", data, collection="my_collection")

        assert "my_collection" in file_path
        assert os.path.isdir(os.path.dirname(file_path))

    def test_save_writes_correct_data(self, tmp_path):
        """保存的文件内容应与输入一致。"""
        storage = _create_storage(tmp_path)
        data = _make_image_data(42)

        file_path = storage.save_image("img_001", data, collection="test")

        with open(file_path, "rb") as f:
            assert f.read() == data

    def test_save_returns_path(self, tmp_path):
        """save_image 应返回文件路径。"""
        storage = _create_storage(tmp_path)
        data = _make_image_data(1)

        file_path = storage.save_image("img_001", data, collection="test")

        assert isinstance(file_path, str)
        assert len(file_path) > 0

    def test_save_with_metadata(self, tmp_path):
        """保存时应记录 doc_hash 和 page_num。"""
        storage = _create_storage(tmp_path)
        data = _make_image_data(1)

        storage.save_image(
            "img_001", data, collection="test",
            doc_hash="abc123", page_num=3,
        )

        info = storage.get_image_info("img_001")
        assert info is not None
        assert info["doc_hash"] == "abc123"
        assert info["page_num"] == 3

    def test_save_overwrite_same_id(self, tmp_path):
        """相同 image_id 重复保存应覆盖。"""
        storage = _create_storage(tmp_path)

        storage.save_image("img_001", _make_image_data(1), collection="test")
        storage.save_image("img_001", _make_image_data(2), collection="test")

        assert storage.count() == 1
        info = storage.get_image_info("img_001")
        assert info is not None


class TestImageStorageLookup:
    """图片查询功能测试。"""

    def test_get_existing_image_path(self, tmp_path):
        """查找已存在的图片应返回正确路径。"""
        storage = _create_storage(tmp_path)
        data = _make_image_data(1)

        saved_path = storage.save_image("img_001", data, collection="test")
        found_path = storage.get_image_path("img_001")

        assert found_path == saved_path

    def test_get_nonexistent_returns_none(self, tmp_path):
        """查找不存在的图片应返回 None。"""
        storage = _create_storage(tmp_path)

        result = storage.get_image_path("nonexistent")

        assert result is None

    def test_get_image_info(self, tmp_path):
        """get_image_info 应返回完整信息。"""
        storage = _create_storage(tmp_path)
        data = _make_image_data(1)

        storage.save_image(
            "img_001", data, collection="test",
            doc_hash="abc123", page_num=5,
        )

        info = storage.get_image_info("img_001")
        assert info is not None
        assert info["image_id"] == "img_001"
        assert info["collection"] == "test"
        assert info["doc_hash"] == "abc123"
        assert info["page_num"] == 5

    def test_get_info_nonexistent_returns_none(self, tmp_path):
        """查找不存在的图片信息应返回 None。"""
        storage = _create_storage(tmp_path)

        result = storage.get_image_info("nonexistent")

        assert result is None


class TestImageStorageListByCollection:
    """按集合查询测试。"""

    def test_list_by_collection(self, tmp_path):
        """list_by_collection 应返回指定集合的所有图片。"""
        storage = _create_storage(tmp_path)

        storage.save_image("img_001", _make_image_data(1), collection="col_a")
        storage.save_image("img_002", _make_image_data(2), collection="col_a")
        storage.save_image("img_003", _make_image_data(3), collection="col_b")

        result = storage.list_by_collection("col_a")

        assert len(result) == 2
        ids = {r["image_id"] for r in result}
        assert ids == {"img_001", "img_002"}

    def test_list_empty_collection(self, tmp_path):
        """查询空集合应返回空列表。"""
        storage = _create_storage(tmp_path)

        result = storage.list_by_collection("empty")

        assert result == []


class TestImageStorageListByDocHash:
    """按文档哈希查询测试。"""

    def test_list_by_doc_hash(self, tmp_path):
        """list_by_doc_hash 应返回指定文档的所有图片。"""
        storage = _create_storage(tmp_path)

        storage.save_image("img_001", _make_image_data(1), doc_hash="doc_a", page_num=1)
        storage.save_image("img_002", _make_image_data(2), doc_hash="doc_a", page_num=2)
        storage.save_image("img_003", _make_image_data(3), doc_hash="doc_b", page_num=1)

        result = storage.list_by_doc_hash("doc_a")

        assert len(result) == 2
        # 应按 page_num 排序
        assert result[0]["page_num"] == 1
        assert result[1]["page_num"] == 2


class TestImageStorageDelete:
    """图片删除测试。"""

    def test_delete_removes_file_and_index(self, tmp_path):
        """delete_image 应同时删除文件和索引。"""
        storage = _create_storage(tmp_path)
        data = _make_image_data(1)

        file_path = storage.save_image("img_001", data, collection="test")
        assert os.path.isfile(file_path)

        deleted = storage.delete_image("img_001")

        assert deleted is True
        assert not os.path.isfile(file_path)
        assert storage.get_image_path("img_001") is None

    def test_delete_nonexistent_returns_false(self, tmp_path):
        """删除不存在的图片应返回 False。"""
        storage = _create_storage(tmp_path)

        deleted = storage.delete_image("nonexistent")

        assert deleted is False

    def test_delete_by_doc_hash(self, tmp_path):
        """delete_by_doc_hash 应删除指定文档的所有图片。"""
        storage = _create_storage(tmp_path)

        storage.save_image("img_001", _make_image_data(1), doc_hash="doc_a")
        storage.save_image("img_002", _make_image_data(2), doc_hash="doc_a")
        storage.save_image("img_003", _make_image_data(3), doc_hash="doc_b")

        deleted = storage.delete_by_doc_hash("doc_a")

        assert deleted == 2
        assert storage.count() == 1
        assert storage.get_image_path("img_003") is not None


class TestImageStorageCount:
    """计数功能测试。"""

    def test_count_all(self, tmp_path):
        """count() 不传参数应统计全部。"""
        storage = _create_storage(tmp_path)

        storage.save_image("img_001", _make_image_data(1), collection="a")
        storage.save_image("img_002", _make_image_data(2), collection="b")

        assert storage.count() == 2

    def test_count_by_collection(self, tmp_path):
        """count(collection=) 应只统计指定集合。"""
        storage = _create_storage(tmp_path)

        storage.save_image("img_001", _make_image_data(1), collection="a")
        storage.save_image("img_002", _make_image_data(2), collection="a")
        storage.save_image("img_003", _make_image_data(3), collection="b")

        assert storage.count(collection="a") == 2
        assert storage.count(collection="b") == 1

    def test_count_empty(self, tmp_path):
        """空存储应返回 0。"""
        storage = _create_storage(tmp_path)

        assert storage.count() == 0


class TestImageStorageCollections:
    """集合列表测试。"""

    def test_list_collections(self, tmp_path):
        """list_collections 应返回所有集合名称。"""
        storage = _create_storage(tmp_path)

        storage.save_image("img_001", _make_image_data(1), collection="beta")
        storage.save_image("img_002", _make_image_data(2), collection="alpha")
        storage.save_image("img_003", _make_image_data(3), collection="beta")

        collections = storage.list_collections()

        assert collections == ["alpha", "beta"]

    def test_list_collections_empty(self, tmp_path):
        """空存储应返回空列表。"""
        storage = _create_storage(tmp_path)

        assert storage.list_collections() == []


class TestImageStoragePersistence:
    """SQLite 持久化测试。"""

    def test_data_persists_across_instances(self, tmp_path):
        """关闭后重新打开应保留数据。"""
        db_path = os.path.join(str(tmp_path), "db", "image_index.db")
        image_dir = os.path.join(str(tmp_path), "images")

        # 第一次写入
        storage1 = ImageStorage(db_path=db_path, image_dir=image_dir)
        storage1.save_image("img_001", _make_image_data(1), collection="test")

        # 第二次打开
        storage2 = ImageStorage(db_path=db_path, image_dir=image_dir)

        assert storage2.count() == 1
        assert storage2.get_image_path("img_001") is not None


class TestImageStorageMissingFile:
    """文件缺失场景测试。"""

    def test_get_path_returns_none_when_file_missing(self, tmp_path):
        """索引存在但文件缺失时应返回 None。"""
        storage = _create_storage(tmp_path)
        data = _make_image_data(1)

        file_path = storage.save_image("img_001", data, collection="test")

        # 手动删除文件（保留索引）
        os.remove(file_path)

        result = storage.get_image_path("img_001")
        assert result is None
