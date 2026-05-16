"""C13 ImageStorage 手动测试脚本。

测试图片文件存储、SQLite 索引、集合查询、文档级清理等功能。
使用临时目录隔离测试环境。
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.storage.image_storage import ImageStorage


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def safe_print(text: str):
    """安全打印，忽略无法编码的字符。"""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("gbk", errors="replace").decode("gbk", errors="replace"))


def _make_image_data(seed: int = 0) -> bytes:
    """生成模拟图片数据。"""
    return bytes([seed % 256] * 200)


def _make_storage(tmp: str) -> ImageStorage:
    """在临时目录中创建 ImageStorage。"""
    return ImageStorage(
        db_path=os.path.join(tmp, "db", "image_index.db"),
        image_dir=os.path.join(tmp, "images"),
    )


def test_save_and_lookup():
    """测试保存与查找。"""
    section("保存与查找")
    tmp = tempfile.mkdtemp()
    try:
        storage = _make_storage(tmp)

        # 保存 3 张图片
        for i in range(3):
            path = storage.save_image(
                f"img_{i:03d}",
                _make_image_data(i),
                collection="test_docs",
                doc_hash="abc123",
                page_num=i + 1,
            )
            safe_print(f"  保存 img_{i:03d} -> {os.path.basename(path)}")

        # 查找
        found = storage.get_image_path("img_001")
        safe_print(f"\n  查找 img_001: {os.path.basename(found)}")

        info = storage.get_image_info("img_002")
        safe_print(f"  img_002 详情: collection={info['collection']}, page={info['page_num']}")

        # 统计
        safe_print(f"\n  总数: {storage.count()}")
        safe_print(f"  test_docs 集合: {storage.count(collection='test_docs')}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_collection_operations():
    """测试集合操作。"""
    section("集合操作")
    tmp = tempfile.mkdtemp()
    try:
        storage = _make_storage(tmp)

        # 写入不同集合
        storage.save_image("img_a1", _make_image_data(1), collection="alpha")
        storage.save_image("img_a2", _make_image_data(2), collection="alpha")
        storage.save_image("img_b1", _make_image_data(3), collection="beta")
        storage.save_image("img_b2", _make_image_data(4), collection="beta")
        storage.save_image("img_c1", _make_image_data(5), collection="gamma")

        # 列出集合
        collections = storage.list_collections()
        safe_print(f"  集合列表: {collections}")

        # 按集合查询
        alpha_images = storage.list_by_collection("alpha")
        safe_print(f"  alpha 集合: {len(alpha_images)} 张")

        beta_images = storage.list_by_collection("beta")
        safe_print(f"  beta 集合: {len(beta_images)} 张")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_doc_hash_operations():
    """测试按文档哈希操作。"""
    section("文档哈希操作")
    tmp = tempfile.mkdtemp()
    try:
        storage = _make_storage(tmp)

        # 同一文档的多张图片
        storage.save_image("img_001", _make_image_data(1), doc_hash="doc_a", page_num=1)
        storage.save_image("img_002", _make_image_data(2), doc_hash="doc_a", page_num=2)
        storage.save_image("img_003", _make_image_data(3), doc_hash="doc_a", page_num=3)
        storage.save_image("img_004", _make_image_data(4), doc_hash="doc_b", page_num=1)

        # 按文档查询
        doc_a_images = storage.list_by_doc_hash("doc_a")
        safe_print(f"  doc_a 图片: {len(doc_a_images)} 张")
        for img in doc_a_images:
            safe_print(f"    {img['image_id']} (page {img['page_num']})")

        # 按文档删除
        deleted = storage.delete_by_doc_hash("doc_a")
        safe_print(f"\n  删除 doc_a: {deleted} 张")
        safe_print(f"  剩余: {storage.count()} 张")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_delete_operations():
    """测试删除操作。"""
    section("删除操作")
    tmp = tempfile.mkdtemp()
    try:
        storage = _make_storage(tmp)

        storage.save_image("img_001", _make_image_data(1), collection="test")
        storage.save_image("img_002", _make_image_data(2), collection="test")
        safe_print(f"  保存后: {storage.count()} 张")

        # 删除单张
        deleted = storage.delete_image("img_001")
        safe_print(f"  删除 img_001: {deleted}")
        safe_print(f"  剩余: {storage.count()} 张")

        # 查找已删除
        path = storage.get_image_path("img_001")
        safe_print(f"  查找已删除: {path}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_persistence():
    """测试数据持久化。"""
    section("数据持久化")
    tmp = tempfile.mkdtemp()
    try:
        db_path = os.path.join(tmp, "db", "image_index.db")
        image_dir = os.path.join(tmp, "images")

        # 第一次写入
        storage1 = ImageStorage(db_path=db_path, image_dir=image_dir)
        storage1.save_image("img_001", _make_image_data(1), collection="test")
        safe_print(f"  实例1 写入: {storage1.count()} 张")

        # 第二次打开（storage1 的连接会被 GC 回收）
        storage2 = ImageStorage(db_path=db_path, image_dir=image_dir)
        safe_print(f"  实例2 读取: {storage2.count()} 张")
        safe_print(f"  路径: {storage2.get_image_path('img_001')}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_save_and_lookup()
    test_collection_operations()
    test_doc_hash_operations()
    test_delete_operations()
    test_persistence()

    print(f"\n{'='*60}")
    print("  全部手动测试完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
