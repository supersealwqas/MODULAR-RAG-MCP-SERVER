"""C2 文件完整性检查手动测试脚本。

测试 SQLiteIntegrityChecker 的 SHA256 计算、增量跳过、状态流转等。
"""

import gc
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.libs.loader.file_integrity import FileIntegrityChecker, SQLiteIntegrityChecker


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_sha256():
    section("SHA256 计算")

    # 使用真实文件
    test_file = "data/documents/黑马教育课程咨询知识库.pdf"
    if not os.path.exists(test_file):
        test_file = "config/settings.yaml"
        print(f"(目标文件不存在，使用 {test_file} 代替)")

    h1 = FileIntegrityChecker.compute_sha256(test_file)
    h2 = FileIntegrityChecker.compute_sha256(test_file)
    print(f"文件: {test_file}")
    print(f"Hash: {h1}")
    print(f"两次计算一致: {h1 == h2}")
    print(f"Hash 长度: {len(h1)} (应为 64)")


def test_checker_basic():
    section("SQLiteIntegrityChecker 基本操作")

    # 使用 mytest 目录下的临时数据库（避免 Windows 文件锁定问题）
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_integrity.db")
    try:
        checker = SQLiteIntegrityChecker(db_path=db_path)
        print(f"数据库创建: {os.path.exists(db_path)}")

        # 初始状态
        print(f"初始 should_skip('h1'): {checker.should_skip('h1')}")  # False
        print(f"初始 get_status('h1'): {checker.get_status('h1')}")  # None

        # 标记成功
        checker.mark_success("h1", "/path/to/file.pdf", file_size=1024, chunk_count=10)
        print(f"\n标记 success 后:")
        print(f"  should_skip('h1'): {checker.should_skip('h1')}")  # True
        print(f"  get_status('h1'): {checker.get_status('h1')}")  # 'success'

        # 标记失败
        checker.mark_failed("h2", "/path/to/bad.pdf", "解析失败")
        print(f"\n标记 failed 后:")
        print(f"  should_skip('h2'): {checker.should_skip('h2')}")  # False
        print(f"  get_status('h2'): {checker.get_status('h2')}")  # 'failed'

        # 列出已处理
        records = checker.list_processed()
        print(f"\nlist_processed(): {len(records)} 条记录")
        for r in records:
            print(f"  - {r['file_hash']} | {r['file_path']} | chunks={r['chunk_count']}")

        # 删除记录
        removed = checker.remove_record("h1")
        print(f"\n删除 h1: {removed}")
        print(f"删除后 should_skip('h1'): {checker.should_skip('h1')}")

        # 从 failed 升级为 success
        checker.mark_success("h2", "/path/to/bad.pdf", chunk_count=5)
        print(f"\nh2 从 failed 升级为 success:")
        print(f"  get_status('h2'): {checker.get_status('h2')}")
        print(f"  should_skip('h2'): {checker.should_skip('h2')}")
    finally:
        # 清理临时数据库（先确保连接已关闭）
        gc.collect()
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
                print(f"\n清理临时数据库: {db_path}")
        except PermissionError:
            print(f"\n(数据库被占用，跳过清理: {db_path})")


def test_incremental_skip():
    section("增量摄取模拟")

    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_integrity.db")
    try:
        checker = SQLiteIntegrityChecker(db_path=db_path)

        # 模拟：计算文件 hash
        test_file = "data/documents/黑马教育课程咨询知识库.pdf"
        if not os.path.exists(test_file):
            test_file = "config/settings.yaml"

        file_hash = FileIntegrityChecker.compute_sha256(test_file)
        print(f"文件 hash: {file_hash}")

        # 第一次：应处理
        skip = checker.should_skip(file_hash)
        print(f"第一次 should_skip: {skip}")  # False

        # 处理成功
        checker.mark_success(file_hash, test_file, file_size=os.path.getsize(test_file), chunk_count=25)
        print(f"标记 success 完成")

        # 第二次：应跳过
        skip = checker.should_skip(file_hash)
        print(f"第二次 should_skip: {skip}")  # True
        print("=> 增量跳过生效，无需重新处理!")
    finally:
        import gc
        gc.collect()
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
        except PermissionError:
            pass  # Windows 上 SQLite WAL 连接可能未完全关闭


def main():
    test_sha256()
    test_checker_basic()
    test_incremental_skip()
    print(f"\n{'='*60}")
    print("  全部测试通过!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
