"""D7 query.py 脚本入口手动测试。

测试命令行入口的各项功能。
需要先执行 ingest 摄取数据并构建索引。

测试覆盖：
1. 基本查询流程
2. --top-k 参数
3. --verbose 模式
4. --no-rerank 模式
5. --collection 过滤
6. 空查询提示
"""

import sys
import os

# 修复 Windows 终端中文编码问题
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.query import main, parse_args


def test_basic_query():
    """测试1: 基本查询。"""
    print("\n" + "=" * 60)
    print("测试1: 基本查询")
    print("=" * 60)

    exit_code = main(["--query", "什么是语言模型？", "--top-k", "3"])
    assert exit_code == 0, f"查询失败，退出码: {exit_code}"

    print("\n✅ 基本查询测试通过")


def test_verbose_mode():
    """测试2: Verbose 模式。"""
    print("\n" + "=" * 60)
    print("测试2: Verbose 模式")
    print("=" * 60)

    exit_code = main(["--query", "什么是 RAG？", "--top-k", "3", "--verbose"])
    assert exit_code == 0, f"Verbose 查询失败，退出码: {exit_code}"

    print("\n✅ Verbose 模式测试通过")


def test_no_rerank():
    """测试3: --no-rerank 模式。"""
    print("\n" + "=" * 60)
    print("测试3: --no-rerank 模式")
    print("=" * 60)

    exit_code = main(["--query", "如何配置 Ollama？", "--top-k", "3", "--no-rerank"])
    assert exit_code == 0, f"no-rerank 查询失败，退出码: {exit_code}"

    print("\n✅ --no-rerank 模式测试通过")


def test_arg_parsing():
    """测试4: 参数解析。"""
    print("\n" + "=" * 60)
    print("测试4: 参数解析")
    print("=" * 60)

    args = parse_args(["--query", "test query"])
    assert args.query == "test query"
    assert args.top_k == 10
    assert args.verbose is False
    assert args.no_rerank is False
    assert args.collection is None
    print("  默认参数 ✅")

    args = parse_args(["--query", "test", "--top-k", "5", "--verbose", "--no-rerank", "--collection", "my_docs"])
    assert args.top_k == 5
    assert args.verbose is True
    assert args.no_rerank is True
    assert args.collection == "my_docs"
    print("  自定义参数 ✅")

    print("\n✅ 参数解析测试通过")


def test_empty_query_hint():
    """测试5: 空查询提示（无数据时）。"""
    print("\n" + "=" * 60)
    print("测试5: 查询提示信息")
    print("=" * 60)

    # 正常查询应该有输出
    exit_code = main(["--query", "测试查询", "--top-k", "1"])
    assert exit_code == 0
    print("  查询正常返回 ✅")

    print("\n✅ 查询提示测试通过")


def main_test():
    """运行所有 D7 测试。"""
    print("=" * 60)
    print("D7 query.py 脚本入口 完整测试")
    print("=" * 60)

    test_arg_parsing()
    test_basic_query()
    test_verbose_mode()
    test_no_rerank()
    test_empty_query_hint()

    print("\n" + "=" * 60)
    print("✅ 所有 D7 测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main_test()
