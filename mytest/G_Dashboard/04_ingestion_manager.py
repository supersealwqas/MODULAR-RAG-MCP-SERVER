"""手动测试 G4: Dashboard Ingestion 管理页面。

用法:
    uv run python mytest/G_Dashboard/04_ingestion_manager.py

验证项:
    1. ingestion_manager 页面可导入
    2. app.py 中 page_ingestion_manager 已接入真实页面
    3. 辅助函数正确性
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_page_import():
    """测试页面可导入。"""
    print("=" * 50)
    print("测试 1: ingestion_manager 页面可导入")
    print("=" * 50)

    from src.observability.dashboard.pages.ingestion_manager import (
        render_ingestion_manager,
    )
    print(f"  render_ingestion_manager: {render_ingestion_manager}")
    assert callable(render_ingestion_manager)

    print("\n✅ 页面导入测试通过\n")


def test_app_wiring():
    """测试 app.py 已接入。"""
    print("=" * 50)
    print("测试 2: app.py 页面接入")
    print("=" * 50)

    import inspect
    from src.observability.dashboard.app import page_ingestion_manager

    source = inspect.getsource(page_ingestion_manager)
    has_real = "render_ingestion_manager" in source
    has_placeholder = "_placeholder_page" in source

    print(f"  使用 render_ingestion_manager: {has_real}")
    print(f"  使用 _placeholder_page: {has_placeholder}")

    assert has_real, "未接入真实页面"
    assert not has_placeholder, "仍使用占位页面"

    print("\n✅ app.py 接入测试通过\n")


def test_helpers():
    """测试辅助函数。"""
    print("=" * 50)
    print("测试 3: 辅助函数")
    print("=" * 50)

    from src.observability.dashboard.pages.ingestion_manager import (
        _stage_label,
        _format_size,
        _shorten_name,
    )

    # 阶段标签
    labels = {
        "integrity": "完整性检查",
        "load": "加载文件",
        "split": "文本切分",
        "transform": "增强处理",
        "encode": "向量编码",
        "store": "存储写入",
    }
    for eng, chn in labels.items():
        assert _stage_label(eng) == chn
        print(f"  {eng} -> {chn}")

    # 文件大小
    assert _format_size(500) == "500 B"
    assert "KB" in _format_size(2048)
    assert "MB" in _format_size(2 * 1024 * 1024)
    assert _format_size(None) == "未知"
    print(f"  500 B -> {_format_size(500)}")
    print(f"  None -> {_format_size(None)}")

    # 文件名缩短
    assert _shorten_name("/path/to/doc.pdf") == "doc.pdf"
    long = "/path/to/very_long_filename_that_exceeds_limit.pdf"
    result = _shorten_name(long, max_len=20)
    assert len(result) <= 20
    print(f"  缩短: {long} -> {result}")

    print("\n✅ 辅助函数测试通过\n")


if __name__ == "__main__":
    print("🧪 G4 Ingestion 管理页面 — 手动测试\n")
    test_page_import()
    test_app_wiring()
    test_helpers()
    print("=" * 50)
    print("所有手动测试完成！")
    print("启动 Dashboard: uv run python scripts/start_dashboard.py")
    print("=" * 50)
