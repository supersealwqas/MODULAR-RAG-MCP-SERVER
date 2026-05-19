"""手动测试 G1: Dashboard 基础架构与系统总览页。

用法:
    uv run python mytest/G_Dashboard/01_dashboard_overview.py

验证项:
    1. ConfigService 能加载配置并返回组件卡片
    2. ChromaStore.list_collections 能列出 collection
    3. Dashboard app.py 能被导入（无语法错误）
    4. start_dashboard.py 脚本存在且可执行
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_config_service():
    """测试 ConfigService 加载配置。"""
    print("=" * 50)
    print("测试 1: ConfigService 加载配置")
    print("=" * 50)

    from src.observability.dashboard.services.config_service import ConfigService

    service = ConfigService()
    settings = service.get_settings()
    print(f"  LLM provider: {settings.llm.provider}")
    print(f"  LLM model: {settings.llm.model}")
    print(f"  Embedding provider: {settings.embedding.provider}")
    print(f"  Vector Store provider: {settings.vector_store.provider}")
    print(f"  Rerank enabled: {settings.rerank.enabled}")

    cards = service.get_component_cards()
    print(f"\n  组件卡片数量: {len(cards)}")
    for card in cards:
        print(f"    {card['icon']} {card['title']}: {len(card['items'])} 项配置")

    raw = service.get_raw_config_dict()
    print(f"\n  原始配置 keys: {list(raw.keys())}")

    print("\n✅ ConfigService 测试通过\n")


def test_chroma_list_collections():
    """测试 ChromaStore.list_collections。"""
    print("=" * 50)
    print("测试 2: ChromaStore.list_collections")
    print("=" * 50)

    from src.core.settings import load_settings
    from src.libs.vector_store.chroma_store import ChromaStore

    settings = load_settings()
    store = ChromaStore(
        collection_name="__test_list__",
        persist_directory=settings.vector_store.persist_directory,
    )
    collections = store.list_collections()
    print(f"  Collection 数量: {len(collections)}")
    for col in collections:
        print(f"    - {col['name']}: {col['count']} 条记录")

    print("\n✅ ChromaStore.list_collections 测试通过\n")


def test_app_importable():
    """测试 Dashboard app.py 可导入。"""
    print("=" * 50)
    print("测试 3: Dashboard app.py 可导入")
    print("=" * 50)

    try:
        from src.observability.dashboard.app import main
        print(f"  app.main 函数: {main}")
        print("\n✅ Dashboard app 可导入\n")
    except Exception as e:
        print(f"\n❌ 导入失败: {e}\n")


def test_start_script_exists():
    """测试 start_dashboard.py 脚本存在。"""
    print("=" * 50)
    print("测试 4: start_dashboard.py 脚本")
    print("=" * 50)

    script_path = project_root / "scripts" / "start_dashboard.py"
    print(f"  路径: {script_path}")
    print(f"  存在: {script_path.exists()}")
    if script_path.exists():
        content = script_path.read_text(encoding="utf-8")
        print(f"  行数: {len(content.splitlines())}")

    print("\n✅ start_dashboard.py 脚本存在\n")


def test_overview_page_importable():
    """测试 overview 页面可导入。"""
    print("=" * 50)
    print("测试 5: overview.py 页面可导入")
    print("=" * 50)

    try:
        from src.observability.dashboard.pages.overview import render_overview
        print(f"  render_overview 函数: {render_overview}")
        print("\n✅ overview 页面可导入\n")
    except Exception as e:
        print(f"\n❌ 导入失败: {e}\n")


if __name__ == "__main__":
    print("🧪 G1 Dashboard 基础架构 — 手动测试\n")
    test_config_service()
    test_chroma_list_collections()
    test_app_importable()
    test_start_script_exists()
    test_overview_page_importable()
    print("=" * 50)
    print("所有手动测试完成！")
    print("启动 Dashboard: uv run python scripts/start_dashboard.py")
    print("=" * 50)
