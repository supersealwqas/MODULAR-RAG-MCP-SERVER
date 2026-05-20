"""E2E：Dashboard 冒烟测试。

使用 Streamlit AppTest 框架自动化验证 Dashboard 的各个页面
是否能够正常加载，确保不抛出 Python 异常（冒烟测试）。
"""

import pytest
from streamlit.testing.v1 import AppTest

# Dashboard 页面入口函数名列表
PAGES = [
    "page_overview",
    "page_data_browser",
    "page_ingestion_manager",
    "page_ingestion_traces",
    "page_query_traces",
    "page_evaluation",
]


def test_dashboard_main_app():
    """测试主应用 app.py 能够正常运行（默认展示首页）。"""
    at = AppTest.from_file("src/observability/dashboard/app.py")
    at.run(timeout=10)
    assert not at.exception, f"Dashboard 主入口抛出异常: {at.exception[0]}"


@pytest.mark.parametrize("page_func", PAGES)
def test_dashboard_pages(page_func: str):
    """单独测试每个页面函数，验证页面内容渲染时不抛出异常。"""
    # 构造一个极简的脚本字符串，调用目标页面函数
    script = f"""
from src.observability.dashboard.app import {page_func}
{page_func}()
"""
    at = AppTest.from_string(script)
    at.run(timeout=10)
    assert not at.exception, f"页面 {page_func} 抛出异常: {at.exception[0]}"
