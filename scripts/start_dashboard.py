"""Dashboard 启动脚本。

使用 streamlit CLI 启动 Dashboard 应用。

用法:
    python scripts/start_dashboard.py
    python scripts/start_dashboard.py --port 8502
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main():
    """启动 Streamlit Dashboard。"""
    # 定位 app.py
    project_root = Path(__file__).parent.parent
    app_path = project_root / "src" / "observability" / "dashboard" / "app.py"

    if not app_path.exists():
        print(f"错误: Dashboard 入口文件不存在: {app_path}")
        sys.exit(1)

    # 解析命令行参数
    port = "8501"
    args = sys.argv[1:]
    if "--port" in args:
        idx = args.index("--port")
        if idx + 1 < len(args):
            port = args[idx + 1]

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", port,
        "--server.headless", "true",
    ]

    print(f"启动 Dashboard: http://localhost:{port}")
    print(f"命令: {' '.join(cmd)}")
    print("按 Ctrl+C 停止")
    print("---")

    try:
        subprocess.run(cmd, cwd=str(project_root))
    except KeyboardInterrupt:
        print("\nDashboard 已停止")


if __name__ == "__main__":
    main()
