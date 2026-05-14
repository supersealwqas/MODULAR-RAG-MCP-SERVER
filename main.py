"""MCP Server entry point."""
from __future__ import annotations

import sys


def main() -> int:
    try:
        from src.core.settings import load_settings
        settings = load_settings()
        print(f"Config loaded: LLM={settings.llm.provider}/{settings.llm.model}", file=sys.stderr)
    except Exception as e:
        print(f"Startup failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
