"""MCP Server entry point."""
from __future__ import annotations

import sys


def main() -> int:
    try:
        from src.core.settings import load_settings
        from src.observability.logger import get_logger

        settings = load_settings()
        logger = get_logger(__name__, settings.observability.log_level)
        logger.info(
            "Config loaded: LLM=%s/%s, Embedding=%s/%s",
            settings.llm.provider,
            settings.llm.model,
            settings.embedding.provider,
            settings.embedding.model,
        )
    except Exception as e:
        print(f"Startup failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
