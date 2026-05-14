# 06_logger.py — 日志工具功能测试
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from src.observability.logger import get_logger

print("=" * 60)
print("日志工具功能测试")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. 基本日志
# ─────────────────────────────────────────────
print("\n[1] 基本日志输出")
print("-" * 40)

logger = get_logger("test_module")
print("以下日志输出到 stderr:")
logger.debug("这条 DEBUG 不会显示（默认 INFO 级别）")
logger.info("这是一条 INFO 日志")
logger.warning("这是一条 WARNING 日志")
logger.error("这是一条 ERROR 日志")

# ─────────────────────────────────────────────
# 2. 不同级别
# ─────────────────────────────────────────────
print("\n[2] DEBUG 级别日志")
print("-" * 40)

debug_logger = get_logger("debug_module", level="DEBUG")
debug_logger.debug("这条 DEBUG 会显示（级别设为 DEBUG）")
debug_logger.info("INFO 也会显示")

# ─────────────────────────────────────────────
# 3. 多模块日志
# ─────────────────────────────────────────────
print("\n[3] 多模块日志区分")
print("-" * 40)

llm_logger = get_logger("llm.openai")
embed_logger = get_logger("embedding.openai")

llm_logger.info("LLM 调用开始")
embed_logger.info("Embedding 计算完成")
llm_logger.info("LLM 调用结束")

# ─────────────────────────────────────────────
# 4. 异常日志
# ─────────────────────────────────────────────
print("\n[4] 异常日志")
print("-" * 40)

try:
    result = 1 / 0
except ZeroDivisionError:
    logger.exception("捕获到异常（包含堆栈信息）")

print("\n" + "=" * 60)
print("日志工具测试完成")
print("=" * 60)
