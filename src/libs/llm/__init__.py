"""LLM 模块。

导入各提供者实现，触发 @register_llm / @register_vision_llm 装饰器注册。
"""

# 导入 LLM 提供者（触发注册）
from src.libs.llm.openai_llm import OpenAILLM  # noqa: F401
from src.libs.llm.ollama_llm import OllamaLLM  # noqa: F401

# 导入 Vision LLM 提供者（触发注册）
from src.libs.llm.openai_vision_llm import OpenAIVisionLLM  # noqa: F401
