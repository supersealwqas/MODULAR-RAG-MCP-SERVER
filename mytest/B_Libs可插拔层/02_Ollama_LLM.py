# 04_ollama.py — Ollama 本地模型功能测试
import sys
sys.stdout.reconfigure(encoding='utf-8')

from src.core.settings import load_settings
from src.libs.llm.ollama_llm import OllamaLLM
from src.libs.llm.base_llm import Message

# 从配置文件读取 ollama 配置
settings = load_settings()
ollama_cfg = settings.ollama

llm = OllamaLLM(
    model=ollama_cfg.model,
    base_url=ollama_cfg.base_url,
    temperature=ollama_cfg.temperature,
    max_tokens=ollama_cfg.max_tokens,
)

print("=" * 50)
print("Ollama 本地模型测试")
print("=" * 50)
print(f"模型: {llm.model}")
print(f"地址: {llm.base_url}")
print(f"温度: {llm.temperature}")
print(f"最大 token: {llm.max_tokens}")
print()

# 测试 1: 简单调用
print("-" * 50)
print("测试 1: chat_simple 简单对话")
print("-" * 50)
result = llm.chat_simple("你好，请用一句话介绍自己")
print(f"回复: {result}")
print()

# 测试 2: 带系统提示
print("-" * 50)
print("测试 2: chat_simple 带系统提示")
print("-" * 50)
result = llm.chat_simple("用三个词形容Python", system="你是一个Python专家，回答要简洁")
print(f"回复: {result}")
print()

# 测试 3: 完整消息列表
print("-" * 50)
print("测试 3: chat 完整消息列表")
print("-" * 50)
messages = [
    Message(role="system", content="你是一个友好的助手"),
    Message(role="user", content="什么是RAG？请用一句话简要回答"),
]
response = llm.chat(messages)
print(f"回复: {response.content}")
print(f"模型: {response.model}")
print(f"用量: {response.usage}")
print()

# 测试 4: 工厂模式创建
print("-" * 50)
print("测试 4: LLMFactory 工厂创建")
print("-" * 50)
from src.libs.llm.llm_factory import LLMFactory
from src.core.settings import LLMConfig

config = LLMConfig(
    provider="ollama",
    model=ollama_cfg.model,
    base_url=ollama_cfg.base_url,
)
factory_llm = LLMFactory.create(config)
print(f"类型: {type(factory_llm).__name__}")
print(f"模型: {factory_llm.model}")
result = factory_llm.chat_simple("从1加到100等于几？只回答数字")
print(f"回复: {result}")
print()

print("=" * 50)
print("全部测试完成")
print("=" * 50)
