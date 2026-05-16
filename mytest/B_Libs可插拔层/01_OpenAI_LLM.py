# test_llm_call.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

from src.core.settings import load_settings
from src.libs.llm.openai_llm import OpenAILLM
from src.libs.llm.base_llm import Message

# 从配置文件读取
settings = load_settings()
llm = OpenAILLM(
    model=settings.llm.model,
    api_key=settings.llm.api_key,
    base_url=settings.llm.base_url,
)

print(f"使用配置: model={settings.llm.model}, base_url={settings.llm.base_url}\n")

# 简单调用
result = llm.chat_simple("你好，请用一句话介绍自己")
print(f"回复: {result}")

# 带系统提示
result = llm.chat_simple("你好", system="你是一个Python专家")
print(f"回复: {result}")

# 完整消息列表
messages = [
    Message(role="system", content="你是一个友好的助手"),
    Message(role="user", content="什么是RAG？全称是什么？用来干啥的？请简要回答我"),
]
response = llm.chat(messages)
print(f"回复: {response.content}")
print(f"模型: {response.model}")
print(f"用量: {response.usage}")
