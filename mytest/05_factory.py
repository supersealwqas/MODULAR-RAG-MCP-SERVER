# 03_factory.py — 所有工厂模式功能测试
import sys
sys.stdout.reconfigure(encoding='utf-8')

from src.core.settings import load_settings

# 导入实现模块以触发 @register 装饰器注册
from src.libs.llm import openai_llm, ollama_llm  # noqa: F401
from src.libs.reranker import base_reranker  # noqa: F401 (NoneReranker)
from src.libs.evaluator import custom_evaluator  # noqa: F401

print("=" * 60)
print("工厂模式功能测试")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. LLM 工厂
# ─────────────────────────────────────────────
print("\n[1] LLM 工厂")
print("-" * 40)

from src.libs.llm.llm_factory import LLMFactory
from src.core.settings import LLMConfig

# 列出已注册的提供者
providers = LLMFactory.list_providers()
print(f"已注册的 LLM 提供者: {providers}")

# 创建 OpenAI 兼容实例
settings = load_settings()
config = LLMConfig(
    provider=settings.llm.provider,
    model=settings.llm.model,
    base_url=settings.llm.base_url,
    api_key=settings.llm.api_key,
)
llm = LLMFactory.create(config)
print(f"创建实例: {type(llm).__name__}")
print(f"  模型: {llm.model}")
print(f"  地址: {llm.base_url}")

# 创建 Ollama 实例
from src.core.settings import OllamaConfig
ollama_config = LLMConfig(
    provider="ollama",
    model=settings.ollama.model,
    base_url=settings.ollama.base_url,
)
ollama_llm = LLMFactory.create(ollama_config)
print(f"创建实例: {type(ollama_llm).__name__}")
print(f"  模型: {ollama_llm.model}")
print(f"  地址: {ollama_llm.base_url}")

# ─────────────────────────────────────────────
# 2. Embedding 工厂
# ─────────────────────────────────────────────
print("\n[2] Embedding 工厂")
print("-" * 40)

from src.libs.embedding.embedding_factory import EmbeddingFactory

providers = EmbeddingFactory.list_providers()
print(f"已注册的 Embedding 提供者: {providers}")

# 注意: 目前没有实际的 Embedding 实现，只有基类
# 工厂创建会抛出 ValueError
try:
    from src.core.settings import EmbeddingConfig
    emb_config = EmbeddingConfig(provider="openai", model="test")
    emb = EmbeddingFactory.create(emb_config)
except ValueError as e:
    print(f"预期错误: {e}")

# ─────────────────────────────────────────────
# 3. Splitter 工厂
# ─────────────────────────────────────────────
print("\n[3] Splitter 工厂")
print("-" * 40)

from src.libs.splitter.splitter_factory import SplitterFactory

providers = SplitterFactory.list_strategies()
print(f"已注册的 Splitter 策略: {providers}")

# 注意: 目前没有实际的 Splitter 实现
try:
    splitter = SplitterFactory.create("recursive")
except ValueError as e:
    print(f"预期错误: {e}")

# ─────────────────────────────────────────────
# 4. VectorStore 工厂
# ─────────────────────────────────────────────
print("\n[4] VectorStore 工厂")
print("-" * 40)

from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.core.settings import VectorStoreConfig

providers = VectorStoreFactory.list_providers()
print(f"已注册的 VectorStore 提供者: {providers}")

# 注意: 目前没有实际的 VectorStore 实现
try:
    vs_config = VectorStoreConfig(provider="chroma")
    store = VectorStoreFactory.create(vs_config)
except ValueError as e:
    print(f"预期错误: {e}")

# ─────────────────────────────────────────────
# 5. Reranker 工厂
# ─────────────────────────────────────────────
print("\n[5] Reranker 工厂")
print("-" * 40)

from src.libs.reranker.reranker_factory import RerankerFactory

providers = RerankerFactory.list_providers()
print(f"已注册的 Reranker 提供者: {providers}")

# 创建 NoneReranker（内置实现）
reranker = RerankerFactory.create("none")
print(f"创建实例: {type(reranker).__name__}")

# ─────────────────────────────────────────────
# 6. Evaluator 工厂
# ─────────────────────────────────────────────
print("\n[6] Evaluator 工厂")
print("-" * 40)

from src.libs.evaluator.evaluator_factory import EvaluatorFactory

providers = EvaluatorFactory.list_providers()
print(f"已注册的 Evaluator 提供者: {providers}")

# 创建 CustomEvaluator（内置实现）
evaluator = EvaluatorFactory.create("custom")
print(f"创建实例: {type(evaluator).__name__}")

print("\n" + "=" * 60)
print("工厂模式测试完成")
print("=" * 60)
