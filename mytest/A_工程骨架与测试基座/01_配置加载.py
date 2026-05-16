# 05_config.py — 配置加载与校验功能测试
import sys
sys.stdout.reconfigure(encoding='utf-8')

from src.core.settings import (
    load_settings,
    validate_settings,
    Settings,
    LLMConfig,
    OllamaConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    RetrievalConfig,
    RerankConfig,
    EvaluationConfig,
    ObservabilityConfig,
)

print("=" * 60)
print("配置加载与校验测试")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. 加载默认配置
# ─────────────────────────────────────────────
print("\n[1] 加载默认配置文件")
print("-" * 40)

settings = load_settings()

print(f"LLM 提供者: {settings.llm.provider}")
print(f"LLM 模型:   {settings.llm.model}")
print(f"LLM 地址:   {settings.llm.base_url}")
print(f"LLM 温度:   {settings.llm.temperature}")
print(f"LLM token:  {settings.llm.max_tokens}")

print(f"\nOllama 模型: {settings.ollama.model}")
print(f"Ollama 地址: {settings.ollama.base_url}")

print(f"\nEmbedding:   {settings.embedding.provider} / {settings.embedding.model}")
print(f"VectorStore: {settings.vector_store.provider}")
print(f"检索 top_k:  {settings.retrieval.top_k}")
print(f"混合检索:    {settings.retrieval.hybrid}")
print(f"重排序:      {settings.rerank.enabled}")
print(f"评估后端:    {settings.evaluation.backends}")
print(f"日志级别:    {settings.observability.log_level}")

# ─────────────────────────────────────────────
# 2. 配置校验
# ─────────────────────────────────────────────
print("\n[2] 配置校验")
print("-" * 40)

try:
    validate_settings(settings)
    print("校验通过: 所有必填字段存在")
except ValueError as e:
    print(f"校验失败: {e}")

# ─────────────────────────────────────────────
# 3. 缺失字段校验
# ─────────────────────────────────────────────
print("\n[3] 缺失字段校验")
print("-" * 40)

bad_settings = Settings(
    llm=LLMConfig(provider="", model=""),
    ollama=OllamaConfig(),
    embedding=EmbeddingConfig(provider="", model=""),
    vector_store=VectorStoreConfig(provider=""),
    retrieval=RetrievalConfig(),
    rerank=RerankConfig(),
    evaluation=EvaluationConfig(),
    observability=ObservabilityConfig(),
)
try:
    validate_settings(bad_settings)
    print("校验通过（不应该到这里）")
except ValueError as e:
    print(f"预期错误: {e}")

# ─────────────────────────────────────────────
# 4. 数据类默认值
# ─────────────────────────────────────────────
print("\n[4] 数据类默认值")
print("-" * 40)

llm_cfg = LLMConfig(provider="test", model="test")
print(f"LLMConfig 默认值:")
print(f"  api_key:    '{llm_cfg.api_key}'")
print(f"  temperature: {llm_cfg.temperature}")
print(f"  max_tokens:  {llm_cfg.max_tokens}")

ollama_cfg = OllamaConfig()
print(f"\nOllamaConfig 默认值:")
print(f"  model:       {ollama_cfg.model}")
print(f"  base_url:    {ollama_cfg.base_url}")
print(f"  temperature: {ollama_cfg.temperature}")
print(f"  max_tokens:  {ollama_cfg.max_tokens}")

emb_cfg = EmbeddingConfig(provider="test", model="test")
print(f"\nEmbeddingConfig 默认值:")
print(f"  dimensions: {emb_cfg.dimensions}")
print(f"  api_key:    '{emb_cfg.api_key}'")

print("\n" + "=" * 60)
print("配置测试完成")
print("=" * 60)
