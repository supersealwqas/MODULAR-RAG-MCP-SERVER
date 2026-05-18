"""配置加载与校验模块。

负责读取 config/settings.yaml 配置文件，解析为结构化的 Settings 对象，
并在启动时校验必填字段是否存在。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import yaml
from dotenv import load_dotenv

# 加载 .env 文件（如果存在）
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


def _resolve_env_vars(value: str) -> str:
    """解析字符串中的环境变量引用。

    支持格式: ${VAR_NAME} 或 ${VAR_NAME:-default_value}

    参数:
        value: 可能包含环境变量引用的字符串

    返回:
        替换环境变量后的字符串
    """
    if not isinstance(value, str):
        return value

    # 匹配 ${VAR_NAME} 或 ${VAR_NAME:-default}
    pattern = r'\$\{([^}]+)\}'

    def replace_match(match):
        var_expr = match.group(1)
        if ':-' in var_expr:
            # 有默认值: ${VAR:-default}
            var_name, default = var_expr.split(':-', 1)
            return os.environ.get(var_name.strip(), default)
        else:
            # 无默认值: ${VAR}
            var_name = var_expr.strip()
            env_value = os.environ.get(var_name)
            if env_value is None:
                # 环境变量不存在时返回空字符串（避免配置加载失败）
                return ""
            return env_value

    return re.sub(pattern, replace_match, value)


@dataclass
class LLMConfig:
    """LLM 提供者配置。

    属性:
        provider: 提供者名称（如 "openai"、"ollama"）
        model: 模型名称（如 "gpt-4o"、"deepseek-v4-flash"）
        base_url: API 基础 URL（兼容 OpenAI 格式的自定义端点）
        api_key: API 密钥
        temperature: 生成温度，控制随机性（0.0-2.0）
        max_tokens: 最大生成 token 数
    """
    provider: str
    model: str
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096


@dataclass
class OllamaConfig:
    """Ollama 本地模型配置。

    属性:
        model: 模型名称（如 "gemma4"、"llama3"、"qwen2"）
        base_url: Ollama 服务地址（默认 http://localhost:11434）
        temperature: 生成温度（0.0-2.0）
        max_tokens: 最大生成 token 数
    """
    model: str = "gemma4"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 4096


@dataclass
class EmbeddingConfig:
    """Embedding 提供者配置。

    属性:
        provider: 提供者名称（如 "openai"、"bge"、"ollama"）
        model: 嵌入模型名称
        dimensions: 向量维度
        api_key: API 密钥
        model_path: 本地模型路径（用于 BGE 等本地模型）
        sparse_backend: 稀疏向量后端（"jieba" 或 "bge"）
    """
    provider: str
    model: str
    dimensions: int = 1024
    api_key: str = ""
    model_path: str = ""
    sparse_backend: str = "jieba"


@dataclass
class SplitterConfig:
    """文本分块配置。

    属性:
        strategy: 分块策略（如 "recursive"、"semantic"、"fixed_length"）
        chunk_size: 每个文本块的最大字符数
        chunk_overlap: 相邻文本块之间的重叠字符数
        separators: 自定义分隔符列表（空列表使用策略默认值）
        keep_code_blocks: 是否保护代码块不被切碎
        keep_headers: 是否保护标题不与正文分离
    """
    strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: List[str] = field(default_factory=list)
    keep_code_blocks: bool = True
    keep_headers: bool = True


@dataclass
class VectorStoreConfig:
    """向量存储配置。

    属性:
        provider: 存储提供者（如 "chroma"）
        persist_directory: 持久化目录路径
    """
    provider: str
    persist_directory: str = "data/db/chroma"


@dataclass
class RetrievalConfig:
    """检索管线配置。

    属性:
        top_k: 返回的最大结果数
        hybrid: 是否启用混合检索（Dense + Sparse）
        dense_weight: Dense 检索权重（语义相似度），默认 0.7
        sparse_weight: Sparse 检索权重（BM25 关键词匹配），默认 0.3
    """
    top_k: int = 10
    hybrid: bool = True
    dense_weight: float = 0.7
    sparse_weight: float = 0.3


@dataclass
class RerankConfig:
    """重排序器配置。

    属性:
        enabled: 是否启用重排序
        provider: 重排序提供者（如 "cross-encoder"、"llm"、"none"）
        model: 重排序模型名称
        top_k: 重排序后保留的结果数
    """
    enabled: bool = False
    provider: str = "none"
    model: str = ""
    top_k: int = 5


@dataclass
class EvaluationConfig:
    """评估框架配置。

    属性:
        backends: 评估后端列表（如 ["ragas", "custom"]）
    """
    backends: List[str] = field(default_factory=lambda: ["custom"])


@dataclass
class VisionLLMConfig:
    """Vision LLM 配置，用于图片描述等多模态任务。

    属性:
        provider: 提供者名称（如 "openai"、"ollama"）
        model: 模型名称（如 "mimo-v2.5"、"llava"）
        base_url: API 基础 URL
        api_key: API 密钥
        temperature: 生成温度
        max_tokens: 最大生成 token 数
        max_image_size: 图片最大尺寸（像素），超过时自动压缩
    """
    provider: str = ""
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096
    max_image_size: int = 2048


@dataclass
class ObservabilityConfig:
    """可观测性配置。

    属性:
        log_level: 日志级别（DEBUG、INFO、WARNING、ERROR）
        traces_file: 追踪日志文件路径（JSONL 格式）
    """
    log_level: str = "INFO"
    traces_file: str = "logs/traces.jsonl"


@dataclass
class PipelineConfig:
    """摄取管线行为配置。

    属性:
        use_vision_llm: 是否启用 Vision LLM 生成图片描述（默认 False）
        use_llm_refiner: 是否启用 LLM 辅助去噪（默认 False）
        use_llm_enricher: 是否启用 LLM 辅助元数据增强（默认 False）
    """
    use_vision_llm: bool = False
    use_llm_refiner: bool = False
    use_llm_enricher: bool = False


@dataclass
class Settings:
    """根配置容器，包含所有子系统配置。

    属性:
        llm: LLM 配置
        embedding: Embedding 配置
        vector_store: 向量存储配置
        retrieval: 检索配置
        rerank: 重排序配置
        evaluation: 评估配置
        observability: 可观测性配置
    """
    llm: LLMConfig
    vision_llm: VisionLLMConfig
    ollama: OllamaConfig
    embedding: EmbeddingConfig
    splitter: SplitterConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    rerank: RerankConfig
    evaluation: EvaluationConfig
    observability: ObservabilityConfig
    pipeline: PipelineConfig


def load_settings(path: str = "config/settings.yaml") -> Settings:
    """加载并校验配置文件。

    参数:
        path: 配置文件路径，默认为 config/settings.yaml

    返回:
        校验通过的 Settings 对象

    异常:
        FileNotFoundError: 配置文件不存在
        ValueError: 必填字段缺失或格式错误
    """
    settings_path = Path(path)
    if not settings_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(settings_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"配置文件必须包含 YAML 映射，实际类型: {type(raw).__name__}")

    settings = _parse_settings(raw)
    validate_settings(settings)
    return settings


def _parse_settings(raw: dict) -> Settings:
    """将原始字典解析为 Settings 数据类。

    参数:
        raw: 从 YAML 解析的原始字典

    返回:
        Settings 对象，缺失字段使用默认值
    """
    llm_raw = raw.get("llm", {})
    vision_llm_raw = raw.get("vision_llm", {})
    ollama_raw = raw.get("ollama", {})
    embedding_raw = raw.get("embedding", {})
    splitter_raw = raw.get("splitter", {})
    vector_store_raw = raw.get("vector_store", {})
    retrieval_raw = raw.get("retrieval", {})
    rerank_raw = raw.get("rerank", {})
    evaluation_raw = raw.get("evaluation", {})
    observability_raw = raw.get("observability", {})
    pipeline_raw = raw.get("pipeline", {})

    return Settings(
        llm=LLMConfig(
            provider=llm_raw.get("provider", ""),
            model=llm_raw.get("model", ""),
            base_url=llm_raw.get("base_url", ""),
            api_key=_resolve_env_vars(llm_raw.get("api_key", "")),
            temperature=llm_raw.get("temperature", 0.0),
            max_tokens=llm_raw.get("max_tokens", 4096),
        ),
        vision_llm=VisionLLMConfig(
            provider=vision_llm_raw.get("provider", ""),
            model=vision_llm_raw.get("model", ""),
            base_url=vision_llm_raw.get("base_url", ""),
            api_key=_resolve_env_vars(vision_llm_raw.get("api_key", "")),
            temperature=vision_llm_raw.get("temperature", 0.0),
            max_tokens=vision_llm_raw.get("max_tokens", 4096),
            max_image_size=vision_llm_raw.get("max_image_size", 2048),
        ),
        ollama=OllamaConfig(
            model=ollama_raw.get("model", "gemma4"),
            base_url=ollama_raw.get("base_url", "http://localhost:11434"),
            temperature=ollama_raw.get("temperature", 0.0),
            max_tokens=ollama_raw.get("max_tokens", 4096),
        ),
        embedding=EmbeddingConfig(
            provider=embedding_raw.get("provider", ""),
            model=embedding_raw.get("model", ""),
            dimensions=embedding_raw.get("dimensions", 1024),
            api_key=embedding_raw.get("api_key", ""),
            model_path=embedding_raw.get("model_path", ""),
            sparse_backend=embedding_raw.get("sparse_backend", "jieba"),
        ),
        splitter=SplitterConfig(
            strategy=splitter_raw.get("strategy", "recursive"),
            chunk_size=splitter_raw.get("chunk_size", 1000),
            chunk_overlap=splitter_raw.get("chunk_overlap", 200),
            separators=splitter_raw.get("separators", []),
            keep_code_blocks=splitter_raw.get("keep_code_blocks", True),
            keep_headers=splitter_raw.get("keep_headers", True),
        ),
        vector_store=VectorStoreConfig(
            provider=vector_store_raw.get("provider", ""),
            persist_directory=vector_store_raw.get("persist_directory", "data/db/chroma"),
        ),
        retrieval=RetrievalConfig(
            top_k=retrieval_raw.get("top_k", 10),
            hybrid=retrieval_raw.get("hybrid", True),
            dense_weight=retrieval_raw.get("dense_weight", 0.7),
            sparse_weight=retrieval_raw.get("sparse_weight", 0.3),
        ),
        rerank=RerankConfig(
            enabled=rerank_raw.get("enabled", False),
            provider=rerank_raw.get("provider", "none"),
            model=rerank_raw.get("model", ""),
            top_k=rerank_raw.get("top_k", 5),
        ),
        evaluation=EvaluationConfig(
            backends=evaluation_raw.get("backends", ["custom"]),
        ),
        observability=ObservabilityConfig(
            log_level=observability_raw.get("log_level", "INFO"),
            traces_file=observability_raw.get("traces_file", "logs/traces.jsonl"),
        ),
        pipeline=PipelineConfig(
            use_vision_llm=pipeline_raw.get("use_vision_llm", False),
            use_llm_refiner=pipeline_raw.get("use_llm_refiner", False),
            use_llm_enricher=pipeline_raw.get("use_llm_enricher", False),
        ),
    )


def validate_settings(settings: Settings) -> None:
    """校验配置中的必填字段。

    参数:
        settings: 待校验的 Settings 对象

    异常:
        ValueError: 必填字段缺失时抛出，错误信息包含字段路径（如 "llm.provider"）
    """
    errors = []

    if not settings.llm.provider:
        errors.append("llm.provider")
    if not settings.llm.model:
        errors.append("llm.model")
    if not settings.embedding.provider:
        errors.append("embedding.provider")
    if not settings.embedding.model:
        errors.append("embedding.model")
    if not settings.vector_store.provider:
        errors.append("vector_store.provider")

    if errors:
        raise ValueError(f"缺失必填配置: {', '.join(errors)}")
