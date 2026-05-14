"""Configuration loading and validation for Modular RAG MCP Server."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str
    model: str
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096


@dataclass
class EmbeddingConfig:
    """Embedding provider configuration."""
    provider: str
    model: str
    dimensions: int = 1536
    api_key: str = ""


@dataclass
class VectorStoreConfig:
    """Vector store configuration."""
    provider: str
    persist_directory: str = "data/db/chroma"


@dataclass
class RetrievalConfig:
    """Retrieval pipeline configuration."""
    top_k: int = 10
    hybrid: bool = True


@dataclass
class RerankConfig:
    """Reranker configuration."""
    enabled: bool = False
    provider: str = "none"
    model: str = ""
    top_k: int = 5


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    backends: List[str] = field(default_factory=lambda: ["custom"])


@dataclass
class ObservabilityConfig:
    """Observability configuration."""
    log_level: str = "INFO"
    traces_file: str = "logs/traces.jsonl"


@dataclass
class Settings:
    """Root settings container."""
    llm: LLMConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    rerank: RerankConfig
    evaluation: EvaluationConfig
    observability: ObservabilityConfig


def load_settings(path: str = "config/settings.yaml") -> Settings:
    """Load and validate settings from YAML file.

    Args:
        path: Path to settings YAML file.

    Returns:
        Validated Settings object.

    Raises:
        FileNotFoundError: If settings file not found.
        ValueError: If required fields are missing or invalid.
    """
    settings_path = Path(path)
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")

    with open(settings_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Settings file must contain a YAML mapping, got {type(raw).__name__}")

    settings = _parse_settings(raw)
    validate_settings(settings)
    return settings


def _parse_settings(raw: dict) -> Settings:
    """Parse raw dict into Settings dataclass."""
    llm_raw = raw.get("llm", {})
    embedding_raw = raw.get("embedding", {})
    vector_store_raw = raw.get("vector_store", {})
    retrieval_raw = raw.get("retrieval", {})
    rerank_raw = raw.get("rerank", {})
    evaluation_raw = raw.get("evaluation", {})
    observability_raw = raw.get("observability", {})

    return Settings(
        llm=LLMConfig(
            provider=llm_raw.get("provider", ""),
            model=llm_raw.get("model", ""),
            api_key=llm_raw.get("api_key", ""),
            temperature=llm_raw.get("temperature", 0.0),
            max_tokens=llm_raw.get("max_tokens", 4096),
        ),
        embedding=EmbeddingConfig(
            provider=embedding_raw.get("provider", ""),
            model=embedding_raw.get("model", ""),
            dimensions=embedding_raw.get("dimensions", 1536),
            api_key=embedding_raw.get("api_key", ""),
        ),
        vector_store=VectorStoreConfig(
            provider=vector_store_raw.get("provider", ""),
            persist_directory=vector_store_raw.get("persist_directory", "data/db/chroma"),
        ),
        retrieval=RetrievalConfig(
            top_k=retrieval_raw.get("top_k", 10),
            hybrid=retrieval_raw.get("hybrid", True),
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
    )


def validate_settings(settings: Settings) -> None:
    """Validate required fields in settings.

    Args:
        settings: Settings object to validate.

    Raises:
        ValueError: If required fields are missing.
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
        raise ValueError(f"Missing required settings: {', '.join(errors)}")
