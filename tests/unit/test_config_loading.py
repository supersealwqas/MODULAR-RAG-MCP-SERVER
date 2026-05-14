"""Tests for configuration loading and validation."""

import pytest
import yaml
from pathlib import Path

from src.core.settings import (
    Settings,
    load_settings,
    validate_settings,
    _parse_settings,
    LLMConfig,
    OllamaConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    RetrievalConfig,
    RerankConfig,
    EvaluationConfig,
    ObservabilityConfig,
)


@pytest.mark.unit
class TestLoadSettings:
    """Test load_settings function."""

    def test_load_default_config(self):
        """应成功加载默认 settings.yaml 配置。"""
        settings = load_settings()
        assert isinstance(settings, Settings)
        assert settings.llm.provider == "openai"
        assert settings.llm.model == "mimo-v2.5-pro"
        assert settings.llm.base_url == "https://token-plan-cn.xiaomimimo.com/v1"
        assert settings.embedding.provider == "bge"
        assert settings.vector_store.provider == "chroma"

    def test_load_custom_path(self, tmp_path):
        """Should load from custom path."""
        config = {
            "llm": {"provider": "test", "model": "test-model"},
            "embedding": {"provider": "test", "model": "test-embed"},
            "vector_store": {"provider": "memory"},
        }
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(yaml.dump(config))

        settings = load_settings(str(config_file))
        assert settings.llm.provider == "test"

    def test_file_not_found(self):
        """文件不存在时应抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError, match="配置文件不存在"):
            load_settings("nonexistent.yaml")

    def test_invalid_yaml(self, tmp_path):
        """非字典 YAML 应抛出 ValueError。"""
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("- item1\n- item2")

        with pytest.raises(ValueError, match="YAML 映射"):
            load_settings(str(config_file))


@pytest.mark.unit
class TestParseSettings:
    """Test _parse_settings function."""

    def test_parse_full_config(self):
        """Should parse all fields correctly."""
        raw = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "sk-test",
                "temperature": 0.5,
                "max_tokens": 2048,
            },
            "embedding": {
                "provider": "openai",
                "model": "ada-002",
                "dimensions": 768,
                "api_key": "sk-embed",
            },
            "vector_store": {"provider": "chroma", "persist_directory": "/tmp/db"},
            "retrieval": {"top_k": 5, "hybrid": False},
            "rerank": {"enabled": True, "provider": "cross-encoder", "model": "ms-marco", "top_k": 3},
            "evaluation": {"backends": ["ragas"]},
            "observability": {"log_level": "DEBUG", "traces_file": "/tmp/traces.jsonl"},
        }
        settings = _parse_settings(raw)

        assert settings.llm.provider == "openai"
        assert settings.llm.model == "gpt-4"
        assert settings.llm.api_key == "sk-test"
        assert settings.llm.temperature == 0.5
        assert settings.llm.max_tokens == 2048
        assert settings.embedding.dimensions == 768
        assert settings.vector_store.persist_directory == "/tmp/db"
        assert settings.retrieval.top_k == 5
        assert settings.retrieval.hybrid is False
        assert settings.rerank.enabled is True
        assert settings.evaluation.backends == ["ragas"]
        assert settings.observability.log_level == "DEBUG"

    def test_parse_empty_config(self):
        """Should use defaults for missing fields."""
        raw = {}
        settings = _parse_settings(raw)

        assert settings.llm.provider == ""
        assert settings.llm.temperature == 0.0
        assert settings.embedding.dimensions == 1536
        assert settings.retrieval.top_k == 10
        assert settings.rerank.enabled is False
        assert settings.evaluation.backends == ["custom"]


@pytest.mark.unit
class TestValidateSettings:
    """Test validate_settings function."""

    def test_valid_settings(self):
        """Should pass with all required fields."""
        settings = Settings(
            llm=LLMConfig(provider="openai", model="gpt-4"),
            ollama=OllamaConfig(),
            embedding=EmbeddingConfig(provider="openai", model="ada-002"),
            vector_store=VectorStoreConfig(provider="chroma"),
            retrieval=RetrievalConfig(),
            rerank=RerankConfig(),
            evaluation=EvaluationConfig(),
            observability=ObservabilityConfig(),
        )
        validate_settings(settings)  # Should not raise

    def test_missing_llm_provider(self):
        """Should raise for missing llm.provider."""
        settings = Settings(
            llm=LLMConfig(provider="", model="gpt-4"),
            ollama=OllamaConfig(),
            embedding=EmbeddingConfig(provider="openai", model="ada-002"),
            vector_store=VectorStoreConfig(provider="chroma"),
            retrieval=RetrievalConfig(),
            rerank=RerankConfig(),
            evaluation=EvaluationConfig(),
            observability=ObservabilityConfig(),
        )
        with pytest.raises(ValueError, match="llm.provider"):
            validate_settings(settings)

    def test_missing_multiple_fields(self):
        """Should report all missing fields."""
        settings = Settings(
            llm=LLMConfig(provider="", model=""),
            ollama=OllamaConfig(),
            embedding=EmbeddingConfig(provider="", model=""),
            vector_store=VectorStoreConfig(provider=""),
            retrieval=RetrievalConfig(),
            rerank=RerankConfig(),
            evaluation=EvaluationConfig(),
            observability=ObservabilityConfig(),
        )
        with pytest.raises(ValueError) as exc_info:
            validate_settings(settings)
        error_msg = str(exc_info.value)
        assert "llm.provider" in error_msg
        assert "llm.model" in error_msg
        assert "embedding.provider" in error_msg
        assert "embedding.model" in error_msg
        assert "vector_store.provider" in error_msg


@pytest.mark.unit
class TestSettingsDataclasses:
    """Test Settings dataclass structure."""

    def test_settings_has_all_sections(self):
        """Settings should have all configuration sections."""
        settings = Settings(
            llm=LLMConfig(provider="openai", model="gpt-4"),
            ollama=OllamaConfig(),
            embedding=EmbeddingConfig(provider="openai", model="ada-002"),
            vector_store=VectorStoreConfig(provider="chroma"),
            retrieval=RetrievalConfig(),
            rerank=RerankConfig(),
            evaluation=EvaluationConfig(),
            observability=ObservabilityConfig(),
        )
        assert hasattr(settings, "llm")
        assert hasattr(settings, "ollama")
        assert hasattr(settings, "embedding")
        assert hasattr(settings, "vector_store")
        assert hasattr(settings, "retrieval")
        assert hasattr(settings, "rerank")
        assert hasattr(settings, "evaluation")
        assert hasattr(settings, "observability")

    def test_llm_config_defaults(self):
        """LLMConfig should have sensible defaults."""
        config = LLMConfig(provider="test", model="test")
        assert config.api_key == ""
        assert config.temperature == 0.0
        assert config.max_tokens == 4096
