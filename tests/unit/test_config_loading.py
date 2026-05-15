"""测试配置加载与校验。"""

import pytest
import yaml
from pathlib import Path

from src.core.settings import (
    Settings,
    load_settings,
    validate_settings,
    _parse_settings,
    LLMConfig,
    VisionLLMConfig,
    OllamaConfig,
    EmbeddingConfig,
    SplitterConfig,
    VectorStoreConfig,
    RetrievalConfig,
    RerankConfig,
    EvaluationConfig,
    ObservabilityConfig,
)


@pytest.mark.unit
class TestLoadSettings:
    """测试 load_settings 函数。"""

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
        """应能从自定义路径加载配置。"""
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
    """测试 _parse_settings 函数。"""

    def test_parse_full_config(self):
        """应正确解析所有字段。"""
        raw = {
            "llm": {
                "provider": "openai",
                "model": "mimo-v2.5-pro",
                "api_key": "sk-test",
                "temperature": 0.5,
                "max_tokens": 2048,
            },
            "embedding": {
                "provider": "bge",
                "model": "bge-m3",
                "dimensions": 1024,
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
        assert settings.llm.model == "mimo-v2.5-pro"
        assert settings.llm.api_key == "sk-test"
        assert settings.llm.temperature == 0.5
        assert settings.llm.max_tokens == 2048
        assert settings.embedding.dimensions == 1024
        assert settings.vector_store.persist_directory == "/tmp/db"
        assert settings.retrieval.top_k == 5
        assert settings.retrieval.hybrid is False
        assert settings.rerank.enabled is True
        assert settings.evaluation.backends == ["ragas"]
        assert settings.observability.log_level == "DEBUG"

    def test_parse_empty_config(self):
        """缺失字段应使用默认值。"""
        raw = {}
        settings = _parse_settings(raw)

        assert settings.llm.provider == ""
        assert settings.llm.temperature == 0.0
        assert settings.embedding.dimensions == 1024
        assert settings.retrieval.top_k == 10
        assert settings.rerank.enabled is False
        assert settings.evaluation.backends == ["custom"]


@pytest.mark.unit
class TestValidateSettings:
    """测试 validate_settings 函数。"""

    def test_valid_settings(self):
        """所有必填字段存在时应通过校验。"""
        settings = Settings(
            llm=LLMConfig(provider="openai", model="mimo-v2.5-pro"),
            vision_llm=VisionLLMConfig(),
            ollama=OllamaConfig(),
            embedding=EmbeddingConfig(provider="bge", model="bge-m3"),
            splitter=SplitterConfig(),
            vector_store=VectorStoreConfig(provider="chroma"),
            retrieval=RetrievalConfig(),
            rerank=RerankConfig(),
            evaluation=EvaluationConfig(),
            observability=ObservabilityConfig(),
        )
        validate_settings(settings)  # 不应抛出异常

    def test_missing_llm_provider(self):
        """缺少 llm.provider 时应抛出异常。"""
        settings = Settings(
            llm=LLMConfig(provider="", model="mimo-v2.5-pro"),
            vision_llm=VisionLLMConfig(),
            ollama=OllamaConfig(),
            embedding=EmbeddingConfig(provider="bge", model="bge-m3"),
            splitter=SplitterConfig(),
            vector_store=VectorStoreConfig(provider="chroma"),
            retrieval=RetrievalConfig(),
            rerank=RerankConfig(),
            evaluation=EvaluationConfig(),
            observability=ObservabilityConfig(),
        )
        with pytest.raises(ValueError, match="llm.provider"):
            validate_settings(settings)

    def test_missing_multiple_fields(self):
        """缺少多个字段时应报告所有缺失字段。"""
        settings = Settings(
            llm=LLMConfig(provider="", model=""),
            vision_llm=VisionLLMConfig(),
            ollama=OllamaConfig(),
            embedding=EmbeddingConfig(provider="", model=""),
            splitter=SplitterConfig(),
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
    """测试 Settings 数据类结构。"""

    def test_settings_has_all_sections(self):
        """Settings 应包含所有配置节。"""
        settings = Settings(
            llm=LLMConfig(provider="openai", model="mimo-v2.5-pro"),
            vision_llm=VisionLLMConfig(),
            ollama=OllamaConfig(),
            embedding=EmbeddingConfig(provider="bge", model="bge-m3"),
            splitter=SplitterConfig(),
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
        """LLMConfig 应有合理的默认值。"""
        config = LLMConfig(provider="test", model="test")
        assert config.api_key == ""
        assert config.temperature == 0.0
        assert config.max_tokens == 4096
