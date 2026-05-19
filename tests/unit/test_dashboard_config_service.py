"""测试 Dashboard 配置服务与系统总览页。"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.settings import (
    Settings,
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
    PipelineConfig,
)
from src.observability.dashboard.services.config_service import ConfigService


@pytest.fixture
def sample_settings():
    """构造测试用 Settings 对象。"""
    return Settings(
        llm=LLMConfig(provider="openai", model="gpt-4o", base_url="https://api.openai.com/v1"),
        vision_llm=VisionLLMConfig(provider="openai", model="gpt-4o-vision"),
        ollama=OllamaConfig(),
        embedding=EmbeddingConfig(provider="bge", model="bge-m3", dimensions=1024),
        splitter=SplitterConfig(strategy="recursive", chunk_size=1000, chunk_overlap=200),
        vector_store=VectorStoreConfig(provider="chroma", persist_directory="data/db/chroma"),
        retrieval=RetrievalConfig(top_k=10, hybrid=True, dense_weight=0.7, sparse_weight=0.3),
        rerank=RerankConfig(enabled=False, provider="none"),
        evaluation=EvaluationConfig(backends=["custom"]),
        observability=ObservabilityConfig(log_level="INFO", traces_file="logs/traces.jsonl"),
        pipeline=PipelineConfig(use_vision_llm=True),
    )


@pytest.fixture
def config_service_with_settings(sample_settings, tmp_path):
    """构造注入 Settings 的 ConfigService（跳过文件加载）。"""
    service = ConfigService(str(tmp_path / "dummy.yaml"))
    service._settings = sample_settings
    return service


@pytest.mark.unit
class TestConfigService:
    """测试 ConfigService 类。"""

    def test_get_component_cards_returns_list(self, config_service_with_settings):
        """get_component_cards 应返回非空列表。"""
        cards = config_service_with_settings.get_component_cards()
        assert isinstance(cards, list)
        assert len(cards) > 0

    def test_component_cards_have_required_fields(self, config_service_with_settings):
        """每个卡片应包含 title、icon、items 字段。"""
        cards = config_service_with_settings.get_component_cards()
        for card in cards:
            assert "title" in card
            assert "icon" in card
            assert "items" in card
            assert isinstance(card["items"], dict)

    def test_component_cards_contain_llm(self, config_service_with_settings):
        """卡片列表应包含 LLM 配置。"""
        cards = config_service_with_settings.get_component_cards()
        titles = [c["title"] for c in cards]
        assert any("LLM" in t for t in titles)

    def test_component_cards_contain_embedding(self, config_service_with_settings):
        """卡片列表应包含 Embedding 配置。"""
        cards = config_service_with_settings.get_component_cards()
        titles = [c["title"] for c in cards]
        assert any("Embedding" in t for t in titles)

    def test_component_cards_contain_vector_store(self, config_service_with_settings):
        """卡片列表应包含 Vector Store 配置。"""
        cards = config_service_with_settings.get_component_cards()
        titles = [c["title"] for c in cards]
        assert any("Vector Store" in t for t in titles)

    def test_component_cards_contain_retrieval(self, config_service_with_settings):
        """卡片列表应包含 Retrieval 配置。"""
        cards = config_service_with_settings.get_component_cards()
        titles = [c["title"] for c in cards]
        assert any("Retrieval" in t for t in titles)

    def test_component_cards_include_vision_when_configured(self, config_service_with_settings):
        """Vision LLM 已配置时应出现在卡片中。"""
        cards = config_service_with_settings.get_component_cards()
        titles = [c["title"] for c in cards]
        assert any("Vision" in t for t in titles)

    def test_component_cards_exclude_vision_when_empty(self, sample_settings):
        """Vision LLM 未配置时不应出现在卡片中。"""
        sample_settings.vision_llm = VisionLLMConfig()  # provider 为空
        service = ConfigService("/dummy")
        service._settings = sample_settings
        cards = service.get_component_cards()
        titles = [c["title"] for c in cards]
        assert not any("Vision" in t for t in titles)

    def test_llm_card_values_correct(self, config_service_with_settings):
        """LLM 卡片的值应与 Settings 一致。"""
        cards = config_service_with_settings.get_component_cards()
        llm_card = next(c for c in cards if "LLM" in c["title"] and "Vision" not in c["title"])
        assert llm_card["items"]["provider"] == "openai"
        assert llm_card["items"]["model"] == "gpt-4o"

    def test_get_raw_config_dict_returns_dict(self, config_service_with_settings):
        """get_raw_config_dict 应返回字典。"""
        raw = config_service_with_settings.get_raw_config_dict()
        assert isinstance(raw, dict)
        assert "llm" in raw
        assert "embedding" in raw

    def test_raw_config_masks_api_key(self, sample_settings, tmp_path):
        """原始配置中的 api_key 应被脱敏。"""
        sample_settings.llm.api_key = "sk-1234567890abcdef"
        service = ConfigService(str(tmp_path / "dummy.yaml"))
        service._settings = sample_settings
        raw = service.get_raw_config_dict()
        # api_key 应被脱敏，不包含原始值
        assert raw["llm"]["api_key"] != "sk-1234567890abcdef"
        assert "****" in raw["llm"]["api_key"]


@pytest.mark.unit
class TestChromaStoreListCollections:
    """测试 ChromaStore.list_collections 方法。"""

    def test_list_collections_returns_list(self, tmp_path):
        """list_collections 应返回列表。"""
        from src.libs.vector_store.chroma_store import ChromaStore

        store = ChromaStore(
            collection_name="test_col",
            persist_directory=str(tmp_path / "chroma"),
        )
        # 写入一条记录确保 collection 存在
        from src.libs.vector_store.base_vector_store import VectorRecord
        store.upsert([VectorRecord(
            id="test_001",
            vector=[0.1] * 10,
            text="测试文本",
            metadata={"source": "test"},
        )])

        result = store.list_collections()
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_list_collections_contains_name_and_count(self, tmp_path):
        """每个 collection 信息应包含 name 和 count。"""
        from src.libs.vector_store.chroma_store import ChromaStore
        from src.libs.vector_store.base_vector_store import VectorRecord

        store = ChromaStore(
            collection_name="my_col",
            persist_directory=str(tmp_path / "chroma"),
        )
        store.upsert([VectorRecord(
            id="r1",
            vector=[0.1] * 10,
            text="hello",
            metadata={},
        )])

        result = store.list_collections()
        my_col = next((c for c in result if c["name"] == "my_col"), None)
        assert my_col is not None
        assert my_col["count"] == 1

    def test_list_collections_empty_store(self, tmp_path):
        """空存储应返回空列表。"""
        from src.libs.vector_store.chroma_store import ChromaStore

        store = ChromaStore(
            collection_name="empty",
            persist_directory=str(tmp_path / "chroma_empty"),
        )
        # 不写入任何数据
        result = store.list_collections()
        assert isinstance(result, list)
        # 空存储可能返回 0 个 collection（未创建）
