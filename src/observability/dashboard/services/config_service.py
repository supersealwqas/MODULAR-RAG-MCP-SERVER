"""配置读取服务模块。

封装 Settings 加载与格式化，为 Dashboard 页面提供配置数据。
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.settings import Settings, load_settings


class ConfigService:
    """配置读取服务。

    负责加载 Settings 并格式化为 Dashboard 可用的展示数据。
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        """初始化配置服务。

        参数:
            config_path: 配置文件路径
        """
        self._config_path = config_path
        self._settings: Optional[Settings] = None

    def get_settings(self) -> Settings:
        """获取 Settings 对象（懒加载）。

        返回:
            Settings 对象
        """
        if self._settings is None:
            self._settings = load_settings(self._config_path)
        return self._settings

    def get_component_cards(self) -> list[Dict[str, Any]]:
        """获取组件配置卡片数据。

        将各子系统配置格式化为 Dashboard 展示用的卡片列表。

        返回:
            卡片数据列表，每个包含 title、icon、items 字段
        """
        settings = self.get_settings()
        cards = []

        # LLM 配置
        cards.append({
            "title": "LLM 大语言模型",
            "icon": "🤖",
            "items": {
                "provider": settings.llm.provider,
                "model": settings.llm.model,
                "base_url": settings.llm.base_url or "(默认)",
                "temperature": settings.llm.temperature,
                "max_tokens": settings.llm.max_tokens,
            },
        })

        # Vision LLM 配置
        if settings.vision_llm.provider:
            cards.append({
                "title": "Vision LLM 多模态",
                "icon": "👁️",
                "items": {
                    "provider": settings.vision_llm.provider,
                    "model": settings.vision_llm.model,
                    "max_image_size": f"{settings.vision_llm.max_image_size}px",
                },
            })

        # Embedding 配置
        cards.append({
            "title": "Embedding 嵌入模型",
            "icon": "📐",
            "items": {
                "provider": settings.embedding.provider,
                "model": settings.embedding.model,
                "dimensions": settings.embedding.dimensions,
                "sparse_backend": settings.embedding.sparse_backend,
            },
        })

        # Splitter 配置
        cards.append({
            "title": "Splitter 文本切分",
            "icon": "✂️",
            "items": {
                "strategy": settings.splitter.strategy,
                "chunk_size": settings.splitter.chunk_size,
                "chunk_overlap": settings.splitter.chunk_overlap,
            },
        })

        # Vector Store 配置
        cards.append({
            "title": "Vector Store 向量存储",
            "icon": "🗄️",
            "items": {
                "provider": settings.vector_store.provider,
                "persist_directory": settings.vector_store.persist_directory,
            },
        })

        # Retrieval 配置
        cards.append({
            "title": "Retrieval 检索配置",
            "icon": "🔍",
            "items": {
                "top_k": settings.retrieval.top_k,
                "hybrid": "启用" if settings.retrieval.hybrid else "关闭",
                "dense_weight": settings.retrieval.dense_weight,
                "sparse_weight": settings.retrieval.sparse_weight,
            },
        })

        # Rerank 配置
        cards.append({
            "title": "Rerank 重排序",
            "icon": "📊",
            "items": {
                "enabled": "启用" if settings.rerank.enabled else "关闭",
                "provider": settings.rerank.provider,
                "top_k": settings.rerank.top_k,
            },
        })

        # Pipeline 配置
        cards.append({
            "title": "Pipeline 摄取管线",
            "icon": "⚙️",
            "items": {
                "vision_llm": "启用" if settings.pipeline.use_vision_llm else "关闭",
                "llm_refiner": "启用" if settings.pipeline.use_llm_refiner else "关闭",
                "llm_enricher": "启用" if settings.pipeline.use_llm_enricher else "关闭",
            },
        })

        # Observability 配置
        cards.append({
            "title": "Observability 可观测性",
            "icon": "📈",
            "items": {
                "log_level": settings.observability.log_level,
                "traces_file": settings.observability.traces_file,
            },
        })

        return cards

    def get_raw_config_dict(self) -> Dict[str, Any]:
        """获取原始配置字典（用于调试展示）。

        返回:
            配置字典，敏感字段已脱敏
        """
        settings = self.get_settings()
        config_dict = asdict(settings)
        # 脱敏 api_key 字段
        for section in config_dict.values():
            if isinstance(section, dict) and "api_key" in section:
                key = section["api_key"]
                if key:
                    section["api_key"] = key[:4] + "****" + key[-4:] if len(key) > 8 else "****"
        return config_dict
