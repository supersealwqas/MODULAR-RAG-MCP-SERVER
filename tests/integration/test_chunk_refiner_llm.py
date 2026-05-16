"""C5 ChunkRefiner 集成测试（真实 LLM 调用）。

使用 config/settings.yaml 中配置的 LLM 进行真实 refinement。
验证 LLM 配置正确性、输出质量、降级机制。

⚠️ 会产生真实 API 调用与费用。
"""

from __future__ import annotations

import os

import pytest

from src.core.settings import load_settings
from src.core.types import Chunk
from src.ingestion.transform.chunk_refiner import ChunkRefiner
# 导入 OpenAI LLM 以触发 @register_llm 注册
from src.libs.llm.openai_llm import OpenAILLM  # noqa: F401
from src.libs.llm.llm_factory import LLMFactory


def _make_chunk(text: str, chunk_id: str = "integ_0000_abcd1234") -> Chunk:
    """创建测试用 Chunk。"""
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "/test.pdf", "chunk_index": 0},
        source_ref="doc_001",
    )


@pytest.mark.integration
class TestChunkRefinerLLMIntegration:
    """ChunkRefiner 真实 LLM 集成测试。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """加载配置并创建 LLM 实例。"""
        self.settings = load_settings()
        self.llm = LLMFactory.create(self.settings.llm)

    def test_llm_refine_removes_noise(self):
        """LLM 精炼能去除噪声并保留有效内容。"""
        noisy_text = (
            "人工智能导论\n\n第一章 绪论\n\n"
            "人工智能（AI）是计算机科学的一个分支。\n\n"
            "人工智能导论 - 第1页\n\n"
            "它致力于创建能够执行通常需要人类智能的任务的系统。\n\n"
            "版权所有 © 2024"
        )
        refiner = ChunkRefiner(
            settings=self.settings,
            llm=self.llm,
            use_llm=True,
        )
        chunk = _make_chunk(noisy_text)
        result = refiner.transform([chunk])

        refined = result[0].text
        # 有效内容应保留
        assert "人工智能" in refined
        assert "计算机科学" in refined
        # metadata 标记
        assert result[0].metadata["refined_by"] == "llm"

    def test_llm_refine_preserves_meaning(self):
        """LLM 精炼保留语义完整性。"""
        technical_text = (
            "深度学习使用多层神经网络来学习数据的层次化表示。"
            "卷积神经网络（CNN）特别适合处理图像数据，"
            "而循环神经网络（RNN）和 Transformer 架构则在序列数据上表现优异。"
        )
        refiner = ChunkRefiner(
            settings=self.settings,
            llm=self.llm,
            use_llm=True,
        )
        chunk = _make_chunk(technical_text)
        result = refiner.transform([chunk])

        refined = result[0].text
        # 核心术语应保留
        assert "深度学习" in refined
        assert "神经网络" in refined

    def test_degradation_on_invalid_model(self):
        """无效模型名称时优雅降级到 rule-based，不崩溃。"""
        from src.core.settings import LLMConfig
        from unittest.mock import MagicMock

        # 创建一个使用无效模型的 LLM
        bad_settings = MagicMock()
        bad_settings.llm = LLMConfig(
            provider="openai",
            model="nonexistent-model-xyz",
            api_key="invalid-key",
            base_url=self.settings.llm.base_url,
        )

        refiner = ChunkRefiner(
            settings=bad_settings,
            use_llm=True,
        )
        chunk = _make_chunk("测试降级行为的文本内容。")
        result = refiner.transform([chunk])

        # 降级到 rule-based，不崩溃
        assert len(result) == 1
        assert result[0].metadata["refined_by"] == "rule"
        assert "测试降级行为" in result[0].text
