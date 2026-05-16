"""ChunkRefiner 模块。

先做规则去噪（正则匹配+分段处理），再通过 LLM 进行智能增强。
LLM 失败时回退到规则结果，不阻塞 ingestion。
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List, Optional

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.base_llm import BaseLLM
from src.libs.llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)

# 默认 prompt 模板路径
_DEFAULT_PROMPT_PATH = os.path.join("config", "prompts", "chunk_refinement.txt")

# 默认 prompt 模板（当文件不存在时使用）
_FALLBACK_PROMPT = (
    "请清理以下文本段，去除噪声（页眉页脚、格式残留等），"
    "同时保留所有有意义的内容。只返回清理后的文本。\n\n"
    "文本：\n{text}"
)


class ChunkRefiner(BaseTransform):
    """Chunk 精炼器：规则去噪 + 可选 LLM 增强。

    处理流程：
    1. 对每个 chunk 先做规则去噪（_rule_based_refine）
    2. 若启用 LLM，再调用 LLM 进行智能增强（_llm_refine）
    3. LLM 失败时回退到规则结果，metadata 标记降级原因

    属性:
        use_llm: 是否启用 LLM 增强
        llm: LLM 实例（use_llm=True 时使用）
        prompt_template: prompt 模板字符串
    """

    def __init__(
        self,
        settings: Settings,
        llm: Optional[BaseLLM] = None,
        prompt_path: Optional[str] = None,
        use_llm: bool = False,
    ) -> None:
        """初始化 ChunkRefiner。

        参数:
            settings: 全局配置对象
            llm: LLM 实例（可选，不传时根据 settings 自动创建）
            prompt_path: prompt 模板文件路径（可选）
            use_llm: 是否启用 LLM 增强（默认 False）
        """
        self.use_llm = use_llm
        self._llm = llm
        self._settings = settings
        self.prompt_template = self._load_prompt(prompt_path)

    def _get_llm(self) -> Optional[BaseLLM]:
        """获取 LLM 实例（延迟创建）。"""
        if self._llm is None and self.use_llm:
            try:
                self._llm = LLMFactory.create(self._settings.llm)
            except Exception as e:
                logger.warning("LLM 创建失败，将仅使用规则模式: %s", e)
                return None
        return self._llm

    def _load_prompt(self, prompt_path: Optional[str] = None) -> str:
        """从文件加载 prompt 模板。

        参数:
            prompt_path: prompt 文件路径，为 None 时使用默认路径

        返回:
            prompt 模板字符串（包含 {text} 占位符）
        """
        path = prompt_path or _DEFAULT_PROMPT_PATH
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("Prompt 文件不存在: %s，使用内置 fallback", path)
            return _FALLBACK_PROMPT

    def transform(self, chunks: List[Chunk], trace: Optional[TraceContext] = None) -> List[Chunk]:
        """对 Chunk 列表进行精炼处理。

        参数:
            chunks: 待处理的 Chunk 列表
            trace: 可选的追踪上下文

        返回:
            精炼后的 Chunk 列表
        """
        refined: List[Chunk] = []
        for chunk in chunks:
            try:
                refined_chunk = self._refine_single(chunk, trace)
                refined.append(refined_chunk)
            except Exception as e:
                # 单个 chunk 处理异常不影响其他 chunk，保留原文
                logger.warning("Chunk %s 精炼失败，保留原文: %s", chunk.id, e)
                chunk.metadata["refined_by"] = "error"
                chunk.metadata["refine_error"] = str(e)
                refined.append(chunk)
        return refined

    def _refine_single(self, chunk: Chunk, trace: Optional[TraceContext] = None) -> Chunk:
        """精炼单个 Chunk。

        参数:
            chunk: 待精炼的 Chunk
            trace: 可选的追踪上下文

        返回:
            精炼后的 Chunk（可能是新对象或修改后的原对象）
        """
        original_text = chunk.text

        # 第一步：规则去噪
        rule_text = self._rule_based_refine(original_text)

        # 第二步：可选 LLM 增强
        if self.use_llm:
            llm_text = self._llm_refine(rule_text, trace)
            if llm_text is not None:
                chunk.text = llm_text
                chunk.metadata["refined_by"] = "llm"
                if trace:
                    trace.record_stage("chunk_refine", method="llm", chunk_id=chunk.id)
                return chunk

        # 使用规则结果（LLM 未启用或失败）
        chunk.text = rule_text
        chunk.metadata["refined_by"] = "rule"
        if trace:
            trace.record_stage("chunk_refine", method="rule", chunk_id=chunk.id)
        return chunk

    def _rule_based_refine(self, text: str) -> str:
        """规则去噪：去除页眉页脚、多余空白、格式标记、HTML 注释等。

        参数:
            text: 原始文本

        返回:
            去噪后的文本
        """
        # 保护代码块：提取后单独处理
        code_blocks: list[str] = []
        code_pattern = re.compile(r"```[\s\S]*?```", re.MULTILINE)
        placeholder_map: dict[str, str] = {}

        def replace_code(match: re.Match) -> str:
            idx = len(code_blocks)
            key = f"__CODE_BLOCK_{idx}__"
            code_blocks.append(match.group())
            placeholder_map[key] = match.group()
            return key

        protected_text = code_pattern.sub(replace_code, text)

        # 1. 去除 HTML 注释
        protected_text = re.sub(r"<!--[\s\S]*?-->", "", protected_text)

        # 2. 去除 style/script 标签
        protected_text = re.sub(r"<style[\s\S]*?</style>", "", protected_text, flags=re.IGNORECASE)
        protected_text = re.sub(r"<script[\s\S]*?</script>", "", protected_text, flags=re.IGNORECASE)

        # 3. 去除 HTML 标签（保留标签内的文本内容）
        protected_text = re.sub(r"<[^>]+>", "", protected_text)

        # 4. 去除常见页眉页脚模式
        #    - "标题 - 第N页" / "标题 - Page N"
        protected_text = re.sub(
            r"^.*(?:第\s*\d+\s*页|Page\s*\d+|-\s*\d+\s*-).*$",
            "",
            protected_text,
            flags=re.MULTILINE,
        )
        #    - "版权所有 © ..."
        protected_text = re.sub(r"^.*版权所有.*$", "", protected_text, flags=re.MULTILINE)

        # 5. 去除分隔线（---、===、*** 等）
        protected_text = re.sub(r"^[\-=*]{3,}\s*$", "", protected_text, flags=re.MULTILINE)

        # 6. 去除多余空白行（保留最多1个空行）
        protected_text = re.sub(r"\n{3,}", "\n\n", protected_text)

        # 7. 去除行首行尾多余空白
        lines = [line.rstrip() for line in protected_text.split("\n")]
        protected_text = "\n".join(lines)

        # 8. 去除连续多个空格（保留单个空格）
        protected_text = re.sub(r" {2,}", " ", protected_text)

        # 9. 去除多余制表符
        protected_text = re.sub(r"\t+", " ", protected_text)

        # 10. 去除首尾空白
        protected_text = protected_text.strip()

        # 恢复代码块
        for key, original in placeholder_map.items():
            protected_text = protected_text.replace(key, original)

        return protected_text

    def _llm_refine(self, text: str, trace: Optional[TraceContext] = None) -> Optional[str]:
        """LLM 增强：调用 LLM 对文本进行智能清洗。

        参数:
            text: 规则去噪后的文本
            trace: 可选的追踪上下文

        返回:
            LLM 精炼后的文本，失败时返回 None
        """
        llm = self._get_llm()
        if llm is None:
            return None

        try:
            prompt = self.prompt_template.replace("{text}", text)
            result = llm.chat_simple(prompt)
            if result and result.strip():
                return result.strip()
            return None
        except Exception as e:
            logger.warning("LLM 精炼失败，回退到规则结果: %s", e)
            if trace:
                trace.record_stage("chunk_refine_llm_error", error=str(e))
            return None
