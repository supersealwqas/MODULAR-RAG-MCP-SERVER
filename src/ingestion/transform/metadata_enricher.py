"""MetadataEnricher 模块。

对 Chunk 进行元数据增强：生成 title（标题）、summary（摘要）、tags（标签）。
支持规则模式（兜底）和 LLM 模式（高质量语义增强），LLM 失败时自动降级到规则模式。
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.base_llm import BaseLLM
from src.libs.llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)

# 默认 prompt 模板路径
_DEFAULT_PROMPT_PATH = os.path.join("config", "prompts", "metadata_enrichment.txt")

# 默认 prompt 模板（当文件不存在时使用）
_FALLBACK_PROMPT = (
    "请根据以下文本内容，生成结构化的元数据信息。\n"
    "请严格按 JSON 格式返回，包含以下字段：\n"
    '- "title": 简洁准确的标题（10-30字）\n'
    '- "summary": 内容摘要（50-100字）\n'
    '- "tags": 标签列表（3-8个关键词）\n\n'
    "文本内容：\n{text}\n\n"
    "只返回 JSON，不要添加其他内容。"
)

# LLM 输出中必须包含的字段
_REQUIRED_FIELDS = {"title", "summary", "tags"}


class MetadataEnricher(BaseTransform):
    """元数据增强器：规则增强 + 可选 LLM 增强 + 降级。

    处理流程：
    1. 对每个 chunk 先做规则增强（_rule_based_enrich）生成基础 metadata
    2. 若启用 LLM，再调用 LLM 进行语义增强（_llm_enrich）
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
        """初始化 MetadataEnricher。

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

    def transform(
        self, chunks: List[Chunk], trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """对 Chunk 列表进行元数据增强。

        参数:
            chunks: 待处理的 Chunk 列表
            trace: 可选的追踪上下文

        返回:
            增强后的 Chunk 列表
        """
        enriched: List[Chunk] = []
        for chunk in chunks:
            try:
                enriched_chunk = self._enrich_single(chunk, trace)
                enriched.append(enriched_chunk)
            except Exception as e:
                # 单个 chunk 处理异常不影响其他 chunk，保留原文并标记错误
                logger.warning("Chunk %s 元数据增强失败，保留原文: %s", chunk.id, e)
                chunk.metadata["enriched_by"] = "error"
                chunk.metadata["enrich_error"] = str(e)
                enriched.append(chunk)
        return enriched

    def _enrich_single(
        self, chunk: Chunk, trace: Optional[TraceContext] = None
    ) -> Chunk:
        """增强单个 Chunk 的元数据。

        参数:
            chunk: 待增强的 Chunk
            trace: 可选的追踪上下文

        返回:
            增强后的 Chunk
        """
        # 第一步：规则增强（兜底）
        rule_metadata = self._rule_based_enrich(chunk.text)

        # 第二步：可选 LLM 增强
        if self.use_llm:
            llm_metadata = self._llm_enrich(chunk.text, trace)
            if llm_metadata is not None:
                chunk.metadata.update(llm_metadata)
                chunk.metadata["enriched_by"] = "llm"
                if trace:
                    trace.record_stage(
                        "metadata_enrich", method="llm", chunk_id=chunk.id
                    )
                return chunk

        # 使用规则结果（LLM 未启用或失败）
        chunk.metadata.update(rule_metadata)
        chunk.metadata["enriched_by"] = "rule"
        if trace:
            trace.record_stage(
                "metadata_enrich", method="rule", chunk_id=chunk.id
            )
        return chunk

    def _rule_based_enrich(self, text: str) -> Dict[str, Any]:
        """规则增强：从文本中提取 title、summary、tags。

        参数:
            text: chunk 文本内容

        返回:
            包含 title/summary/tags 的字典
        """
        # 提取标题：优先取 Markdown 标题，否则取第一行非空文本
        title = self._extract_title(text)

        # 提取摘要：取前 2-3 句话
        summary = self._extract_summary(text)

        # 提取标签：提取关键词
        tags = self._extract_tags(text)

        return {"title": title, "summary": summary, "tags": tags}

    def _extract_title(self, text: str) -> str:
        """从文本中提取标题。

        优先级：Markdown 标题 > 第一行非空文本 > 截断文本前 30 字。

        参数:
            text: 文本内容

        返回:
            标题字符串
        """
        lines = text.strip().split("\n")

        # 查找 Markdown 标题（# 开头）
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                # 去掉 # 前缀
                title = re.sub(r"^#+\s*", "", stripped).strip()
                if title:
                    return title[:50]

        # 取第一个非空行
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) > 2:
                return stripped[:50]

        # 兜底：截断前 30 字
        clean = re.sub(r"\s+", " ", text).strip()
        return clean[:30] if clean else "未知标题"

    def _extract_summary(self, text: str) -> str:
        """从文本中提取摘要。

        取前 2-3 个完整句子，最多 200 字。

        参数:
            text: 文本内容

        返回:
            摘要字符串
        """
        # 去除 Markdown 标记
        clean = re.sub(r"^#+\s+.*$", "", text, flags=re.MULTILINE)
        clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean)  # 去除链接
        clean = re.sub(r"[*_`~]", "", clean)  # 去除格式标记
        clean = re.sub(r"\s+", " ", clean).strip()

        if not clean:
            return "无摘要"

        # 按句子分割（中文句号、英文句号、问号、感叹号）
        sentences = re.split(r"(?<=[。！？.!?])\s*", clean)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]

        if not sentences:
            return clean[:100] if len(clean) > 100 else clean

        # 取前 2-3 句，限制总长度
        summary_parts: List[str] = []
        total_len = 0
        for sent in sentences[:3]:
            if total_len + len(sent) > 200:
                break
            summary_parts.append(sent)
            total_len += len(sent)

        return "".join(summary_parts) if summary_parts else sentences[0][:100]

    def _extract_tags(self, text: str) -> List[str]:
        """从文本中提取标签（关键词）。

        使用简单规则：提取中英文关键词，去重，取 top-N。

        参数:
            text: 文本内容

        返回:
            标签列表（3-8 个）
        """
        # 去除 Markdown 标记和标点
        clean = re.sub(r"[#*_[\](){}<>|/\\~`]", " ", text)
        clean = re.sub(r"[　，。！？；：“”‘’、\s]+", " ", clean)

        # 提取中文词（2-6 字）和英文词（3+ 字）
        cn_words = re.findall(r"[一-鿿]{2,6}", clean)
        en_words = re.findall(r"[a-zA-Z]{3,}", clean.lower())

        # 停用词过滤
        cn_stopwords = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
            "这个", "那个", "什么", "可以", "可能", "已经", "因为", "所以",
            "如果", "但是", "或者", "以及", "而且", "虽然", "不过", "然后",
            "对于", "关于", "通过", "进行", "使用", "需要", "能够", "应该",
            "其中", "以下", "以上", "一些", "这些", "那些", "所有", "任何",
            "本文", "介绍", "说明", "描述", "内容", "部分", "方面", "方法",
        }
        en_stopwords = {
            "the", "and", "for", "that", "this", "with", "from", "are",
            "was", "were", "been", "have", "has", "had", "not", "but",
            "all", "can", "her", "his", "one", "our", "out", "its",
        }

        # 统计词频
        word_freq: Dict[str, int] = {}
        for word in cn_words:
            if word not in cn_stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1
        for word in en_words:
            if word not in en_stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1

        # 按频率排序，取 top 3-8
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        tags = [word for word, _ in sorted_words[:8]]

        # 保证至少 3 个标签
        if len(tags) < 3 and len(tags) > 0:
            return tags
        if not tags:
            return ["未知"]

        return tags[:8]

    def _llm_enrich(
        self, text: str, trace: Optional[TraceContext] = None
    ) -> Optional[Dict[str, Any]]:
        """LLM 增强：调用 LLM 生成语义丰富的 metadata。

        参数:
            text: chunk 文本内容
            trace: 可选的追踪上下文

        返回:
            包含 title/summary/tags 的字典，失败时返回 None
        """
        llm = self._get_llm()
        if llm is None:
            return None

        try:
            prompt = self.prompt_template.replace("{text}", text)
            result = llm.chat_simple(prompt)
            if not result or not result.strip():
                logger.warning("LLM 返回空内容，降级到规则模式")
                return None

            # 解析 LLM 返回的 JSON
            metadata = self._parse_llm_output(result.strip())
            if metadata is None:
                logger.warning("LLM 输出解析失败，降级到规则模式")
                return None

            return metadata

        except Exception as e:
            logger.warning("LLM 元数据增强失败，降级到规则模式: %s", e)
            if trace:
                trace.record_stage("metadata_enrich_llm_error", error=str(e))
            return None

    def _parse_llm_output(self, output: str) -> Optional[Dict[str, Any]]:
        """解析 LLM 返回的 JSON 输出。

        支持从 markdown 代码块中提取 JSON。

        参数:
            output: LLM 原始输出

        返回:
            解析后的字典，失败返回 None
        """
        # 尝试直接解析 JSON
        try:
            data = json.loads(output)
            return self._validate_metadata(data)
        except json.JSONDecodeError:
            pass

        # 尝试从 markdown 代码块提取 JSON
        json_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", output)
        if json_match:
            try:
                data = json.loads(json_match.group(1).strip())
                return self._validate_metadata(data)
            except json.JSONDecodeError:
                pass

        # 尝试从花括号提取 JSON
        brace_match = re.search(r"\{[\s\S]*\}", output)
        if brace_match:
            try:
                data = json.loads(brace_match.group())
                return self._validate_metadata(data)
            except json.JSONDecodeError:
                pass

        return None

    def _validate_metadata(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """校验 LLM 输出的 metadata 结构。

        参数:
            data: 解析后的字典

        返回:
            校验通过的字典，失败返回 None
        """
        if not isinstance(data, dict):
            return None

        # 检查必需字段
        for field_name in _REQUIRED_FIELDS:
            if field_name not in data:
                return None

        # 类型校验与修正
        title = data.get("title", "")
        if not isinstance(title, str) or not title.strip():
            return None

        summary = data.get("summary", "")
        if not isinstance(summary, str) or not summary.strip():
            return None

        tags = data.get("tags", [])
        if isinstance(tags, list):
            # 确保所有 tag 都是字符串
            tags = [str(t) for t in tags if t]
        else:
            return None

        return {"title": title.strip(), "summary": summary.strip(), "tags": tags}
