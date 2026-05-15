"""LLM 重排序器模块。

通过 LLM 对候选文档进行语义级重排序。
读取 config/prompts/rerank.txt 构造 prompt，调用 LLM 获取排序结果。
失败时返回可回退信号（fallback），由上层决定是否使用原始排序。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.libs.llm.base_llm import BaseLLM
from src.libs.reranker.base_reranker import (
    BaseReranker,
    Candidate,
    RankedCandidate,
)
from src.libs.reranker.reranker_factory import register_reranker


# 默认 prompt 模板路径
DEFAULT_PROMPT_PATH = "config/prompts/rerank.txt"


@register_reranker("llm")
class LLMReranker(BaseReranker):
    """LLM 重排序器。

    通过 LLM 对候选文档进行语义级重排序：
    1. 将候选文档格式化为带编号的列表
    2. 构造 rerank prompt 发送给 LLM
    3. 解析 LLM 返回的排序 ID 列表
    4. 按排序结果重新排列候选文档

    失败时通过 fallback 信号通知上层回退到原始排序。
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        prompt_path: Optional[str] = None,
        **kwargs,
    ):
        """初始化 LLM 重排序器。

        参数:
            llm: LLM 实例（可选，延迟注入）
            prompt_path: rerank prompt 模板文件路径（默认 config/prompts/rerank.txt）
            **kwargs: 其他参数（忽略，保留接口兼容）
        """
        self._llm = llm
        self._prompt_path = prompt_path or DEFAULT_PROMPT_PATH
        self._prompt_template: Optional[str] = None

    def set_llm(self, llm: BaseLLM) -> None:
        """注入 LLM 实例（支持延迟初始化）。

        参数:
            llm: BaseLLM 子类实例
        """
        self._llm = llm

    def _load_prompt_template(self) -> str:
        """从 prompt 文件加载模板。

        优先使用缓存，其次从文件加载。

        返回:
            prompt 模板字符串，包含 {query} 和 {passages} 占位符

        异常:
            FileNotFoundError: prompt 文件不存在时抛出
        """
        if self._prompt_template is not None:
            return self._prompt_template

        prompt_path = Path(self._prompt_path)
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Rerank prompt 文件不存在: {self._prompt_path}，"
                f"请确保 config/prompts/rerank.txt 存在。"
            )

        self._prompt_template = prompt_path.read_text(encoding="utf-8")
        return self._prompt_template

    def _format_passages(self, candidates: List[Candidate]) -> str:
        """将候选文档格式化为带编号的文本列表。

        参数:
            candidates: 候选文档列表

        返回:
            格式化后的文本，每行格式为 "[ID] text"
        """
        lines = []
        for c in candidates:
            # 截断过长文本，避免超出 LLM 上下文
            text = c.text[:500] + "..." if len(c.text) > 500 else c.text
            lines.append(f"[{c.id}] {text}")
        return "\n".join(lines)

    def _parse_ranked_ids(self, response: str, valid_ids: set) -> List[str]:
        """从 LLM 响应中解析排序后的 ID 列表。

        支持多种常见格式：
        - 逗号分隔：id1, id2, id3
        - 换行分隔：
          id1
          id2
        - 编号列表：1. id1, 2. id2
        - 方括号引用：[id1], [id2]

        参数:
            response: LLM 原始响应文本
            valid_ids: 有效的候选 ID 集合

        返回:
            解析出的 ID 列表（按出现顺序），仅包含 valid_ids 中存在的 ID
        """
        # 提取所有 [id] 格式的 ID
        bracket_ids = re.findall(r"\[([^\]]+)\]", response)

        # 提取所有看起来像 ID 的 token（字母数字、下划线、连字符）
        token_ids = re.findall(r"[a-zA-Z0-9_\-]+(?:\.[a-zA-Z0-9_\-]+)*", response)

        # 合并去重，保持顺序
        seen = set()
        ordered_ids = []
        for id_str in bracket_ids + token_ids:
            if id_str in valid_ids and id_str not in seen:
                seen.add(id_str)
                ordered_ids.append(id_str)

        return ordered_ids

    def rerank(
        self,
        query: str,
        candidates: List[Candidate],
        top_k: Optional[int] = None,
        **kwargs,
    ) -> List[RankedCandidate]:
        """通过 LLM 对候选文档进行重排序。

        参数:
            query: 查询文本
            candidates: 待重排序的候选文档列表
            top_k: 返回的最大结果数（None 表示返回全部）
            **kwargs: 额外参数
                - llm: 临时覆盖 LLM 实例

        返回:
            按重排序分数降序排列的 RankedCandidate 列表

        异常:
            RuntimeError: LLM 调用失败且无法回退时抛出
        """
        # 空候选直接返回
        if not candidates:
            return []

        # 获取 LLM 实例
        llm = kwargs.get("llm", self._llm)
        if llm is None:
            raise RuntimeError(
                "LLMReranker 未配置 LLM 实例，"
                "请通过构造函数或 set_llm() 注入。"
            )

        # 构造 prompt
        template = self._load_prompt_template()
        passages = self._format_passages(candidates)
        prompt = template.format(query=query, passages=passages)

        # 调用 LLM
        try:
            response = llm.chat_simple(prompt)
        except Exception as e:
            raise RuntimeError(
                f"LLM Reranker 调用失败: {type(e).__name__}: {e}"
            ) from e

        # 解析排序结果
        valid_ids = {c.id for c in candidates}
        ranked_ids = self._parse_ranked_ids(response, valid_ids)

        if not ranked_ids:
            raise RuntimeError(
                f"LLM Reranker 无法从响应中解析有效的排序 ID。"
                f"响应内容: {response[:200]}"
            )

        # 按 LLM 排序构建结果
        id_to_candidate = {c.id: c for c in candidates}
        ranked_candidates = []
        for rank, cid in enumerate(ranked_ids):
            c = id_to_candidate[cid]
            # 分数从 1.0 递减，rank 越小分数越高
            score = 1.0 - (rank / max(len(ranked_ids), 1))
            ranked_candidates.append(RankedCandidate(
                id=c.id,
                text=c.text,
                rerank_score=score,
                original_score=c.score,
                metadata=c.metadata,
            ))

        # 补充 LLM 未提及的候选（放在末尾，低分）
        mentioned = set(ranked_ids)
        for c in candidates:
            if c.id not in mentioned:
                ranked_candidates.append(RankedCandidate(
                    id=c.id,
                    text=c.text,
                    rerank_score=0.0,
                    original_score=c.score,
                    metadata={**c.metadata, "rerank_missing": True},
                ))

        if top_k is not None:
            ranked_candidates = ranked_candidates[:top_k]

        return ranked_candidates
