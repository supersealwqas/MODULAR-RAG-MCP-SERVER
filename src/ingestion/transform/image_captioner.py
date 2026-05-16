"""ImageCaptioner 模块。

当启用 Vision LLM 且存在 image_refs 时，对 chunk 中的图片生成 caption 并写回 metadata。
当禁用/不可用/异常时走降级路径，不阻塞 ingestion。
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk, ImageRef, make_image_placeholder
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.base_vision_llm import BaseVisionLLM
from src.libs.llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)

# 默认 captioning prompt 模板路径
_DEFAULT_PROMPT_PATH = os.path.join("config", "prompts", "image_captioning.txt")

# 默认 prompt 模板（当文件不存在时使用）
_FALLBACK_PROMPT = (
    "请详细描述这张图片的内容，重点关注其中的文字、图表、流程图和技术内容。"
    "请提供简洁但全面的描述，使其对文档检索有帮助。"
)


class ImageCaptioner(BaseTransform):
    """图片描述生成器：调用 Vision LLM 为 chunk 中的图片生成 caption。

    处理流程：
    1. 遍历 chunk 列表，检查每个 chunk 的 metadata["image_refs"]
    2. 对有 image_refs 的 chunk，逐张图片调用 Vision LLM 生成 caption
    3. 将 caption 写入 chunk.metadata["image_captions"]（{image_id: caption}）
    4. 将图片占位符替换为 caption 文本（内联到 chunk.text）
    5. Vision LLM 不可用时走降级路径，标记 has_unprocessed_images

    属性:
        use_vision_llm: 是否启用 Vision LLM
        vision_llm: Vision LLM 实例
        prompt_template: captioning prompt 模板
    """

    def __init__(
        self,
        settings: Settings,
        vision_llm: Optional[BaseVisionLLM] = None,
        prompt_path: Optional[str] = None,
        use_vision_llm: bool = True,
    ) -> None:
        """初始化 ImageCaptioner。

        参数:
            settings: 全局配置对象
            vision_llm: Vision LLM 实例（可选，不传时根据 settings 自动创建）
            prompt_path: prompt 模板文件路径（可选）
            use_vision_llm: 是否启用 Vision LLM（默认 True）
        """
        self.use_vision_llm = use_vision_llm
        self._vision_llm = vision_llm
        self._settings = settings
        self.prompt_template = self._load_prompt(prompt_path)

    def _get_vision_llm(self) -> Optional[BaseVisionLLM]:
        """获取 Vision LLM 实例（延迟创建）。

        返回:
            Vision LLM 实例，创建失败时返回 None
        """
        if self._vision_llm is None and self.use_vision_llm:
            try:
                self._vision_llm = LLMFactory.create_vision_llm(self._settings.vision_llm)
            except Exception as e:
                logger.warning("Vision LLM 创建失败，将跳过图片描述: %s", e)
                return None
        return self._vision_llm

    def _load_prompt(self, prompt_path: Optional[str] = None) -> str:
        """从文件加载 prompt 模板。

        参数:
            prompt_path: prompt 文件路径，为 None 时使用默认路径

        返回:
            prompt 模板字符串
        """
        path = prompt_path or _DEFAULT_PROMPT_PATH
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.warning("Prompt 文件不存在: %s，使用内置 fallback", path)
            return _FALLBACK_PROMPT

    def transform(self, chunks: List[Chunk], trace: Optional[TraceContext] = None) -> List[Chunk]:
        """对 Chunk 列表进行图片描述处理。

        参数:
            chunks: 待处理的 Chunk 列表
            trace: 可选的追踪上下文

        返回:
            处理后的 Chunk 列表（数量不变）
        """
        processed: List[Chunk] = []
        for chunk in chunks:
            try:
                processed_chunk = self._process_single(chunk, trace)
                processed.append(processed_chunk)
            except Exception as e:
                # 单个 chunk 处理异常不影响其他 chunk
                logger.warning("Chunk %s 图片描述处理失败: %s", chunk.id, e)
                chunk.metadata["has_unprocessed_images"] = True
                chunk.metadata["image_caption_error"] = str(e)
                processed.append(chunk)
        return processed

    def _process_single(self, chunk: Chunk, trace: Optional[TraceContext] = None) -> Chunk:
        """处理单个 Chunk 的图片描述。

        参数:
            chunk: 待处理的 Chunk
            trace: 可选的追踪上下文

        返回:
            处理后的 Chunk
        """
        image_refs = chunk.metadata.get("images", [])
        image_ref_ids = chunk.metadata.get("image_refs", [])

        # 无图片引用，直接返回
        if not image_ref_ids:
            return chunk

        # Vision LLM 不可用时走降级路径
        vision_llm = self._get_vision_llm()
        if vision_llm is None:
            chunk.metadata["has_unprocessed_images"] = True
            if trace:
                trace.record_stage(
                    "image_caption",
                    method="fallback",
                    reason="vision_llm_unavailable",
                    chunk_id=chunk.id,
                )
            return chunk

        # 构建 image_id -> ImageRef 映射
        ref_map: dict[str, ImageRef] = {}
        for img in image_refs:
            if isinstance(img, dict):
                ref_map[img["id"]] = ImageRef.from_dict(img)
            elif isinstance(img, ImageRef):
                ref_map[img.id] = img

        # 逐张图片生成 caption
        captions: dict[str, str] = {}
        caption_errors: list[str] = []
        for image_id in image_ref_ids:
            img_ref = ref_map.get(image_id)
            if img_ref is None:
                logger.warning("图片引用 %s 在 metadata.images 中未找到", image_id)
                caption_errors.append(f"{image_id}: not_found")
                continue

            try:
                caption = self._generate_caption(img_ref, vision_llm, trace)
                if caption:
                    captions[image_id] = caption
                else:
                    caption_errors.append(f"{image_id}: empty_response")
            except Exception as e:
                logger.warning("图片 %s 描述生成失败: %s", image_id, e)
                caption_errors.append(f"{image_id}: {e}")

        # 写入 caption 结果
        if captions:
            chunk.metadata["image_captions"] = captions
            # 将占位符替换为 caption 文本
            for image_id, caption in captions.items():
                placeholder = make_image_placeholder(image_id)
                chunk.text = chunk.text.replace(placeholder, caption)

        # 部分失败时标记
        if caption_errors:
            chunk.metadata["has_unprocessed_images"] = True
            chunk.metadata["image_caption_errors"] = caption_errors

        if trace:
            trace.record_stage(
                "image_caption",
                method="vision_llm",
                chunk_id=chunk.id,
                total_images=len(image_ref_ids),
                captioned=len(captions),
                errors=len(caption_errors),
            )

        return chunk

    def _generate_caption(
        self,
        img_ref: ImageRef,
        vision_llm: BaseVisionLLM,
        trace: Optional[TraceContext] = None,
    ) -> Optional[str]:
        """为单张图片生成 caption。

        参数:
            img_ref: 图片引用信息
            vision_llm: Vision LLM 实例
            trace: 可选的追踪上下文

        返回:
            图片描述文本，失败时返回 None
        """
        if not img_ref.path:
            logger.warning("图片 %s 路径为空，跳过", img_ref.id)
            return None

        if not os.path.exists(img_ref.path):
            logger.warning("图片文件不存在: %s，跳过", img_ref.path)
            return None

        try:
            response = vision_llm.chat_with_image(
                text=self.prompt_template,
                image=img_ref.path,
            )
            if response and response.content and response.content.strip():
                return response.content.strip()
            return None
        except Exception as e:
            logger.warning("Vision LLM 调用失败（图片 %s）: %s", img_ref.id, e)
            if trace:
                trace.record_stage(
                    "image_caption_llm_error",
                    image_id=img_ref.id,
                    error=str(e),
                )
            raise
