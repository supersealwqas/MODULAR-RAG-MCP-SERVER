"""OpenAI 兼容 Vision LLM 实现模块。

支持通过 OpenAI-Compatible API 调用多模态模型（如 mimo-v2.5）进行图像理解。
支持图片文件路径和 base64 两种输入方式，图片过大时自动压缩。
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Union

from src.libs.llm.base_llm import LLMResponse
from src.libs.llm.base_vision_llm import BaseVisionLLM
from src.libs.llm.llm_factory import register_vision_llm

logger = logging.getLogger(__name__)


@register_vision_llm("openai")
class OpenAIVisionLLM(BaseVisionLLM):
    """OpenAI 兼容 Vision LLM 实现。

    支持通过 base_url 配置自定义 API 端点，
    适用于 OpenAI 官方 API 及兼容格式的第三方服务。
    图片过大时自动压缩至 max_image_size 配置的尺寸。
    """

    def __init__(
        self,
        model: str,
        api_key: str = "",
        base_url: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_image_size: int = 2048,
        **kwargs,
    ):
        """初始化 OpenAI Vision LLM 实例。

        参数:
            model: 模型名称（如 "mimo-v2.5"、"gpt-4o"）
            api_key: API 密钥
            base_url: 自定义 API 端点（为空时使用 OpenAI 默认地址）
            temperature: 生成温度（0.0-2.0）
            max_tokens: 最大生成 token 数
            max_image_size: 图片最大尺寸（像素），超过时自动压缩
            **kwargs: 其他参数
        """
        super().__init__(model=model, api_key=api_key, max_image_size=max_image_size)
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat_with_image(
        self,
        text: str,
        image: Union[str, bytes],
        **kwargs,
    ) -> LLMResponse:
        """发送包含图片的多模态请求到 OpenAI 兼容 API。

        参数:
            text: 文本提示词
            image: 图片文件路径或原始字节数据
            **kwargs: 额外参数（可覆盖 temperature、max_tokens）

        返回:
            LLMResponse 包含生成内容和元数据

        异常:
            FileNotFoundError: 图片路径不存在时抛出
            ValueError: 图片数据为空时抛出
            ImportError: 未安装 openai 库时抛出
            RuntimeError: API 调用失败时抛出
        """
        # 先校验图片输入（早于 openai 导入，确保输入错误优先报错）
        image_bytes = self._load_image_bytes(image)

        # 压缩图片（如果超过最大尺寸）
        image_bytes = self._compress_image_if_needed(image_bytes)

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "请安装 openai 库: uv pip install openai"
            )

        # 构建 base64 data URI
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = self._detect_mime_type(image, image_bytes)
        data_uri = f"data:{mime_type};base64,{image_b64}"

        # 构建客户端参数
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        # 构建请求参数
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        # 构建 OpenAI vision 消息格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    },
                ],
            }
        ]

        try:
            client = OpenAI(**client_kwargs)
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
            )
        except Exception as e:
            raise RuntimeError(
                f"OpenAI Vision API 调用失败 (provider=openai, model={self.model}): {e}"
            ) from e

    def _compress_image_if_needed(self, image_bytes: bytes) -> bytes:
        """压缩超过最大尺寸的图片。

        使用 Pillow 将图片等比缩放至 max_image_size 以内。
        如果 Pillow 不可用或压缩失败，返回原始字节。

        参数:
            image_bytes: 原始图片字节

        返回:
            压缩后的图片字节（或原始字节）
        """
        try:
            from PIL import Image
        except ImportError:
            logger.warning("Pillow 未安装，跳过图片压缩。安装命令: uv pip install pillow")
            return image_bytes

        try:
            img = Image.open(io.BytesIO(image_bytes))
            width, height = img.size

            # 检查是否需要压缩
            if max(width, height) <= self.max_image_size:
                return image_bytes

            # 等比缩放
            scale = self.max_image_size / max(width, height)
            new_size = (int(width * scale), int(height * scale))
            img = img.resize(new_size, Image.LANCZOS)

            # 输出格式保持一致
            output_format = img.format or "PNG"
            buffer = io.BytesIO()
            img.save(buffer, format=output_format)
            compressed = buffer.getvalue()

            logger.info(
                f"图片已压缩: {width}x{height} -> {new_size[0]}x{new_size[1]}，"
                f"{len(image_bytes)} -> {len(compressed)} 字节"
            )
            return compressed
        except Exception as e:
            logger.warning(f"图片压缩失败，使用原始图片: {e}")
            return image_bytes

    @staticmethod
    def _detect_mime_type(image: Union[str, bytes], image_bytes: bytes) -> str:
        """检测图片 MIME 类型。

        优先从文件扩展名推断，其次从 magic bytes 推断，
        最终回退到 image/png。

        参数:
            image: 原始输入（路径或字节）
            image_bytes: 图片字节数据

        返回:
            MIME 类型字符串
        """
        # 从文件扩展名推断
        if isinstance(image, str):
            ext = Path(image).suffix.lower()
            ext_to_mime = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
                ".bmp": "image/bmp",
            }
            if ext in ext_to_mime:
                return ext_to_mime[ext]

        # 从 magic bytes 推断
        if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        if image_bytes[:2] == b"\xff\xd8":
            return "image/jpeg"
        if image_bytes[:4] == b"GIF8":
            return "image/gif"
        if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
            return "image/webp"

        # 默认 PNG
        return "image/png"
