"""Vision LLM 抽象基类模块。

定义多模态 LLM（文本+图片）的接口规范，为 ImageCaptioner 等上层模块提供底层抽象。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

from src.libs.llm.base_llm import LLMResponse


class BaseVisionLLM(ABC):
    """所有 Vision LLM 实现的抽象基类。

    子类必须实现 `chat_with_image` 方法，支持文本+图片的多模态输入。
    图片可通过文件路径（str）或原始字节（bytes）传入。
    """

    def __init__(
        self,
        model: str,
        api_key: str = "",
        max_image_size: int = 2048,
        **kwargs,
    ):
        """初始化 Vision LLM 实例。

        参数:
            model: 模型名称（如 "mimo-v2.5"、"llava"）
            api_key: API 密钥（可选，部分本地模型不需要）
            max_image_size: 图片最大尺寸（像素），超过时由子类决定是否压缩
            **kwargs: 其他提供者特定参数（如 base_url、temperature）
        """
        self.model = model
        self.api_key = api_key
        self.max_image_size = max_image_size

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image: Union[str, bytes],
        **kwargs,
    ) -> LLMResponse:
        """发送包含图片的多模态请求。

        参数:
            text: 文本提示词，描述需要对图片执行的任务
            image: 图片输入，支持两种形式：
                - str: 图片文件路径
                - bytes: 图片原始字节数据
            **kwargs: 提供者特定参数（如 temperature、max_tokens）

        返回:
            LLMResponse 对象，包含生成内容和元数据

        异常:
            FileNotFoundError: 图片路径不存在时抛出（仅 str 类型）
            ValueError: 图片数据为空或格式不支持时抛出
            RuntimeError: API 调用失败时抛出
        """
        ...

    @staticmethod
    def _load_image_bytes(image: Union[str, bytes]) -> bytes:
        """将图片输入统一转换为字节数据。

        参数:
            image: 图片文件路径或原始字节

        返回:
            图片的字节数据

        异常:
            FileNotFoundError: 图片路径不存在时抛出
            ValueError: 图片数据为空时抛出
        """
        if isinstance(image, bytes):
            if not image:
                raise ValueError("图片字节数据不能为空")
            return image

        # str 类型：当作文件路径处理
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"图片文件不存在: {image}")
        return path.read_bytes()
