"""PDF 加载器模块。

使用 MarkItDown 将 PDF 解析为 Markdown 文本，
可选使用 PyMuPDF 提取嵌入图片并插入占位符。
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import List, Optional

from src.core.types import (
    Document,
    ImageRef,
    make_image_placeholder,
)
from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class PdfLoader(BaseLoader):
    """PDF 文档加载器。

    使用 MarkItDown 将 PDF 转换为 Markdown 格式。
    可选提取 PDF 中的嵌入图片，插入占位符并保存到本地。
    图片提取失败不会阻塞文本解析。
    """

    def __init__(
        self,
        image_dir: str = "data/images",
        extract_images: bool = True,
    ):
        """初始化 PDF 加载器。

        参数:
            image_dir: 图片存储根目录
            extract_images: 是否提取图片（默认 True）
        """
        self.image_dir = image_dir
        self.extract_images = extract_images

    def load(self, path: str, collection: str = "default") -> Document:
        """加载 PDF 文件并解析为标准 Document 对象。

        参数:
            path: PDF 文件路径
            collection: 目标集合名称

        返回:
            Document 对象

        异常:
            FileNotFoundError: 文件不存在
            ValueError: 文件不是有效 PDF
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"文件不存在: {path}")

        # 计算文档 hash 作为 ID
        doc_hash = self._compute_file_hash(path)

        # 使用 MarkItDown 提取文本
        text, title = self._extract_text(path)

        # 可选提取图片
        images: List[ImageRef] = []
        if self.extract_images:
            try:
                images = self._extract_images(path, doc_hash, collection)
                # 在文本中插入图片占位符
                if images:
                    text = self._insert_image_placeholders(text, images)
            except Exception as e:
                logger.warning(f"图片提取失败（不影响文本解析）: {path} - {e}")

        # 构建 metadata
        metadata = {
            "source_path": os.path.abspath(path),
            "doc_type": "pdf",
            "doc_hash": doc_hash,
            "file_size": os.path.getsize(path),
            "title": title or os.path.basename(path),
        }
        if images:
            metadata["images"] = [img.to_dict() for img in images]

        return Document(
            id=doc_hash,
            text=text,
            metadata=metadata,
        )

    def _compute_file_hash(self, path: str) -> str:
        """计算文件 SHA256 前 16 位作为文档 ID。"""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]

    def _extract_text(self, path: str) -> tuple[str, Optional[str]]:
        """使用 MarkItDown 提取 PDF 文本。

        返回:
            (markdown_text, title) 元组
        """
        from markitdown import MarkItDown

        md = MarkItDown()
        result = md.convert(path)
        return result.text_content, result.title

    def _extract_images(
        self, path: str, doc_hash: str, collection: str
    ) -> List[ImageRef]:
        """使用 PyMuPDF 提取 PDF 中的嵌入图片。

        参数:
            path: PDF 文件路径
            doc_hash: 文档 hash
            collection: 集合名称

        返回:
            ImageRef 列表
        """
        import fitz  # PyMuPDF

        images: List[ImageRef] = []
        image_save_dir = os.path.join(self.image_dir, collection, doc_hash)
        os.makedirs(image_save_dir, exist_ok=True)

        pdf_doc = fitz.open(path)
        seq = 0
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            image_list = page.get_images()

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    # 提取图片数据
                    base_image = pdf_doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image.get("ext", "png")

                    # 生成 image_id
                    image_id = f"{doc_hash}_{page_num + 1}_{seq}"
                    image_filename = f"{image_id}.{image_ext}"
                    image_path = os.path.join(image_save_dir, image_filename)

                    # 保存图片文件
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)

                    # 计算图片在文本中的位置（后续由 Splitter 填充）
                    img_ref = ImageRef(
                        id=image_id,
                        path=os.path.abspath(image_path),
                        page=page_num + 1,
                        text_offset=0,
                        text_length=0,
                        position={
                            "xref": xref,
                            "width": base_image.get("width", 0),
                            "height": base_image.get("height", 0),
                        },
                    )
                    images.append(img_ref)
                    seq += 1
                except Exception as e:
                    logger.warning(
                        f"提取图片失败: page={page_num + 1}, xref={xref} - {e}"
                    )
                    continue

        pdf_doc.close()
        return images

    def _insert_image_placeholders(
        self, text: str, images: List[ImageRef]
    ) -> str:
        """在文本末尾附加图片占位符。

        由于 MarkItDown 转换后的 Markdown 不包含图片位置信息，
        将所有图片占位符附加在文本末尾，由后续 Splitter 根据
        上下文将其分发到对应的 Chunk 中。

        参数:
            text: 原始 Markdown 文本
            images: 图片引用列表

        返回:
            插入占位符后的文本
        """
        if not images:
            return text

        # 在文本末尾附加图片引用区域
        placeholders = []
        for img in images:
            placeholder = make_image_placeholder(img.id)
            # 更新 text_offset（占位符在最终文本中的位置）
            img.text_offset = len(text) + len("\n\n".join(placeholders)) + 2
            img.text_length = len(placeholder)
            placeholders.append(placeholder)

        return text + "\n\n" + "\n\n".join(placeholders)
