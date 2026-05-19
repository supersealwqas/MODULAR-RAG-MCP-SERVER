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
                # 将文本中的原生图片标记替换为系统占位符
                if images:
                    text = self._align_and_replace_images(text, images)
            except Exception as e:
                logger.warning(f"图片提取失败（不影响文本解析）: {path} - {e}")

        # 构建 metadata
        metadata = {
            "source_path": os.path.abspath(path),
            "doc_type": "pdf",
            "doc_hash": doc_hash,
            "collection": collection,
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

    def _align_and_replace_images(
        self, text: str, images: List[ImageRef]
    ) -> str:
        """将图片占位符智能融合到文本中。

        尝试 1: 扫描 MarkItDown 产出的 Markdown 图片语法进行精准替换（适用于 Word/HTML 等）。
        尝试 2: (针对 PDF) 若未发现图片语法，按图片所在页码(page)比例，
                启发式地将图片分散插入到对应的上下文段落中。
        """
        if not images:
            return text

        md_image_pattern = re.compile(r"!\[.*?\]\(.*?\)")
        matches = list(md_image_pattern.finditer(text))

        # ==========================================
        # 尝试 1：精确匹配替换 (适用于原生带图片标记的文档)
        # ==========================================
        if len(matches) > 0:
            result_parts = []
            last_end = 0
            img_idx = 0

            for match in matches:
                result_parts.append(text[last_end:match.start()])
                if img_idx < len(images):
                    img = images[img_idx]
                    placeholder = make_image_placeholder(img.id)
                    img.text_offset = len("".join(result_parts))
                    img.text_length = len(placeholder)
                    result_parts.append(placeholder)
                    img_idx += 1
                else:
                    result_parts.append(match.group())
                last_end = match.end()

            result_parts.append(text[last_end:])
            result_text = "".join(result_parts)

            # 多余图片兜底附在末尾 (修复了换行粘连 Bug)
            if img_idx < len(images):
                if not result_text.endswith("\n\n"):
                    result_text += "\n" if result_text.endswith("\n") else "\n\n"
                for remaining_img in images[img_idx:]:
                    placeholder = make_image_placeholder(remaining_img.id)
                    remaining_img.text_offset = len(result_text)
                    remaining_img.text_length = len(placeholder)
                    result_text += placeholder + "\n\n"
                result_text = result_text.strip() + "\n"

            return result_text

        # ==========================================
        # 尝试 2：启发式页面映射 (专治 PDF 被吞图片的场景)
        # ==========================================
        logger.info("未发现原生图片标记，启动按页码分布的启发式插入...")

        # 找出最大页码，用于计算比例
        max_page = max((img.page for img in images), default=1)
        text_len = len(text)

        # 确保图片按页码顺序插入
        sorted_images = sorted(images, key=lambda x: x.page)

        result_parts = []
        last_idx = 0

        for img in sorted_images:
            # 估算该页在全文中的大致字符位置
            # 减去 0.5 是为了让图片尽量插在这一页内容的中间偏上位置
            estimated_pos = int(max(0, (img.page - 0.5)) / max_page * text_len)

            # 为了不切断句子，我们在估算位置附近寻找最近的段落边界 (\n\n)
            insert_pos = text.find("\n\n", estimated_pos)

            if insert_pos == -1:  # 如果往后找不到，往前找
                insert_pos = text.rfind("\n\n", 0, estimated_pos)

            if insert_pos == -1:  # 还是找不到，就硬插
                insert_pos = estimated_pos
            else:
                insert_pos += 2  # 放在 \n\n 之后

            # 确保插入位置严格向后，防止乱序
            insert_pos = max(insert_pos, last_idx)

            # 截取上一段文本
            result_parts.append(text[last_idx:insert_pos])

            # 计算并插入占位符
            placeholder = make_image_placeholder(img.id)
            img.text_offset = len("".join(result_parts))
            img.text_length = len(placeholder)

            result_parts.append(placeholder + "\n\n")
            last_idx = insert_pos

        # 补充剩余全文
        result_parts.append(text[last_idx:])

        return "".join(result_parts)
