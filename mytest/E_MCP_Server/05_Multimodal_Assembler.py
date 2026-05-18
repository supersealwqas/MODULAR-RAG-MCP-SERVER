import sys
import os
import base64
import json
import tempfile
import io

# 强制使用 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 将 src 加入路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.core.response.multimodal_assembler import (
    assemble_image_content,
    assemble_multimodal_response,
    _get_mime_type,
)
from src.core.types import RetrievalResult

def main():
    print(f"\n{'=' * 60}\n  E6: Multimodal Assembler 测试\n{'=' * 60}")

    # 1. 创建临时图片
    png_data = (
        b'\x89PNG\r\n\x1a\n'
        b'\x00\x00\x00\rIHDR'
        b'\x00\x00\x00\x01'
        b'\x00\x00\x00\x01'
        b'\x08\x02'
        b'\x00\x00\x00'
        b'\x90wS\xde'
        b'\x00\x00\x00\x0cIDAT'
        b'x\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05'
        b'\x18\xd8N\x00\x00\x00\x00IEND'
        b'\xaeB`\x82'
    )
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(png_data)
        image_path = f.name

    try:
        # 2. MIME 检测
        print("\n[1] MIME 类型检测测试:")
        for ext in ["test.png", "test.jpg", "test.webp"]:
            print(f"  {ext} -> {_get_mime_type(ext)}")

        # 3. 单图片组装
        print("\n[2] 组装 ImageContent (Base64)...")
        img_content = assemble_image_content(image_path)
        print(f"  Type: {img_content['type']}")
        print(f"  Mime: {img_content['mimeType']}")
        print(f"  Data 首 20 位: {img_content['data'][:20]}...")

        # 4. 多模态响应组装
        print("\n[3] 组装 Text + Image 响应...")
        mock_results = [
            RetrievalResult(
                chunk_id="c1",
                score=1.0,
                text="这张图展示了架构设计",
                metadata={"images": [{"id": "img1", "path": image_path}]}
            )
        ]
        full_content = assemble_multimodal_response(mock_results, "## 标题\n这是文本内容")
        
        print(f"  Content 列表长度: {len(full_content)}")
        for i, item in enumerate(full_content):
            print(f"    Item {i}: Type={item['type']}")

        # 5. 序列化检查
        print("\n[4] JSON 序列化压力测试...")
        try:
            json_str = json.dumps(full_content)
            print(f"  序列化成功 (长度: {len(json_str)})")
        except Exception as e:
            print(f"  序列化失败: {e}")

        print("\n=== E6 多模态测试完成 ===")

    finally:
        if os.path.exists(image_path):
            os.unlink(image_path)

if __name__ == "__main__":
    main()
