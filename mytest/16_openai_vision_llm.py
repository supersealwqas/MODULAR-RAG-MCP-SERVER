# test_openai_vision_llm_real.py
# 使用真实 API 测试 OpenAIVisionLLM 实现
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from src.core.settings import load_settings
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.openai_vision_llm import OpenAIVisionLLM

# 加载配置
settings = load_settings()

print("=" * 60)
print("OpenAI Vision LLM 真实 API 测试")
print("=" * 60)

# 1. 工厂创建测试
print("\n[1] 工厂创建测试")
vision_llm = LLMFactory.create_vision_llm(settings.vision_llm)
print(f"    类型: {type(vision_llm).__name__}")
print(f"    模型: {vision_llm.model}")
print(f"    base_url: {vision_llm.base_url}")
print(f"    max_image_size: {vision_llm.max_image_size}")
assert isinstance(vision_llm, OpenAIVisionLLM)
print("    ✓ 工厂创建成功")

# 2. 查找测试图片
print("\n[2] 查找测试图片")
image_dir = Path("data/images")
images = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
print(f"    找到 {len(images)} 张图片:")
for img in images:
    print(f"    - {img} ({img.stat().st_size / 1024:.1f} KB)")

if not images:
    print("    ✗ 未找到测试图片，退出")
    sys.exit(1)

# 3. 图片文件路径调用测试
print("\n[3] 图片文件路径调用测试")
test_image = images[0]
print(f"    使用图片: {test_image}")
try:
    response = vision_llm.chat_with_image(
        text="请详细描述这张图片的内容，包括主要元素、颜色和风格",
        image=str(test_image),
    )
    print(f"    模型: {response.model}")
    print(f"    用量: {response.usage}")
    print(f"    回复:\n    {response.content[:200]}...")
    print("    ✓ 文件路径调用成功")
except Exception as e:
    print(f"    ✗ 调用失败: {e}")

# 4. bytes 输入调用测试
print("\n[4] bytes 输入调用测试")
image_bytes = test_image.read_bytes()
print(f"    图片大小: {len(image_bytes) / 1024:.1f} KB")
try:
    response = vision_llm.chat_with_image(
        text="这张图片里有什么？请用一句话概括",
        image=image_bytes,
    )
    print(f"    回复: {response.content}")
    print("    ✓ bytes 调用成功")
except Exception as e:
    print(f"    ✗ 调用失败: {e}")

# 5. 图片压缩测试（用大尺寸限制）
print("\n[5] 图片压缩测试")
vision_llm_small = OpenAIVisionLLM(
    model=settings.vision_llm.model,
    api_key=settings.vision_llm.api_key,
    base_url=settings.vision_llm.base_url,
    max_image_size=512,  # 强制压缩到 512px
)
print(f"    max_image_size: {vision_llm_small.max_image_size}")
try:
    response = vision_llm_small.chat_with_image(
        text="描述图片内容",
        image=str(test_image),
    )
    print(f"    回复: {response.content[:100]}...")
    print("    ✓ 压缩调用成功")
except Exception as e:
    print(f"    ✗ 调用失败: {e}")

# 6. 错误处理测试
print("\n[6] 错误处理测试")
try:
    vision_llm.chat_with_image("测试", b"")
    print("    ✗ 应抛出 ValueError")
except ValueError as e:
    print(f"    ✓ 空 bytes 正确抛出 ValueError: {e}")

try:
    vision_llm.chat_with_image("测试", "/nonexistent.png")
    print("    ✗ 应抛出 FileNotFoundError")
except FileNotFoundError as e:
    print(f"    ✓ 不存在路径正确抛出 FileNotFoundError: {e}")

# 7. MIME 类型检测测试
print("\n[7] MIME 类型检测")
from src.libs.llm.openai_vision_llm import OpenAIVisionLLM as OV
print(f"    PNG magic bytes: {OV._detect_mime_type('x.png', b'\\x89PNG')}")
print(f"    JPEG magic bytes: {OV._detect_mime_type('x.jpg', b'\\xff\\xd8\\xff')}")
print(f"    .webp 扩展名: {OV._detect_mime_type('test.webp', b'\\x00')}")
print(f"    未知格式: {OV._detect_mime_type('test.bin', b'\\x00\\x01')}")
print("    ✓ MIME 检测正常")

print("\n" + "=" * 60)
print("全部测试完成")
print("=" * 60)
