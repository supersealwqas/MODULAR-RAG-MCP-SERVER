# 11_splitter.py — RecursiveSplitter 测试：使用 README.md 作为测试文档
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from src.libs.splitter.recursive_splitter import RecursiveSplitter
from src.libs.splitter.splitter_factory import SplitterFactory

print("=" * 60)
print("RecursiveSplitter 功能测试（README.md）")
print("=" * 60)

# 读取 README.md 作为测试文档
readme_path = Path(__file__).parent.parent / "README.md"
with open(readme_path, "r", encoding="utf-8") as f:
    SAMPLE_TEXT = f.read()

print(f"文档路径: {readme_path}")
print(f"文档长度: {len(SAMPLE_TEXT)} 字符")

# ─────────────────────────────────────────────
# 1. 不同 chunk_size 效果对比
# ─────────────────────────────────────────────
print("\n[1] 不同 chunk_size 切分效果对比")
print("-" * 60)

sizes = [200, 400, 600, 1000, 1500]
for size in sizes:
    splitter = RecursiveSplitter(chunk_size=size, chunk_overlap=50)
    chunks = splitter.split_text(SAMPLE_TEXT)
    avg_len = sum(len(c) for c in chunks) / len(chunks)
    print(f"\nchunk_size={size:>4}  |  块数: {len(chunks):>3}  |  平均长度: {avg_len:>6.0f}")
    # 只显示前5块预览
    for i, chunk in enumerate(chunks[:5]):
        preview = chunk[:60].replace("\n", "\\n") + "..." if len(chunk) > 60 else chunk.replace("\n", "\\n")
        print(f"  [{i:>2}] len={len(chunk):>4}  |  {preview}")
    if len(chunks) > 5:
        print(f"  ... 共 {len(chunks)} 块")

# ─────────────────────────────────────────────
# 2. 代码块保护测试
# ─────────────────────────────────────────────
print("\n[2] 代码块保护测试")
print("-" * 60)

# 统计 README 中的代码块数量和大小
import re
code_blocks = re.findall(r"```[\s\S]*?```", SAMPLE_TEXT)
print(f"文档中代码块数量: {len(code_blocks)}")
for i, block in enumerate(code_blocks):
    first_line = block.split("\n")[0][:40]
    print(f"  代码块[{i}] len={len(block)}: {first_line}...")

# 测试1：大 chunk_size 保护代码块完整性
print("\n测试1: chunk_size=1000（代码块应完整保留）")
splitter = RecursiveSplitter(chunk_size=1000, chunk_overlap=50)
chunks = splitter.split_text(SAMPLE_TEXT)

code_blocks_found = 0
for i, chunk in enumerate(chunks):
    for block in code_blocks:
        # 代码块的开头在该 chunk 中
        if block[:30] in chunk:
            # 检查代码块结尾是否也在同一 chunk
            if block[-30:] in chunk:
                code_blocks_found += 1
                print(f"  代码块(len={len(block)})完整保留在 chunk [{i}]")

print(f"  完整保留: {code_blocks_found}/{len(code_blocks)}")

# 测试2：小 chunk_size 时代码块内部切分
print("\n测试2: chunk_size=200（代码块应内部切分，不跨非代码边界）")
splitter_small = RecursiveSplitter(chunk_size=200, chunk_overlap=0)
chunks_small = splitter_small.split_text(SAMPLE_TEXT)

code_chunk_count = 0
for i, chunk in enumerate(chunks_small):
    if "```" in chunk:
        code_chunk_count += 1
        preview = chunk[:50].replace("\n", "\\n")
        print(f"  chunk [{i}] len={len(chunk)}: {preview}...")
print(f"  含代码块的 chunk 数: {code_chunk_count}")

# ─────────────────────────────────────────────
# 3. 标题保护测试
# ─────────────────────────────────────────────
print("\n[3] 标题保护测试")
print("-" * 60)

# 统计 README 中的标题
headers = [line for line in SAMPLE_TEXT.split("\n") if line.startswith("#")]
print(f"文档中标题数量: {len(headers)}")
for h in headers[:8]:
    print(f"  {h}")
if len(headers) > 8:
    print(f"  ... 共 {len(headers)} 个标题")

splitter = RecursiveSplitter(chunk_size=300, chunk_overlap=0)
chunks = splitter.split_text(SAMPLE_TEXT)

header_issues = []
for i, chunk in enumerate(chunks):
    lines = chunk.strip().split("\n")
    # 检查纯标题块
    if len(lines) == 1 and lines[0].startswith("#"):
        header_issues.append(f"  chunk [{i}] 只有标题: {lines[0]}")

if header_issues:
    print("\n发现孤立标题:")
    for issue in header_issues:
        print(issue)
else:
    print("\n所有标题都与后续内容在同一 chunk 中 ✓")

# ─────────────────────────────────────────────
# 4. overlap 效果测试（使用 README 片段）
# ─────────────────────────────────────────────
print("\n[4] overlap 效果测试")
print("-" * 60)

# 取 README 中间一段文字做 overlap 对比
snippet = SAMPLE_TEXT[500:700]
print(f"测试片段长度: {len(snippet)} 字符")
print(f"片段预览: {snippet[:80].replace(chr(10), ' ')}...")

splitter_no = RecursiveSplitter(chunk_size=80, chunk_overlap=0)
splitter_yes = RecursiveSplitter(chunk_size=80, chunk_overlap=20)

chunks_no = splitter_no.split_text(snippet)
chunks_yes = splitter_yes.split_text(snippet)

print(f"\n无重叠: {len(chunks_no)} 块")
for i, c in enumerate(chunks_no):
    preview = c[:50].replace("\n", "\\n")
    print(f"  [{i}] len={len(c)}  {preview}...")

print(f"\n有重叠(20): {len(chunks_yes)} 块")
for i, c in enumerate(chunks_yes):
    preview = c[:50].replace("\n", "\\n")
    print(f"  [{i}] len={len(c)}  {preview}...")

# 验证重叠内容
if len(chunks_no) >= 2 and len(chunks_yes) >= 2:
    print(f"\n重叠验证:")
    print(f"  块[0]末尾: ...{chunks_yes[0][-20:]}")
    print(f"  块[1]开头: {chunks_yes[1][:20]}...")
    overlap_ok = chunks_yes[0][-20:] in chunks_yes[1][:25]
    print(f"  重叠内容匹配: {'✓' if overlap_ok else '✗'}")

# ─────────────────────────────────────────────
# 5. 通过工厂创建
# ─────────────────────────────────────────────
print("\n[5] 通过 SplitterFactory 创建")
print("-" * 60)

splitter = SplitterFactory.create(
    strategy="recursive",
    chunk_size=1000,
    chunk_overlap=200,
)
print(f"类型: {type(splitter).__name__}")
print(f"chunk_size: {splitter.chunk_size}")
print(f"chunk_overlap: {splitter.chunk_overlap}")

chunks = splitter.split_text(SAMPLE_TEXT)
print(f"切分结果: {len(chunks)} 块")

# 显示块大小分布
sizes = [len(c) for c in chunks]
print(f"最小块: {min(sizes)} 字符")
print(f"最大块: {max(sizes)} 字符")
print(f"平均块: {sum(sizes)/len(sizes):.0f} 字符")

# ─────────────────────────────────────────────
# 6. 中文长文本测试（使用 README 中的中文部分）
# ─────────────────────────────────────────────
print("\n[6] 中文长文本测试")
print("-" * 60)

# 提取 README 中包含中文的段落
chinese_lines = []
for line in SAMPLE_TEXT.split("\n"):
    if any('一' <= ch <= '鿿' for ch in line):
        chinese_lines.append(line)
chinese_text = "\n".join(chinese_lines)

splitter = RecursiveSplitter(chunk_size=300, chunk_overlap=30)
chunks = splitter.split_text(chinese_text)

print(f"中文内容长度: {len(chinese_text)} 字符")
print(f"切分块数: {len(chunks)}")
for i, chunk in enumerate(chunks[:5]):
    preview = chunk[:60].replace("\n", "\\n")
    print(f"  [{i}] len={len(chunk)}  {preview}...")
if len(chunks) > 5:
    print(f"  ... 共 {len(chunks)} 块")

# ─────────────────────────────────────────────
# 7. 推荐配置
# ─────────────────────────────────────────────
print("\n[7] chunk_size 配置建议")
print("-" * 60)

print("""
| 场景         | chunk_size | chunk_overlap | 说明                     |
|-------------|-----------|---------------|--------------------------|
| 中文文档     | 256~512   | 50~100        | 中文信息密度高            |
| 英文文档     | 512~1024  | 50~100        | 英文需要更多上下文        |
| 代码文档     | 1024~2048 | 100~200       | 代码需要完整函数/类       |
| FAQ/问答     | 128~256   | 20~50         | 短文本精确匹配            |

当前项目使用 BGE-M3 模型，建议从 chunk_size=512, chunk_overlap=50 开始。
""")

print("=" * 60)
print("测试完成")
print("=" * 60)
