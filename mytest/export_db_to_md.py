"""将 ChromaDB 数据库内容导出为 Markdown 文件。

展示数据存储的具体方式：ID、元数据结构、文本内容、图片引用信息等。
"""

import chromadb
import io
import json
import os
import sys

# 强制 stdout 使用 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def format_json(data: dict) -> str:
    """将字典格式化为 Markdown JSON 代码块。"""
    return f"```json\n{json.dumps(data, ensure_ascii=False, indent=2)}\n```"


def main():
    db_path = "data/db/chroma"
    output_path = "mytest/db_dump.md"

    if not os.path.exists(db_path):
        print(f"Error: DB path {db_path} not found.")
        return

    client = chromadb.PersistentClient(path=db_path)

    # 列出所有集合
    collections = client.list_collections()
    print(f"发现 {len(collections)} 个集合")

    lines = [
        "# ChromaDB 数据库导出",
        "",
        f"导出时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
    ]

    for coll in collections:
        coll_name = coll.name
        results = coll.get(include=["metadatas", "documents"])
        ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])
        documents = results.get("documents", [])

        lines.append(f"## 集合: `{coll_name}`")
        lines.append("")
        lines.append(f"- **chunk 总数**: {len(ids)}")
        lines.append(f"- **存储字段**: id, embedding, document, metadata")
        lines.append("")

        # 统计
        image_chunks = sum(1 for m in metadatas if m.get("images"))
        lines.append(f"- **包含图片的 chunk**: {image_chunks}/{len(ids)}")
        lines.append("")
        lines.append("---")
        lines.append("")

        for i in range(len(ids)):
            meta = metadatas[i]
            doc = documents[i]
            chunk_id = ids[i]

            lines.append(f"### Chunk {i + 1}: `{chunk_id}`")
            lines.append("")

            # 基本信息表格
            lines.append("| 字段 | 值 |")
            lines.append("|------|-----|")
            lines.append(f"| chunk_index | {meta.get('chunk_index', 'N/A')} |")
            lines.append(f"| source_path | `{meta.get('source_path', 'N/A')}` |")
            lines.append(f"| doc_hash | `{meta.get('doc_hash', 'N/A')}` |")
            lines.append(f"| doc_type | {meta.get('doc_type', 'N/A')} |")
            lines.append(f"| file_size | {meta.get('file_size', 'N/A')} 字节 |")
            lines.append(f"| title | {meta.get('title', 'N/A')} |")
            lines.append(f"| refined_by | {meta.get('refined_by', 'N/A')} |")
            lines.append(f"| enriched_by | {meta.get('enriched_by', 'N/A')} |")
            lines.append("")

            # 标签
            tags_raw = meta.get("tags", "")
            if tags_raw:
                try:
                    tags = json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw
                    lines.append(f"**tags**: {', '.join(tags)}")
                except Exception:
                    lines.append(f"**tags**: {tags_raw}")
                lines.append("")

            # 摘要
            summary = meta.get("summary", "")
            if summary:
                lines.append(f"**summary**: {summary[:200]}{'...' if len(summary) > 200 else ''}")
                lines.append("")

            # 图片引用
            images_raw = meta.get("images")
            if images_raw:
                try:
                    images = json.loads(images_raw) if isinstance(images_raw, str) else images_raw
                    lines.append(f"**图片引用** ({len(images)} 张):")
                    lines.append("")
                    lines.append("| image_id | page | 尺寸 | text_offset | 文件路径 |")
                    lines.append("|----------|------|------|-------------|----------|")
                    for img in images:
                        pos = img.get("position", {})
                        size = f"{pos.get('width', '?')}×{pos.get('height', '?')}"
                        path_short = os.path.basename(img.get("path", ""))
                        lines.append(
                            f"| `{img['id']}` | p{img['page']} | {size} | {img['text_offset']} | `{path_short}` |"
                        )
                    lines.append("")
                except Exception:
                    lines.append(f"**图片原始数据**: `{images_raw}`")
                    lines.append("")
            else:
                lines.append("**图片**: 无")
                lines.append("")

            # 完整元数据 JSON
            lines.append("<details>")
            lines.append(f"<summary>完整元数据 JSON</summary>")
            lines.append("")
            # 解析 JSON 字符串字段以便展示
            parsed_meta = {}
            for k, v in meta.items():
                if isinstance(v, str):
                    try:
                        parsed_meta[k] = json.loads(v)
                    except (json.JSONDecodeError, ValueError):
                        parsed_meta[k] = v
                else:
                    parsed_meta[k] = v
            lines.append(format_json(parsed_meta))
            lines.append("")
            lines.append("</details>")
            lines.append("")

            # 完整文本
            lines.append(f"**文本内容** ({len(doc)} 字符):")
            lines.append("")
            lines.append("```")
            lines.append(doc)
            lines.append("```")
            lines.append("")
            lines.append("---")
            lines.append("")

    # 写入文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"已导出到: {output_path}")
    print(f"共 {sum(len(c.get(include=['metadatas']).get('ids', [])) for c in collections)} 个 chunk")


if __name__ == "__main__":
    main()
