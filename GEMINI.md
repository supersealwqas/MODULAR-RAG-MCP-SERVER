# Modular RAG MCP Server 项目指南

本项目是一个基于 **Model Context Protocol (MCP)** 的可插拔、可观测 RAG（检索增强生成）服务框架。它通过标准化接口为 AI 助手（如 Claude, Copilot）提供私有知识库检索能力。

## 🏗️ 项目概览

- **核心目标**：提供一个模块化的 RAG 闭环，支持混合检索（Dense + Sparse）、多模态图像处理（Image Captioning）及全链路追踪。
- **技术栈**：Python 3.10+, `mcp`, `ChromaDB`, `LangChain`, `Streamlit`, `BGE-M3`.
- **设计哲学**：极致的**可插拔性**。LLM、Embedding、向量数据库等核心组件均通过工厂模式解耦，支持零代码切换。

## 📂 核心目录说明

- `src/core/`：核心定义、配置加载及查询引擎（Hybrid Search）。
- `src/ingestion/`：数据摄取流水线（Loader -> Splitter -> Transform -> Vector Upsert）。
- `src/libs/`：可插拔组件层，包含 LLM、Embedding、VectorStore、Reranker 的抽象基类与具体实现。
- `src/mcp_server/`：MCP 协议处理层，定义对外暴露的 Tools（如 `query_knowledge_hub`）。
- `src/observability/`：全链路追踪（JSON Lines 记录）与可视化管理平台（Streamlit Dashboard）。
- `scripts/`：运维与测试入口脚本（`ingest.py`, `query.py`）。
- `config/`：存放 `settings.yaml` 及 Prompt 模板。
- `data/`：本地持久化数据（Chroma 库、BM25 索引、提取的图片）。

## 🚀 关键命令与流程

### 1. 环境配置
本项目推荐使用 `uv` 管理依赖。
- **安装依赖**：`uv pip install -e .`
- **初始化配置**：在 VS Code 中调用 `setup` skill 或手动复制 `.env.example` 和 `config/settings.yaml.example`。

### 2. 数据摄取 (Ingestion)
将文档处理并存入向量库。
- **命令行执行**：`python scripts/ingest.py --path data/documents/xxx.pdf --collection default`
- **Dashboard 执行**：在管理平台的 "Ingestion 管理" 页面上传并运行。

### 3. 运行服务
- **MCP Server**：`python main.py`（通常由 MCP Client 如 Claude Desktop 自动启动）。
- **管理平台 (Dashboard)**：`streamlit run src/observability/dashboard/app.py`。

### 4. 测试与验证
遵循 TDD 开发模式。
- **运行所有测试**：`pytest`
- **运行特定类型测试**：`pytest -m unit` 或 `pytest -m integration`。

## 🛠️ 开发规约

1. **可插拔实现**：新增组件需继承 `src/libs/` 下对应的 `Base` 类，并使用 `@register_xxx` 装饰器注册到工厂中。
2. **数据契约**：所有跨模块传递的数据必须使用 `src/core/types.py` 中定义的 `Document`、`Chunk` 等强类型。
3. **可观测性**：在关键路径（Ingestion/Query）必须使用 `TraceContext` 进行打点，并确保阶段信息正确持久化至 `logs/traces.jsonl`。
4. **测试先行**：新功能开发前需在 `tests/` 目录下编写对应的单元测试或集成测试。
5. **配置优先**：避免在代码中硬编码模型名称或 API 密钥，所有配置项应在 `Settings` 类中定义。

## 📡 MCP 接口 (Tools)

- `query_knowledge_hub`：主检索工具，执行混合检索并返回带引用的结果。
- `list_collections`：获取当前可用的文档集合列表。
- `get_document_summary`：获取特定文档的摘要与元数据。

---
*注：本文件由 Gemini CLI 生成，作为项目的核心执行指南，随架构演进持续更新。*
