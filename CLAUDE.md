# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Modular RAG MCP Server — a pluggable, observable RAG framework that exposes tools via MCP (Model Context Protocol) for AI assistants like Copilot/Claude. Also serves as an interview prep project for LLM-related positions.

## Quick Start

```bash
# Use Setup Skill for one-click configuration
setup
```

## Architecture

**Core Pipeline (Ingestion):**
`PDF → Markdown → Splitter → Transform → Embed → Upsert`

**Retrieval Pipeline (Query):**
`Hybrid Search (BM25 + Dense) → RRF Fusion → Rerank → LLM Response`

**Key Directories:**
- `src/ingestion/` — Data ingestion pipeline (Loader, Splitter, Transform, Embed, VectorStore)
- `src/retrieval/` — Hybrid search, BM25, reranking
- `src/mcp_server/` — MCP protocol server exposing tools
- `src/dashboard/` — Streamlit management UI (6 pages)
- `src/evaluation/` — Ragas + custom evaluation framework
- `src/libs/` — Pluggable backends (LLM, Embedding, Reranker, VectorStore)
- `src/core/` — Shared types, config, tracing

**Pluggable Architecture:**
All core components (LLM, Embedding, Reranker, Splitter, VectorStore, Evaluator) use abstract interfaces + factory pattern. Switch backends via `settings.yaml` with zero code changes.

**Storage:**
- ChromaDB — vector storage (dense + sparse + metadata)
- SQLite — ingestion history, image index, BM25 metadata
- Pickle — BM25 inverted index (upgradable to SQLite)

## Development Commands

```bash
# Run MCP server
python -m src.mcp_server

# Run Streamlit dashboard
streamlit run src/dashboard/app.py

# Run tests (pytest)
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/
```

## Skills

Project uses `.claude/skills/` for agent-driven development:
- `auto-coder` — Auto-code from DEV_SPEC
- `qa-tester` — Automated testing and fix loops
- `setup` — One-click environment configuration
- `package` — Clean and package for distribution
- `resume-writer` — Generate tailored resume bullets
- `skill-creator` — Create new skills

## Key Design Decisions

1. **Self-built pipeline** (not LlamaIndex) — full control over pluggable interfaces
2. **Two-stage retrieval** — coarse recall (BM25 + Dense + RRF) then fine ranking (Cross-Encoder/LLM)
3. **Image-to-Text** strategy for multimodal — Vision LLM generates captions, stitched into text chunks
4. **MCP Protocol** — zero frontend needed, integrates directly with Copilot/Claude Desktop
5. **SQLite local-first** — lightweight persistence, WAL mode for concurrency, upgradable to PostgreSQL

## DEV_SPEC

All implementation details are in `DEV_SPEC.md`. When implementing features, reference the spec for interface definitions, data models, and architectural constraints.
