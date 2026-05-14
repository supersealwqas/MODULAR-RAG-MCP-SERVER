# Sample Document for Testing

## Introduction

This is a sample document used for testing the RAG ingestion pipeline.
It contains multiple sections to verify text splitting and chunking behavior.

## Key Concepts

### Retrieval-Augmented Generation (RAG)

RAG combines retrieval from a knowledge base with language model generation.
This approach grounds model responses in factual, up-to-date information.

### Vector Search

Vector search uses embeddings to find semantically similar documents.
Dense vectors capture meaning, while sparse vectors (BM25) capture keyword matches.

## Implementation Notes

The pipeline processes documents through these stages:

1. **Loading** - Parse source documents (PDF, Markdown, etc.)
2. **Splitting** - Break text into manageable chunks
3. **Transforming** - Clean and enrich chunk metadata
4. **Embedding** - Generate vector representations
5. **Storage** - Upsert into vector database

## Conclusion

This document serves as a test fixture for validating the ingestion pipeline.
Multiple sections ensure that chunking produces several distinct chunks.
