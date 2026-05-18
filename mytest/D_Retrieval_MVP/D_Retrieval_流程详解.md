# D 阶段 (Retrieval MVP) 检索流程详解

## 一、核心数据类型

D 阶段依赖两个关键数据类（定义在 `src/core/types.py`）：

### ProcessedQuery (D1 输出)
```python
@dataclass
class ProcessedQuery:
    original: str                    # 原始查询文本
    keywords: List[str]              # 提取的关键词列表
    filters: Dict[str, Any]          # 解析后的过滤条件
```

### RetrievalResult (D2-D6 核心载体)
```python
@dataclass
class RetrievalResult:
    chunk_id: str                    # 命中 Chunk 的 ID
    score: float                     # 相似度/相关性分数
    text: str                        # Chunk 文本内容
    metadata: Dict[str, Any]         # 附加元数据
```

这两个 dataclass 是全链路的"契约类型"，所有检索模块的输入输出都围绕它们设计。

---

## 二、模块总览

| 编号 | 模块 | 文件 | 核心职责 |
|------|------|------|----------|
| D1 | QueryProcessor | `src/core/query_engine/query_processor.py` | 查询预处理：分词、停用词过滤、关键词提取 |
| D2 | DenseRetriever | `src/core/query_engine/dense_retriever.py` | 语义检索：Embedding 向量相似度查询 |
| D3 | SparseRetriever | `src/core/query_engine/sparse_retriever.py` | 关键词检索：BM25 倒排索引查询 |
| D4 | Fusion | `src/core/query_engine/fusion.py` | 结果融合：RRF 算法合并多个排名列表 |
| D5 | HybridSearch | `src/core/query_engine/hybrid_search.py` | 混合检索编排：串联 D1-D4 的核心流程 |
| D6 | Reranker | `src/core/query_engine/reranker.py` | 精排：Cross-Encoder 重排序 + 容错降级 |
| D7 | query.py | `scripts/query.py` | CLI 入口：命令行参数解析 + 格式化输出 |

---

## 三、数据流全景图

```
用户输入: query (str) + filters (Optional[Dict]) + top_k (int)
    |
    v
[D7] scripts/query.py :: run_query()
    |  load_settings() -> Settings
    |  构建 filters (e.g., collection)
    |  创建 TraceContext
    |
    v
[D5] HybridSearch :: search(query, top_k, filters, trace)
    |
    |--- [D1] QueryProcessor :: process(query, filters)
    |       |  extract_keywords(query)
    |       |    jieba.cut() 或 re.findall() -> 分词
    |       |    过滤停用词 + 短词 -> 去重保序
    |       |  parse_filters(filters) -> 规范化
    |       v
    |    ProcessedQuery {original, keywords, filters}
    |
    |--- [D2] DenseRetriever :: retrieve(query, top_k)
    |       |  EmbeddingClient.embed_single(query) -> query_vector
    |       |  VectorStore.query(vector, top_k, filters) -> List[QueryResult]
    |       v
    |    List[RetrievalResult]  (语义检索结果)
    |
    |--- [D3] SparseRetriever :: retrieve(keywords, top_k)
    |       |  BM25Indexer.query(keywords, top_k) -> List[(chunk_id, score)]
    |       |  VectorStore.get_by_ids(chunk_ids) -> 补全文本
    |       v
    |    List[RetrievalResult]  (BM25 检索结果)
    |
    |--- [D4] Fusion :: fuse([dense_results, sparse_results], top_k)
    |       |  RRF: score(d) = Σ 1/(k + rank_i(d)),  k=60
    |       |  合并去重 + 按 RRF 分数降序排序
    |       v
    |    List[RetrievalResult]  (融合结果)
    |
    |--- Metadata 后置过滤 (精确匹配 filters 中的 key-value)
    |--- Top-K 截断
    |
    v
 List[RetrievalResult]  (HybridSearch 最终结果)
    |
    v
[D6] Reranker :: rerank(query, candidates, top_k)
    |  RetrievalResult -> Candidate (类型转换)
    |  Backend.rerank(query, candidates, top_k) -> List[RankedCandidate]
    |  RankedCandidate -> RetrievalResult (类型转换)
    |  异常时 fallback: 保持原始排序
    v
 {"results": List[RetrievalResult], "fallback": bool, "elapsed_ms": float}
    |
    v
[D7] 格式化输出: score, chunk_id, source, text_preview
```

---

## 四、各模块详解

### D1 -- QueryProcessor

**职责**：将原始查询文本转换为结构化的 `ProcessedQuery`。

**核心方法**：
- `process(query, filters, trace)` → `ProcessedQuery` — 主入口
- `extract_keywords(query, trace)` → `List[str]` — 分词 + 停用词过滤

**分词策略**：
1. 优先使用 jieba（延迟加载），加载失败降级到正则分词
2. 正则模式：`r'[一-鿿]|[a-zA-Z]+|\d+'`
3. 停用词过滤：内置中英文混合停用词表（约 130 个）
4. 最小长度：默认 `min_keyword_length=2`，过滤单字符
5. 去重保序：用 `Set` 去重，保持首次出现顺序

---

### D2 -- DenseRetriever

**职责**：通过 Embedding 向量相似度执行语义检索。

**核心流程**：
```
query → embed_single(query) → query_vector → VectorStore.query(vector, top_k) → List[RetrievalResult]
```

**关键设计**：
- Embedding 和 VectorStore 均采用延迟创建（`_get_xxx()`）
- 通过工厂方法动态选择后端：`EmbeddingFactory.create` / `VectorStoreFactory.create`
- 性能计时：分别记录 embed 耗时和向量查询耗时
- 容错：空查询直接返回空列表

---

### D3 -- SparseRetriever

**职责**：通过 BM25 倒排索引执行关键词检索。

**核心流程**：
```
keywords → BM25Indexer.query(keywords, top_k) → [(chunk_id, score)]
         → VectorStore.get_by_ids(chunk_ids) → 补全文本和 metadata
         → List[RetrievalResult]
```

**BM25 公式**：
```
score(q,d) = Σ IDF(t) × (tf × (k1+1)) / (tf + k1 × (1 - b + b × |d| / avgdl))
```
- 参数：k1=1.5, b=0.75
- 持久化：pickle 格式，路径 `data/db/bm25/bm25_index.pkl`

**两步查询设计**：
- BM25 索引只存储 term weights 和 postings，不含完整文本
- 需要二次查询 VectorStore 通过 `get_by_ids` 补全 text 和 metadata

---

### D4 -- Fusion

**职责**：使用 RRF 算法融合多个排名列表。

**加权 RRF 公式**：
```
score(d) = Σ w_i × 1 / (k + rank_i(d))
```
- `k` 默认 60，控制排名靠后结果的权重衰减速度
- `rank` 从 0 开始（排名第一贡献 `1/60`）
- `w_i` 是各排名列表的权重（默认等权）

**配置**（`config/settings.yaml`）：

```yaml
retrieval:
  dense_weight: 0.7    # Dense 检索权重（语义相似度）
  sparse_weight: 0.3   # Sparse 检索权重（BM25 关键词匹配）
```

**分数示例**（权重 0.7/0.3）：
| 场景 | Dense 排名 | Sparse 排名 | RRF 分数 |
|------|-----------|-------------|----------|
| 双料冠军 | 第 1 | 第 1 | 0.7/60 + 0.3/60 = 0.0167 |
| Dense 偏好 | 第 1 | 第 2 | 0.7/60 + 0.3/61 ≈ 0.0166 |
| Sparse 偏好 | 第 2 | 第 1 | 0.7/61 + 0.3/60 ≈ 0.0165 |

**关键特性**：

- 支持加权 RRF，可通过配置调整 Dense/Sparse 的重要程度
- 权重可在初始化时设置，也可在调用时动态覆盖
- 单路失败时自动调整权重（如 Dense 失败，只有 Sparse 权重生效）
- 分数范围 (0, max_weight/k]，确定性输出

---

### D5 -- HybridSearch

**职责**：D 阶段的编排核心，串联 D1-D4 的完整混合检索流程。

**search 方法的 6 步流程**：
1. `QueryProcessor.process(query, filters)` → `ProcessedQuery`
2. `DenseRetriever.retrieve(query, top_k)` → 语义检索结果（容错降级）
3. `SparseRetriever.retrieve(processed.keywords, top_k)` → BM25 检索结果（容错降级）
4. `Fusion.fuse(rankings, top_k)` → RRF 融合结果（只融合非空列表）
5. `_apply_metadata_filters(fused_results, filters)` → 后置 metadata 过滤
6. Top-K 截断 → 最终结果

**关键设计**：
- Dense/Sparse 检索器均有独立容错，一方失败不影响另一方
- Metadata 过滤采用后置策略（先融合再过滤），保证 RRF 融合质量
- 所有子组件支持依赖注入，便于测试

---

### D6 -- Reranker

**职责**：使用 Cross-Encoder 对检索结果精排。

**核心流程**：
```
RetrievalResult → Candidate → backend.rerank() → RankedCandidate → RetrievalResult
```

**返回值结构**：
```python
{
    "results": List[RetrievalResult],  # 精排后的结果
    "fallback": bool,                   # 是否触发了降级
    "elapsed_ms": float                 # 精排耗时
}
```

**容错机制**：
- 后端异常时自动回退到原始排序
- 标记 `fallback=True` 供上层感知
- 默认使用 `NoneReranker`（保持原始顺序）

---

### D7 -- query.py

**职责**：CLI 命令行入口，串联完整查询流程。

**命令行参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--query` | str | 必填 | 查询文本 |
| `--top-k` | int | 10 | 返回结果数 |
| `--collection` | str | None | 限定检索集合 |
| `--verbose` | flag | False | 显示中间结果 |
| `--no-rerank` | flag | False | 跳过 Reranker |
| `--config` | str | `config/settings.yaml` | 配置文件路径 |

**使用示例**：
```bash
# 基本查询
python scripts/query.py --query "如何配置 Ollama？"

# 指定 top-k + verbose
python scripts/query.py --query "什么是 RAG？" --top-k 5 --verbose

# 跳过 reranker
python scripts/query.py --query "语言模型" --no-rerank

# 限定集合
python scripts/query.py --query "Rouge-N是什么" --collection my_docs
```

---

## 五、设计模式总结

| 模式 | 应用场景 | 说明 |
|------|----------|------|
| **延迟创建** | D2/D3/D5/D6 | 首次调用时才实例化依赖组件 |
| **工厂模式** | D2/D6 | 根据配置动态选择后端实现 |
| **抽象基类** | D2/D3/D6 | 统一接口，支持多态替换 |
| **容错降级** | D5/D6 | 组件失败时降级而非崩溃 |
| **Trace 贯穿** | 全链路 | 记录各阶段耗时和指标 |
| **类型契约** | 全链路 | ProcessedQuery/RetrievalResult 作为数据交换格式 |

---

## 六、模块间调用关系

```
                    ┌──────────────┐
                    │   D7 query.py │
                    └──────┬───────┘
                           │ 调用
                    ┌──────▼───────┐
                    │ D5 HybridSearch│
                    └──┬───┬───┬──┘
            ┌──────────┘   │   └──────────┐
            ▼              ▼              ▼
     ┌──────────┐   ┌──────────┐   ┌──────────┐
     │ D1 Query │   │ D2 Dense │   │ D3 Sparse│
     │ Processor│   │ Retriever│   │ Retriever│
     └──────────┘   └────┬─────┘   └────┬─────┘
                         │              │
                         ▼              ▼
                  ┌──────────┐   ┌──────────┐
                  │Embedding │   │  BM25    │
                  │ Client   │   │ Indexer  │
                  └──────────┘   └──────────┘
                         │              │
                         └──────┬───────┘
                                ▼
                         ┌──────────┐
                         │VectorStore│
                         └──────────┘
                                │
                    ┌───────────┘
                    ▼
             ┌──────────┐
             │ D4 Fusion │
             │  (RRF)    │
             └──────────┘
                    │
                    ▼
             ┌──────────┐
             │ D6 Rerank │
             └──────────┘
```
