# Scaling & Model Alternatives — Technical Documentation

This document evaluates alternative models and architecture improvements for the Charter Ingestion Pipeline, covering tradeoffs in accuracy, latency, cost, and production readiness.

---

## Table of Contents

1. [Current Architecture Summary](#current-architecture-summary)
2. [VLM Alternatives (Image Summarization)](#vlm-alternatives-image-summarization)
3. [Embedding Model Alternatives](#embedding-model-alternatives)
4. [Reranker Alternatives](#reranker-alternatives)
5. [Vector Database Scaling](#vector-database-scaling)
6. [Production Architecture Recommendations](#production-architecture-recommendations)

---

## Current Architecture Summary

| Component | Current Model | Size | Where It Runs |
|-----------|--------------|------|---------------|
| VLM (Image Summarization) | `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` | 8B | NVIDIA NIM Cloud |
| Embeddings | `nvidia/llama-nemotron-embed-1b-v2` | 1B | NVIDIA NIM Cloud |
| Answer Synthesis | `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` | 8B | NVIDIA NIM Cloud |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | Local CPU |
| Keyword Search | BM25 (rank-bm25) | — | Local (in-memory) |
| Search Strategy | Hybrid (Semantic + BM25 + RRF) | — | query.py |
| Vector Store | PGVector (PostgreSQL) | — | Docker Container |

---

## VLM Alternatives (Image Summarization)

### Why Consider Alternatives?

The `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` model provides excellent accuracy for document understanding, but if you need alternatives:
- **~2-4 seconds per image** — for 300+ images, that's ~15-25 minutes per PDF
- **Rate limited** at 40 RPM on NVIDIA NIM free tier
- For even faster processing, smaller models like `qwen2.5-vl-7b` can be considered

### Comparison Table

| Model | Total Params | Active Params | Disk Size | Vision? | Speed (est.) | Quality | NIM Available | Best For |
|-------|-------------|---------------|-----------|---------|-------------|---------|---------------|----------|
| **nvidia/llama-3.1-nemotron-nano-vl-8b-v1** (current) | ~8B | ~8B | ~16 GB | ✅ | ~2-4s/img | ★★★★ | ✅ | Fast document understanding |
| **qwen/qwen3.5-122b-a10b** (previous) | 122B | 10B | ~240 GB | ✅ | ~5-10s/img | ★★★★★ | ✅ | Maximum accuracy, complex charts |
| **qwen/qwen3.5-397b-a17b** | 397B | 17B | ~780 GB | ✅ | ~15-30s/img | ★★★★★+ | ✅ | Research-grade quality, very slow |
| **meta/llama-3.2-11b-vision-instruct** | 11B | 11B | ~22 GB | ✅ | ~3-5s/img | ★★★★ | ✅ | General-purpose, well-balanced |
| **qwen/qwen2.5-vl-7b-instruct** | 7B | 7B | ~14 GB | ✅ | ~2-3s/img | ★★★½ | ❓ | Fastest, budget-friendly |

### Detailed Tradeoffs

#### Option A: `nvidia/llama-nemotron-nano-vl` (Recommended Alternative)

```
Accuracy:  ★★★★ (90% of current quality)
Speed:     ★★★★★ (2-3x faster)
Cost:      ★★★★★ (smaller model = fewer GPU-hours)
```

**Pros:**
- Purpose-built by NVIDIA for **document understanding** — tables, charts, infographics
- Optimized for NIM deployment (first-class support)
- ~8B parameters — fast inference, low latency
- Open-source weights available on HuggingFace (can self-host)

**Cons:**
- May miss subtle details in very complex multi-layered charts
- Less capable at general reasoning compared to 122B model
- Newer model — less community testing

**When to choose:** You prioritize **throughput** (processing large volumes of PDFs quickly) over maximum accuracy on individual images.

#### Option B: `meta/llama-3.2-11b-vision-instruct`

```
Accuracy:  ★★★★ (comparable to nemotron-nano)
Speed:     ★★★★ (~3-5s per image)
Cost:      ★★★★ (medium)
```

**Pros:**
- Well-tested, large community, extensive benchmarks
- Strong at OCR-like tasks (reading text from images)
- Available on multiple platforms (not locked to NVIDIA NIM)

**Cons:**
- 11B dense (all parameters active) vs 10B active in current MoE model
- Not specifically optimized for document/chart understanding
- Slightly slower than nemotron-nano

**When to choose:** You want a **well-established model** with broad compatibility and don't need the latest optimizations.

#### Option C: `qwen/qwen3.5-397b-a17b` (Upgrade Path)

```
Accuracy:  ★★★★★+ (state-of-the-art)
Speed:     ★★ (15-30s per image — 3x slower)
Cost:      ★★ (very expensive to self-host)
```

**When to choose:** You need **the absolute best accuracy** and processing time is not a concern (small batch of critical documents).

### Summary: Which VLM to Pick?

```
                    Speed
                     ▲
                     │
     qwen2.5-7b ●   │
                     │   ● nemotron-nano-vl (RECOMMENDED)
                     │
                     │       ● llama-3.2-11b
                     │
                     │           ● qwen3.5-122b (CURRENT)
                     │
                     │                   ● qwen3.5-397b
                     └──────────────────────────── ▶ Accuracy
```

> **Current choice:** We switched to `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` for **2-3x speed improvement** with ~90% of the previous qwen3.5 accuracy. For document images, charts, and tables, the quality difference is minimal. To upgrade further, consider `qwen3.5-122b-a10b` for maximum accuracy on complex charts.

---

## Embedding Model Alternatives

### Current: `nvidia/llama-nemotron-embed-1b-v2`

| Attribute | Value |
|-----------|-------|
| Parameters | 1B |
| Type | Asymmetric (passage/query) |
| Optimized For | Retrieval (RAG) |
| Runs On | NVIDIA NIM (cloud API) |

### Alternatives

| Model | Params | Disk Size | Dims | Type | Multilingual | Self-Hostable | Best For |
|-------|--------|-----------|------|------|-------------|---------------|----------|
| **llama-nemotron-embed-1b-v2** (current) | 1B | ~2 GB | dynamic | Asymmetric | ✅ | ✅ | RAG retrieval |
| **llama-embed-nemotron-8b** | 8B | ~16 GB | 4096 | Asymmetric | ✅ | ✅ | Highest quality retrieval |
| **nv-embedqa-e5-v5** | ~300M | ~600 MB | 1024 | Symmetric | ⚠️ English focus | ✅ | Fast, QA-specific retrieval |
| **BAAI/bge-m3** | 567M | ~1.1 GB | 1024 | Hybrid | ✅ 100+ langs | ✅ Local | Multilingual, open-source |
| **text-embedding-3-large** (OpenAI) | Unknown | N/A (cloud) | 3072 | Symmetric | ✅ | ❌ Cloud only | General purpose, easy setup |

### Detailed Tradeoffs

#### Upgrade: `nvidia/llama-embed-nemotron-8b`

```
Quality:   ★★★★★ (#1 on MTEB benchmark)
Speed:     ★★★ (slower — 8B vs 1B)
Cost:      ★★★ (8x more compute)
```

**When to choose:** You need **maximum retrieval accuracy** and can afford higher latency and cost. Best when documents contain highly technical or nuanced content where retrieval precision is critical.

#### Lightweight: `nvidia/nv-embedqa-e5-v5`

```
Quality:   ★★★★ (optimized for QA retrieval)
Speed:     ★★★★★ (very fast — 300M params)
Cost:      ★★★★★ (cheapest option)
```

**When to choose:** Your documents are primarily **English text** and you want the fastest possible embedding with good QA retrieval quality. This is a symmetric model — no `input_type` handling needed (simpler code).

#### Open-Source: `BAAI/bge-m3`

```
Quality:   ★★★★ (strong multilingual)
Speed:     ★★★★ (runs locally on CPU/GPU)
Cost:      ★★★★★ (free — no API calls)
```

**When to choose:** You want to **eliminate API dependency entirely**, need multilingual support, or are working in air-gapped/offline environments.

---

## Reranker Alternatives

### Current: `cross-encoder/ms-marco-MiniLM-L-6-v2`

| Attribute | Value |
|-----------|-------|
| Parameters | 22M (6 layers) |
| Runs On | Local CPU |
| Latency | ~200-500ms for 20 pairs |
| Quality | Good for English retrieval |

### Why It Matters

The reranker is the **precision layer** — it converts "approximately relevant" results from vector search into "actually relevant" results. Removing it degrades answer quality significantly.

### Alternatives

| Model | Params | Disk Size | Speed (20 pairs) | Quality | GPU Required? | Best For |
|-------|--------|-----------|------------------|---------|---------------|----------|
| **ms-marco-MiniLM-L-6-v2** (current) | 22M | ~90 MB | ~300ms (CPU) | ★★★★ | ❌ | Cost-effective, good quality |
| **ms-marco-TinyBERT-L-2-v2** | 4.4M | ~17 MB | ~60ms (CPU) | ★★★ | ❌ | Maximum speed, basic quality |
| **ms-marco-MiniLM-L-12-v2** | 33M | ~130 MB | ~500ms (CPU) | ★★★★½ | ❌ | Better quality, same speed class |
| **bge-reranker-v2-m3** | 568M | ~1.1 GB | ~100ms (GPU) | ★★★★★ | ⚠️ Recommended | Best quality, multilingual |
| **ColBERT v2** | 110M | ~440 MB | ~10ms (pre-computed) | ★★★★½ | ✅ | Production scale, lowest latency |
| **NVIDIA NIM Reranking API** | Hosted | N/A (cloud) | ~50ms (network) | ★★★★★ | N/A (cloud) | Simplest, no local model |

### Key Architectural Choice: Cross-Encoder vs ColBERT

#### Cross-Encoder (Current Approach)

```
Query: "SLA timelines"  ──┐
                           ├──►  [Cross-Encoder]  ──►  Score: 8.7
Chunk: "SLA is 4 hours" ──┘

Query: "SLA timelines"  ──┐
                           ├──►  [Cross-Encoder]  ──►  Score: 1.2
Chunk: "Network setup"  ──┘

Total: 20 pairs × ~15ms each = ~300ms
```

- Processes query+document **together** (joint encoding)
- Very accurate but **must recompute for every query**
- Latency scales linearly with number of candidates: `O(k)`

#### ColBERT (Production Alternative)

```
Pre-computed (at ingestion time):
  Chunk: "SLA is 4 hours"  ──►  [token_emb_1, token_emb_2, token_emb_3, ...]

At query time:
  Query: "SLA timelines"   ──►  [query_tok_1, query_tok_2, ...]
  
  Score = MaxSim(query_tokens, chunk_tokens)  ──►  8.7    (~1ms per chunk!)
```

- Pre-computes **token-level embeddings** for documents during ingestion
- At query time, only the query is encoded — scoring uses fast token matching
- **~30x faster** than cross-encoder at query time
- Tradeoff: Higher storage (token embeddings per chunk) and slightly lower accuracy

### Recommendation

| Scale | Recommended Reranker |
|-------|---------------------|
| **< 100K chunks** (your case) | Keep `MiniLM-L-6-v2` (current) — simple, fast on CPU |
| **100K–1M chunks, low QPS** | Upgrade to `bge-reranker-v2-m3` on GPU |
| **1M+ chunks, high QPS** | Switch to ColBERT v2 or NVIDIA NIM Reranking API |

---

## Vector Database Scaling

### Current: PGVector (PostgreSQL)

PGVector uses **exact nearest neighbor** search by default — it compares the query vector against every stored vector. This works well up to ~100K vectors.

### Scaling Options

| Scale | Solution | Search Latency | Accuracy | Complexity |
|-------|----------|---------------|----------|------------|
| **< 100K** | PGVector (brute force) | ~50ms | 100% (exact) | ★ Simple |
| **100K–1M** | PGVector + IVFFlat index | ~10ms | ~95% | ★★ |
| **1M–10M** | PGVector + HNSW index | ~5ms | ~98% | ★★ |
| **10M–100M** | Dedicated DB (Milvus/Qdrant) | ~2ms | ~97% | ★★★ |
| **100M+** | Distributed (Pinecone/Weaviate) | ~5ms (managed) | ~97% | ★★★★ |

### PGVector Index Options (No Migration Needed)

You can add an index to your existing PGVector setup — no data migration required:

#### IVFFlat Index (Inverted File Index)

```sql
-- Creates ~100 clusters, searches nearest 10 clusters
CREATE INDEX ON langchain_pg_embedding 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- At query time, increase accuracy by probing more clusters:
SET ivfflat.probes = 20;
```

- **How it works:** Clusters vectors into groups, only searches nearby clusters
- **Tradeoff:** Must be rebuilt after large inserts, slight accuracy loss
- **Speed:** ~5-10x faster than brute force

#### HNSW Index (Hierarchical Navigable Small World)

```sql
-- Creates a graph-based index for fast approximate search
CREATE INDEX ON langchain_pg_embedding 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 200);

-- At query time, increase accuracy:
SET hnsw.ef_search = 100;
```

- **How it works:** Builds a multi-layer graph connecting similar vectors
- **Tradeoff:** Slower to build, uses more memory, but much faster at search
- **Speed:** ~10-50x faster than brute force, higher accuracy than IVFFlat

### When to Migrate Away from PGVector

| Signal | Action |
|--------|--------|
| Search latency > 500ms | Add HNSW index |
| Search latency > 2s with HNSW | Migrate to Milvus/Qdrant |
| Need auto-scaling / multi-region | Migrate to Pinecone/Weaviate (managed) |
| Need hybrid search (text + vector) | ✅ Already implemented (BM25 + semantic + RRF) |

---

## Production Architecture Recommendations

### Tier 1: Quick Wins (No Code Changes)

These improvements can be applied **today** with minimal effort:

| Change | Impact | Effort |
|--------|--------|--------|
| Add HNSW index to PGVector | 10-50x faster search | 1 SQL command |
| Switch VLM to `nemotron-nano-vl` | 2-3x faster ingestion | Change 1 config string |
| Use `MiniLM-L-12-v2` reranker | ~15% better relevance | Change 1 config string |

### Tier 2: Moderate Improvements (Some Code Changes)

| Change | Impact | Effort |
|--------|--------|--------|
| Batch embedding with retry/backoff | More reliable ingestion | ~50 lines |
| Add GPU-based reranking | 5-10x faster reranking | Install CUDA, config change |
| Parallel image summarization | 2-4x faster ingestion | ~100 lines (async/threading) |
| Incremental ingestion (don't delete old) | Support multiple PDFs | Modify `pre_delete_collection` logic |

### Tier 3: Production-Grade (Architecture Changes)

| Change | Impact | Effort |
|--------|--------|--------|
| Replace PGVector with Milvus/Qdrant | Sub-10ms search at any scale | New infra, migration |
| ColBERT reranker | Near-zero reranking latency | New model pipeline |
| Self-host models (NVIDIA NIM containers) | No rate limits, full control | GPU infrastructure |
| Async ingestion pipeline (Celery/RQ) | Process PDFs in background | Significant refactor |
| Caching layer (Redis) | Instant response for repeat queries | Add Redis, cache logic |

### Recommended Production Architecture

```
                                ┌──────────────────────────────────────────────┐
                                │              INGESTION PIPELINE              │
                                │                                              │
  PDF Upload ──►  Task Queue ──►│  PDFExtractor                                │
  (API/S3)       (Celery/RQ)    │       │                                      │
                                │       ▼                                      │
                                │  ImageSummarizer (nemotron-nano-vl, self-hosted)
                                │       │                                      │
                                │       ▼                                      │
                                │  Reassembler → SmartChunker                  │
                                │       │                                      │
                                │       ▼                                      │
                                │  Embed (llama-embed-nemotron-8b)             │
                                │       │                                      │
                                │       ▼                                      │
                                │  Milvus/Qdrant (HNSW index)  ◄──────────────┤
                                └──────────────────────────────────────────────┘

                                ┌──────────────────────────────────────────────┐
                                │              QUERY PIPELINE                  │
                                │                                              │
  User Query ──► API Gateway ──►│  Embed query (nemotron-embed)                │
                    │           │       │                                      │
                    │           │       ▼                                      │
                    │           │  Milvus ANN Search (top 50)     ~2ms         │
                    │           │       │                                      │
                    │           │       ▼                                      │
                    │           │  ColBERT Reranker (top 5)        ~10ms       │
                    │           │       │                                      │
                    ▼           │       ▼                                      │
              Redis Cache ◄────│  LLM Synthesis (cached?)         ~3-5s       │
                    │           │       │                                      │
                    ▼           │       ▼                                      │
              Response ◄───────│  Formatted Answer                             │
                                └──────────────────────────────────────────────┘

  Total Query Latency:  ~3-5 seconds (mostly LLM generation)
  vs Current:           ~5-10 seconds
```

---

## Cost Analysis

### Current (NVIDIA NIM Free Tier)

| Component | Cost | Limit |
|-----------|------|-------|
| VLM (nemotron-nano-vl-8b) | Free | ~1000 credits/key |
| Embeddings (nemotron-embed) | Free | ~1000 credits/key |
| Reranker (MiniLM) | Free | Runs locally |
| BM25 Keyword Search | Free | Runs locally (in-memory) |
| PGVector | Free | Self-hosted Docker |
| **Total** | **$0** | Limited by API credits |

### Estimated Production Costs (Self-Hosted)

| Component | GPU Required | Monthly Cost (Cloud) |
|-----------|-------------|---------------------|
| VLM: nemotron-nano-vl (8B) | 1× A10G (24GB) | ~$500-800 |
| VLM: qwen3.5-122b (current) | 4× A100 (80GB) | ~$6,000-10,000 |
| Embeddings: nemotron-embed-1b | 1× T4 (16GB) | ~$200-300 |
| Reranker: ColBERT | 1× T4 (16GB) | ~$200-300 |
| Vector DB: Milvus | CPU + RAM | ~$200-400 |
| **Total (with nano-vl)** | | **~$1,100-1,800/mo** |
| **Total (with 122b)** | | **~$6,600-11,000/mo** |

> Switching from `qwen3.5-122b` to `nemotron-nano-vl` saves **~$5,000-9,000/month** in production with minimal quality loss.

---

## Decision Matrix

Use this matrix to choose the right configuration for your use case:

| Priority | VLM | Embeddings | Reranker | Search | Vector DB |
|----------|-----|-----------|---------|---------|-----------|
| **Maximum Accuracy** | qwen3.5-122b | llama-embed-nemotron-8b | bge-reranker-v2-m3 | Hybrid (current) | PGVector + HNSW |
| **Best Speed/Quality Balance** | nemotron-nano-vl (current) | nemotron-embed-1b (current) | MiniLM-L-6-v2 (current) | Hybrid (current) | PGVector + HNSW |
| **Lowest Latency** | nemotron-nano-vl (current) | nv-embedqa-e5-v5 | ColBERT v2 | Hybrid (current) | Milvus |
| **Lowest Cost** | qwen2.5-vl-7b | bge-m3 (local) | MiniLM-L-6-v2 (current) | Hybrid (current) | PGVector |
| **Air-Gapped / Offline** | nemotron-nano-vl (self-hosted) | bge-m3 (local) | MiniLM-L-6-v2 (current) | Hybrid (current) | PGVector |
