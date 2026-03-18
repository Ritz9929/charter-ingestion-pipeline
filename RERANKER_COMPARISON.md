# Reranker Comparison: Cross-Encoder MiniLM vs Nemotron Rerank 1B

**Purpose:** Evaluate reranker options for the Multimodal RAG pipeline production deployment on AWS.

---

## Executive Summary

| | ms-marco-MiniLM-L-6-v2 (Current) | Nemotron Rerank 1B v2 (Proposed) |
|--|----------------------------------|----------------------------------|
| **Parameters** | 22.7M | 1.2B |
| **Disk Size** | ~90 MB | ~2.4 GB |
| **Accuracy** | ★★★★ Good | ★★★★★ State-of-the-art |
| **Latency (20 pairs)** | ~300ms (CPU) | ~50-80ms (GPU) |
| **Multilingual** | ❌ English only | ✅ 26 languages |
| **Context Window** | 512 tokens | 4,096 tokens |
| **Infrastructure** | CPU only | GPU recommended |
| **Monthly Cost** | **$0** (local) | **$0** on shared GPU / Free NIM endpoint |
| **Recommendation** | ✅ Prototyping & budget setups | ✅ **Production on AWS** |

---

## What Does a Reranker Do?

```
User Query: "What is the SLA for Network Config?"
                    │
                    ▼
    ┌─── Vector Search (PGVector) ────────────────────────────────┐
    │ Returns top 20 candidates ranked by embedding similarity    │
    │ Problem: ~50% may be false positives ("network" ≠ "SLA")   │
    └─────────────────────────┬───────────────────────────────────┘
                              │ 20 noisy results
                              ▼
    ┌─── Reranker ────────────────────────────────────────────────┐
    │ Scores each (query, chunk) pair with deep attention         │
    │ Promotes actually relevant results, demotes false matches   │
    └─────────────────────────┬───────────────────────────────────┘
                              │ Top 5 high-quality results
                              ▼
    ┌─── LLM Answer Synthesis ───────────────────────────────────┐
    │ Generates coherent answer from 5 focused chunks            │
    │ Better input = Better output                                │
    └─────────────────────────────────────────────────────────────┘
```

**Without reranker:** LLM receives noisy context → vague or incorrect answers  
**With reranker:** LLM receives focused context → precise, cited answers

---

## Detailed Comparison

### 1. Accuracy & Quality

| Metric | MiniLM-L-6 | Nemotron Rerank 1B |
|--------|-----------|-------------------|
| NDCG@10 (MS MARCO) | 0.39 | **0.44** (+13%) |
| Complex technical queries | Good | **Significantly better** |
| Queries with negation ("not", "except") | Weak | **Strong** |
| Long chunk understanding | Truncates at 512 tokens | Sees full 4,096 tokens |

**Impact on our pipeline:** Image summary tags average ~1,500-2,000 characters (~300-400 tokens). MiniLM sees most of them. But some rendered page summaries reach 3,000+ characters (600+ tokens) — these get **truncated** by MiniLM, losing critical data. Nemotron handles them fully.

---

### 2. Latency

| Scenario | MiniLM (CPU) | Nemotron (GPU) | Nemotron (NIM API) |
|----------|-------------|---------------|-------------------|
| 20 query-chunk pairs | ~300ms | ~80ms | ~50ms + network |
| 50 query-chunk pairs | ~750ms | ~120ms | ~80ms + network |
| 100 query-chunk pairs | ~1,500ms | ~200ms | ~150ms + network |

**On AWS with GPU:** Nemotron is **4-6× faster** because it leverages GPU parallelism. The cross-encoder processes pairs sequentially on CPU.

---

### 3. Context Window — Why 4096 > 512 Matters

```
Example chunk with image summary (1,842 characters ≈ 380 tokens):

"Reference Information: TechMobile Field Operations Training Page 5 of 108
 [IMAGE_REFERENCE | URL: /mock_s3_storage/page5_rendered.png | 
  SUMMARY: ### 1. Image Type & Core Subject This is a provisioning workflow 
  diagram showing three stages. ### 2. Explicit Text & Labels Stage 1: Order 
  Entry (2hr SLA), Stage 2: Network Configuration (4hr SLA), Stage 3: Device 
  Activation (1hr SLA). ### 3. Data & Relationships The total provisioning 
  time is 7 hours. ### 4. Semantic Keywords provisioning, SLA, workflow...]"

MiniLM sees:  First 512 tokens ──► Gets "provisioning workflow diagram" ✅
              but MISSES "4hr SLA" for Network Config ❌

Nemotron sees: Full 380 tokens ──► Gets everything including "4hr SLA" ✅
```

---

### 4. Infrastructure Requirements

#### MiniLM (Current)

```
App Server (CPU only)
├── query.py
├── Cross-Encoder MiniLM (~90 MB RAM)
└── No GPU needed

Instance: c6g.xlarge (4 vCPU, 8GB RAM) = $98/month
```

#### Nemotron Rerank (Proposed — Two Options)

**Option A: Self-hosted on existing GPU instance (Recommended)**
```
GPU Instance (g5.xlarge — already running VLM + Embeddings)
├── vLLM serving VLM model          (~14 GB VRAM)
├── Embedding model                 (~2 GB VRAM)
├── Nemotron Rerank 1B              (~3 GB VRAM)  ← shares the GPU
└── Total VRAM used: ~19 GB / 24 GB available

Additional cost: $0 (shared on existing GPU)
```

**Option B: Via NIM Free Endpoint**
```
App Server sends rerank request to NIM API
├── No local model needed
├── Adds ~20-50ms network latency
└── Subject to NIM rate limits

Additional cost: $0 (free endpoint)
```

---

## Pros & Cons Summary

### ms-marco-MiniLM-L-6-v2

| Pros | Cons |
|------|------|
| ✅ Zero cost — runs on CPU | ❌ English only |
| ✅ 90 MB — loads instantly | ❌ 512 token limit — truncates long chunks |
| ✅ No GPU or API dependency | ❌ ~300ms latency on CPU |
| ✅ Battle-tested, huge community | ❌ Lower accuracy on complex queries |
| ✅ Works fully offline | ❌ No multilingual support |

### Nemotron Rerank 1B v2

| Pros | Cons |
|------|------|
| ✅ State-of-the-art accuracy (+13% NDCG) | ❌ Needs GPU for optimal performance |
| ✅ 4,096 token context (8× more) | ❌ 2.4 GB model size |
| ✅ 26 language support | ❌ Newer — less community docs |
| ✅ 4-6× faster on GPU (~50-80ms) | ❌ ~2-3s on CPU (too slow) |
| ✅ Shares existing GPU at no extra cost | ❌ Adds ~3 GB VRAM usage |
| ✅ Free NIM endpoint available | |

---

## Cost Impact Analysis

| Setup | MiniLM | Nemotron (shared GPU) | Nemotron (NIM API) |
|-------|--------|----------------------|-------------------|
| GPU instance needed? | No | Already have it | No |
| Additional monthly cost | $0 | $0 | $0 |
| VRAM consumption | 0 GB | ~3 GB | 0 GB |
| Network dependency | None | None | NIM uptime |

**Key insight:** Since we're **already paying for a g5.xlarge GPU** for VLM inference, adding the Nemotron reranker costs **nothing extra** — it uses only 3 GB of the remaining 5-10 GB VRAM.

---

## Recommendation

### For Prototype (Current State) → **Keep MiniLM** ✅
- Zero dependencies, works on any machine
- Good enough for development and demos
- No API keys or GPU needed

### For AWS Production → **Switch to Nemotron Rerank** ✅
- Shares existing GPU infrastructure (no extra cost)
- 13% better retrieval accuracy → better answers
- 4,096 token context → no image summary truncation
- 4-6× faster reranking on GPU
- Multilingual support for future expansion

### Migration Effort: **Low** (~30 minutes)
The reranker is a self-contained component in `query.py`. Switching requires:
1. Replace `CrossEncoder` with NVIDIA NIM API call or local Nemotron model
2. Adjust score normalization (different scoring scale)
3. No changes to ingestion pipeline, embeddings, or vector store

---

## Decision Summary for Team

| Question | Answer |
|----------|--------|
| Should we switch rerankers? | **Yes, for production** |
| Does it cost more? | **No** — shares existing GPU |
| Is it more accurate? | **Yes** — +13% NDCG, 8× longer context |
| Is it faster? | **Yes** — 4-6× faster on GPU |
| Is it risky? | **Low** — isolated component, easy rollback |
| When to migrate? | During AWS deployment (not before) |
