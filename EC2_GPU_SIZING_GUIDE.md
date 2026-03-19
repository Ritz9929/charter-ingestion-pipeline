# EC2 GPU Sizing Guide — VLM + Embedding on a Single Instance

A detailed guide for deploying both **nemotron-nano-vl-8b** (VLM) and **nemotron-embed-1b** (embeddings) on a single AWS EC2 GPU instance, with deep dives into VRAM math, KV caching, PagedAttention, and vLLM configuration.

---

## Table of Contents

1. [VRAM Budget Breakdown](#vram-budget-breakdown)
2. [What is KV Caching?](#what-is-kv-caching)
3. [What is PagedAttention?](#what-is-pagedattention)
4. [EC2 Instance Options](#ec2-instance-options)
5. [vLLM Setup Guide](#vllm-setup-guide)
6. [Multi-Model Serving Strategies](#multi-model-serving-strategies)
7. [Cost Analysis](#cost-analysis)
8. [Recommendations](#recommendations)

---

## VRAM Budget Breakdown

### Your Models

| Model | Role | Parameters | Precision | Weight Size |
|-------|------|-----------|-----------|-------------|
| `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` | VLM (image summarization + answer synthesis) | 8B | FP16 | **~16 GB** |
| `nvidia/llama-nemotron-embed-1b-v2` | Embedding (text → vector) | 1B | FP16 | **~2 GB** |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker | 22M | FP32 | **~88 MB** (CPU) |
| BM25 (rank-bm25) | Keyword search | — | — | **~50 MB** (CPU RAM) |

### How VRAM Gets Used

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU VRAM (e.g., 24 GB A10G)                  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  MODEL WEIGHTS (static, loaded once at startup)          │   │
│  │                                                          │   │
│  │  VLM (8B × 2 bytes FP16)         = 16.0 GB              │   │
│  │  Embedding (1B × 2 bytes FP16)   =  2.0 GB              │   │
│  │                                   ─────────              │   │
│  │  Subtotal:                         18.0 GB               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  KV CACHE (dynamic, grows with each request)             │   │
│  │                                                          │   │
│  │  Per request (8K context):        ~1.0 GB                │   │
│  │  5 concurrent requests:           ~5.0 GB                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  OVERHEAD (CUDA kernels, activations, buffers)           │   │
│  │                                                          │   │
│  │  CUDA context + kernel overhead:  ~0.5 GB                │   │
│  │  Activation memory (inference):   ~0.3 GB                │   │
│  │  Embedding inference buffer:      ~0.2 GB                │   │
│  │                                   ─────────              │   │
│  │  Subtotal:                         1.0 GB                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  TOTAL MINIMUM (1 request):   ~20.0 GB                          │
│  TOTAL WITH 5 concurrent:     ~24.0 GB  ← exactly 24 GB GPU!   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### VRAM Math Formula

```
Total VRAM = Model Weights + KV Cache + Overhead

Model Weights = Parameters × Bytes_per_param
  FP16: 8B × 2 = 16 GB
  INT8: 8B × 1 = 8 GB
  INT4: 8B × 0.5 = 4 GB

KV Cache (per request) = 2 × num_layers × num_kv_heads × head_dim × seq_len × bytes_per_param
  For nemotron-nano-vl-8b (Llama 3.1 architecture):
    num_layers = 32
    num_kv_heads = 8 (GQA — Grouped Query Attention)
    head_dim = 128
    seq_len = 8192 (max context)

  KV per request = 2 × 32 × 8 × 128 × 8192 × 2 bytes
                 = 2 × 32 × 8 × 128 × 8192 × 2
                 = ~1.07 GB

Overhead ≈ 0.5–1.0 GB (CUDA kernels, activation buffers)
```

> **Key insight:** Each concurrent request adds ~1 GB of KV cache VRAM. On a 24 GB GPU with 18 GB of weights, you have ~5 GB for KV cache = **~4-5 concurrent requests max**.

---

## What is KV Caching?

### The Problem: Redundant Computation

When a language model generates text, it uses the **Transformer attention mechanism**. For each new token, the model needs to look at ALL previous tokens. Without caching, this means **recomputing** the attention for every previous token at every step:

```
Generating "The cat sat on the mat"

Step 1: Process "The"
  Compute attention for: [The]                    → 1 computation

Step 2: Process "cat"
  Compute attention for: [The, cat]               → 2 computations

Step 3: Process "sat"
  Compute attention for: [The, cat, sat]          → 3 computations

Step 4: Process "on"
  Compute attention for: [The, cat, sat, on]      → 4 computations

...

Step N: Process token N
  Compute attention for: [all N tokens]           → N computations

Total work WITHOUT caching: 1 + 2 + 3 + ... + N = N(N+1)/2 = O(N²)  🐌
```

### The Solution: Cache Keys and Values

In the Transformer attention mechanism, each token produces three vectors:
- **Q (Query):** "What am I looking for?"
- **K (Key):** "What do I contain?"
- **V (Value):** "What information do I return?"

The attention score is: `Attention = softmax(Q × K^T) × V`

**KV Caching** stores the K and V vectors of all previously processed tokens. When generating the next token, only the NEW token's Q, K, V need to be computed — the previous K and V are read from cache:

```
Step 1: Process "The"
  Compute Q₁, K₁, V₁
  Cache: K=[K₁], V=[V₁]                          → 1 computation

Step 2: Process "cat"
  Compute Q₂, K₂, V₂
  Read K₁, V₁ from cache
  Cache: K=[K₁,K₂], V=[V₁,V₂]                   → 1 computation ✨

Step 3: Process "sat"
  Compute Q₃, K₃, V₃
  Read K₁,K₂, V₁,V₂ from cache
  Cache: K=[K₁,K₂,K₃], V=[V₁,V₂,V₃]            → 1 computation ✨

Total work WITH KV caching: N computations = O(N)  ⚡
```

### Why KV Cache Is Memory-Hungry

```
KV Cache size per token per layer = 2 × num_kv_heads × head_dim × bytes

For nemotron-nano-vl-8b:
  Per token, per layer:  2 × 8 × 128 × 2 bytes = 4,096 bytes = 4 KB
  Per token, all layers: 4 KB × 32 layers = 128 KB
  Per request (8K ctx):  128 KB × 8,192 tokens = 1,048,576 KB ≈ 1 GB

  5 concurrent requests: 5 × 1 GB = 5 GB of VRAM just for KV cache!
```

### Visual: Where VRAM Goes During Inference

```
                    24 GB GPU VRAM
  ┌────────────────────────────────────────────┐
  │████████████████████████████████│░░░░░░│    │
  │     Model Weights (16 GB)      │KV (1G)│Free│
  │████████████████████████████████│░░░░░░│    │
  └────────────────────────────────────────────┘
  0 GB                           16 GB   17 GB  24 GB

  With 5 concurrent requests:
  ┌────────────────────────────────────────────┐
  │████████████████████████████████│░░░░░░░░░░░│
  │     Model Weights (16 GB)      │  KV (5 GB) │
  │████████████████████████████████│░░░░░░░░░░░│
  └────────────────────────────────────────────┘
  0 GB                           16 GB        21 GB
  ← Only 3 GB free! Adding more requests risks OOM
```

---

## What is PagedAttention?

### The Problem: KV Cache Memory Waste

Traditional KV cache allocates a **fixed, contiguous block** of memory for each request at the **maximum possible sequence length**, even if the actual sequence is much shorter:

```
Without PagedAttention (traditional allocation):

Request 1 (actual: 500 tokens, allocated: 8192 tokens)
┌──────────┬─────────────────────────────────────────────────┐
│ Used 500 │                WASTED (7692 tokens)             │
│  ~62 KB  │                   ~960 KB wasted                │
└──────────┴─────────────────────────────────────────────────┘

Request 2 (actual: 2000 tokens, allocated: 8192 tokens)
┌────────────────────┬───────────────────────────────────────┐
│   Used 2000        │           WASTED (6192 tokens)        │
│    ~250 KB         │              ~773 KB wasted            │
└────────────────────┴───────────────────────────────────────┘

Total waste: 60-80% of allocated KV cache memory! 😱
```

### The Solution: Paging (Like OS Virtual Memory)

**PagedAttention** (invented by vLLM) applies the same concept as operating system virtual memory pages to KV cache:

```
With PagedAttention:

Physical GPU Memory (KV Cache Pool):
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ B1  │ B2  │ B3  │ B4  │ B5  │ B6  │ B7  │FREE │FREE │FREE │
│Req1 │Req1 │Req2 │Req2 │Req2 │Req2 │Req1 │     │     │     │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

Page Table (maps logical → physical blocks):
  Request 1: [B1, B2, B7]        ← 3 blocks (non-contiguous!)
  Request 2: [B3, B4, B5, B6]    ← 4 blocks (non-contiguous!)

Key differences:
  ✅ Blocks allocated on-demand (not pre-allocated for max length)
  ✅ Blocks don't need to be contiguous in memory
  ✅ Freed immediately when request completes
  ✅ Memory waste drops from 60-80% → under 4%
```

### How PagedAttention Improves Your Pipeline

| Metric | Without PagedAttention | With PagedAttention (vLLM) |
|--------|----------------------|---------------------------|
| Memory waste | 60-80% | < 4% |
| Max concurrent requests (24 GB GPU) | 1-2 | 4-5 |
| Throughput (tokens/sec) | Baseline | **2-4× higher** |
| Memory fragmentation | High | Nearly zero |
| Dynamic batching | ❌ | ✅ Continuous batching |

### Continuous Batching (Bonus from vLLM)

Traditional inference waits for the longest request in a batch to finish before starting new ones. vLLM's **continuous batching** allows:

```
Traditional Batching:
  Batch 1: [Request A (100 tokens), Request B (2000 tokens)]
  → Wait for B to finish before starting C
  → GPU idle while A is "done" but B still running

Continuous Batching (vLLM):
  Time 0: Start [A, B]
  Time 1: A finishes → immediately start C in A's slot
  Time 2: C in progress, B still running → no idle GPU cycles
  → GPU stays ~100% utilized
```

---

## EC2 Instance Options

### GPU Instance Comparison

| Instance | GPU | VRAM | vCPUs | System RAM | On-Demand $/hr | Monthly (24/7) | Monthly (8hr/day) |
|----------|-----|------|-------|-----------|----------------|----------------|-------------------|
| **g5.xlarge** | 1× A10G | **24 GB** | 4 | 16 GB | $1.006 | $724 | $241 |
| **g5.2xlarge** | 1× A10G | **24 GB** | 8 | 32 GB | $1.212 | $873 | $291 |
| **g5.4xlarge** | 1× A10G | **24 GB** | 16 | 64 GB | $1.624 | $1,169 | $390 |
| **g6.xlarge** | 1× L4 | **24 GB** | 4 | 16 GB | $0.805 | $580 | $193 |
| **g6.2xlarge** | 1× L4 | **24 GB** | 8 | 32 GB | $0.978 | $704 | $235 |
| **g6e.xlarge** | 1× L40S | **48 GB** | 4 | 32 GB | $1.861 | $1,340 | $447 |
| **g6e.2xlarge** | 1× L40S | **48 GB** | 8 | 64 GB | $2.072 | $1,492 | $497 |

### Which GPU Fits Your Models?

```
Scenario 1: Both models on GPU (FP16)
──────────────────────────────────────
  VLM weights:          16.0 GB
  Embed weights:         2.0 GB
  KV cache (1 req):      1.0 GB
  Overhead:              0.5 GB
                        ─────────
  Total:                19.5 GB  → Fits 24 GB ✅ (4.5 GB free for more KV)
                                 → Max ~4 concurrent VLM requests


Scenario 2: VLM on GPU, Embed on CPU
──────────────────────────────────────
  VLM weights:          16.0 GB
  KV cache (1 req):      1.0 GB
  Overhead:              0.5 GB
                        ─────────
  Total:                17.5 GB  → Fits 24 GB ✅ (6.5 GB free!)
                                 → Max ~6 concurrent VLM requests
  Embed on CPU:          ~200ms per embedding (1B model, acceptable)


Scenario 3: VLM quantized INT4, both on GPU
──────────────────────────────────────
  VLM weights (INT4):    4.0 GB
  Embed weights (FP16):  2.0 GB
  KV cache (1 req):      1.0 GB
  Overhead:              0.5 GB
                        ─────────
  Total:                 7.5 GB  → Fits 24 GB ✅ (16.5 GB free!)
                                 → Max ~16 concurrent VLM requests
  Trade-off: ~2-5% accuracy loss from quantization


Scenario 4: 48 GB GPU (g6e — L40S)
──────────────────────────────────────
  VLM weights (FP16):   16.0 GB
  Embed weights (FP16):  2.0 GB
  KV cache (10 req):    10.0 GB
  Overhead:              1.0 GB
                        ─────────
  Total:                29.0 GB  → Fits 48 GB ✅ (19 GB free!)
                                 → Max ~19 concurrent VLM requests
                                 → True production-grade headroom
```

### Decision Matrix

| Priority | Recommended Instance | VRAM | Config | Monthly Cost | Concurrent Requests |
|----------|---------------------|------|--------|-------------|-------------------|
| **Lowest Cost** | g6.xlarge | 24 GB | VLM on GPU, Embed on CPU | **$580** | ~6 |
| **Best Balance** | g5.2xlarge | 24 GB | Both on GPU (FP16) | **$873** | ~4 |
| **Best Balance + Cheaper GPU** | g6.2xlarge | 24 GB | Both on GPU (FP16) | **$704** | ~4 |
| **Maximum Headroom** | g6e.xlarge | 48 GB | Both on GPU, 10+ concurrent | **$1,340** | ~15+ |
| **Budget (Quantized)** | g6.xlarge | 24 GB | VLM INT4 + Embed FP16 | **$580** | ~16 |

---

## vLLM Setup Guide

### What is vLLM?

vLLM is a high-throughput LLM inference engine that provides:
- **PagedAttention** — efficient KV cache management (< 4% waste)
- **Continuous batching** — no idle GPU cycles
- **OpenAI-compatible API** — drop-in replacement for your current NIM API calls
- **Quantization support** — AWQ, GPTQ, FP8 for smaller memory footprint
- **Multi-model serving** — run VLM + embedding on same GPU

### Installation on EC2

```bash
# 1. Launch EC2 with NVIDIA Deep Learning AMI (Ubuntu 22.04)
#    AMI ID: ami-0xxxx (search "Deep Learning AMI GPU" in EC2 console)

# 2. SSH into the instance
ssh -i your-key.pem ubuntu@<ec2-public-ip>

# 3. Install vLLM
pip install vllm

# 4. Verify GPU
nvidia-smi
# Should show: A10G or L4, 24 GB VRAM
```

### Launching the VLM Server

```bash
# Basic launch (nemotron-nano-vl-8b on full GPU)
python -m vllm.entrypoints.openai.api_server \
    --model nvidia/llama-3.1-nemotron-nano-vl-8b-v1 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --dtype float16
```

### Key vLLM Parameters to Understand

| Parameter | What It Does | Recommended Value | Why |
|-----------|-------------|-------------------|-----|
| `--gpu-memory-utilization` | Fraction of GPU VRAM vLLM can use | **0.70–0.85** | Leave room for embedding model or CPU overhead. 0.85 = 20.4 GB of 24 GB |
| `--max-model-len` | Maximum context window (tokens) | **8192** | Nemotron supports up to 128K, but longer = more KV cache VRAM |
| `--dtype` | Model precision | `float16` or `auto` | FP16 is the default; use `float8` for less VRAM |
| `--quantization` | Weight quantization method | `awq` or `gptq` | Reduces model weights from 16 GB → 4 GB (INT4) |
| `--max-num-seqs` | Max concurrent requests | **4-6** | Controls how many KV caches are active simultaneously |
| `--swap-space` | CPU RAM for KV cache overflow (GB) | **4** | Offloads inactive KV caches to CPU RAM instead of OOM |
| `--enforce-eager` | Disable CUDA graph optimization | Only if OOM | CUDA graphs use extra VRAM but speed up inference |
| `--enable-prefix-caching` | Cache common prompt prefixes | **Enable** | Reuses KV cache for repeated system prompts (saves ~200-500 MB) |
| `--tensor-parallel-size` | Split model across GPUs | **1** (single GPU) | Only useful for multi-GPU instances |

### How `--gpu-memory-utilization` Affects KV Cache

```
24 GB GPU with --gpu-memory-utilization values:

  0.90 → 21.6 GB usable
         - 16.0 GB model weights
         ─────────
         = 5.6 GB for KV cache → ~5 concurrent requests

  0.85 → 20.4 GB usable
         - 16.0 GB model weights
         ─────────
         = 4.4 GB for KV cache → ~4 concurrent requests
         ← RECOMMENDED (leaves 3.6 GB for embedding model)

  0.70 → 16.8 GB usable
         - 16.0 GB model weights
         ─────────
         = 0.8 GB for KV cache → ~0-1 concurrent requests
         ← TOO LOW for this model
```

### Important: `--max-model-len` Controls KV Cache Size

```
KV cache VRAM per request at different context lengths:

  --max-model-len 2048:   ~0.26 GB/request  → ~17 concurrent on 24 GB
  --max-model-len 4096:   ~0.53 GB/request  → ~8 concurrent on 24 GB
  --max-model-len 8192:   ~1.07 GB/request  → ~4 concurrent on 24 GB
  --max-model-len 16384:  ~2.13 GB/request  → ~2 concurrent on 24 GB
  --max-model-len 32768:  ~4.27 GB/request  → ~1 concurrent on 24 GB

For your RAG pipeline:
  - Image summarization prompts: ~500-1000 tokens → 2048 is plenty
  - Answer synthesis prompts: ~2000-4000 tokens → 4096 is enough
  
  RECOMMENDATION: --max-model-len 4096
    → Saves ~50% KV cache vs 8192
    → Doubles your concurrent request capacity
```

### Security: Environment Variables for vLLM

```bash
# Set in /etc/environment or systemd service file
export VLLM_API_KEY="your-secure-api-key"

# Launch with API key enforcement
python -m vllm.entrypoints.openai.api_server \
    --model nvidia/llama-3.1-nemotron-nano-vl-8b-v1 \
    --port 8000 \
    --api-key $VLLM_API_KEY \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85
```

---

## Multi-Model Serving Strategies

### Strategy 1: VLM on GPU + Embedding on CPU (Recommended)

```
┌──────────────────────────────────────────┐
│              EC2 g5.2xlarge               │
│                                          │
│  GPU (A10G, 24 GB):                      │
│  ┌────────────────────────────────────┐  │
│  │  vLLM Server (port 8000)          │  │
│  │  nemotron-nano-vl-8b              │  │
│  │  --gpu-memory-utilization 0.90    │  │
│  │  Max concurrent: ~5 requests      │  │
│  └────────────────────────────────────┘  │
│                                          │
│  CPU (8 vCPUs, 32 GB RAM):              │
│  ┌────────────────────────────────────┐  │
│  │  Embedding Server (port 8001)      │  │
│  │  nemotron-embed-1b (sentence-      │  │
│  │  transformers, CPU mode)           │  │
│  │  ~200ms per embedding              │  │
│  ├────────────────────────────────────┤  │
│  │  Reranker (MiniLM-L-6)            │  │
│  │  ~300ms for 20 pairs              │  │
│  ├────────────────────────────────────┤  │
│  │  Pipeline App (pipeline.py)       │  │
│  │  BM25 index (in-memory)           │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘

Pros: Maximum GPU VRAM for VLM, simple setup
Cons: Embedding slightly slower on CPU (~200ms vs ~50ms on GPU)
```

### Strategy 2: Both on GPU (Separate vLLM Instances)

```
┌──────────────────────────────────────────┐
│              EC2 g5.2xlarge               │
│                                          │
│  GPU (A10G, 24 GB — shared):             │
│  ┌────────────────────────────────────┐  │
│  │  vLLM Instance 1 (port 8000)      │  │
│  │  nemotron-nano-vl-8b              │  │
│  │  --gpu-memory-utilization 0.70    │  │
│  │  Uses: ~16.8 GB                   │  │
│  └────────────────────────────────────┘  │
│  ┌────────────────────────────────────┐  │
│  │  vLLM Instance 2 (port 8001)      │  │
│  │  nemotron-embed-1b                │  │
│  │  --gpu-memory-utilization 0.20    │  │
│  │  Uses: ~4.8 GB                    │  │
│  └────────────────────────────────────┘  │
│  [ 0.70 + 0.20 = 0.90 total ← OK ]     │
│                                          │
│  CPU: Pipeline App + Reranker + BM25    │
└──────────────────────────────────────────┘

Pros: Both models on GPU (fastest inference)
Cons: VLM gets less VRAM → fewer concurrent requests (~1-2)
      More complex setup (two vLLM processes)
```

### Strategy 3: Triton Inference Server (Production-Grade)

```
┌──────────────────────────────────────────┐
│              EC2 g5.2xlarge               │
│                                          │
│  GPU (A10G, 24 GB):                      │
│  ┌────────────────────────────────────┐  │
│  │  NVIDIA Triton Inference Server   │  │
│  │  (single process, manages both)   │  │
│  │                                    │  │
│  │  Model 1: nemotron-nano-vl-8b     │  │
│  │  Model 2: nemotron-embed-1b       │  │
│  │                                    │  │
│  │  • Automatic memory scheduling    │  │
│  │  • Priority queuing              │  │
│  │  • Model-level concurrency control│  │
│  └────────────────────────────────────┘  │
│                                          │
│  CPU: Pipeline App + Reranker + BM25    │
└──────────────────────────────────────────┘

Pros: Best GPU sharing, production-grade, auto-scaling per model
Cons: Most complex setup, steeper learning curve
```

### Strategy Comparison

| Strategy | Setup Complexity | GPU Efficiency | Latency | Best For |
|----------|-----------------|-----------------|---------|----------|
| VLM GPU + Embed CPU | ★ Easy | High (90% for VLM) | Embed: ~200ms | Your use case ✅ |
| Both GPU (2× vLLM) | ★★ Medium | Medium (split VRAM) | Both: ~50ms | Low-latency critical |
| Triton Server | ★★★ Hard | Best (smart scheduling) | Both: ~50ms | Enterprise scale |

---

## Cost Analysis

### Monthly Cost by Instance + Strategy

| Instance | On-Demand (24/7) | 8hr/day Weekdays | Reserved (1yr) | Spot (interruptible) |
|----------|-----------------|------------------|----------------|---------------------|
| **g6.xlarge** (L4, 24 GB) | $580 | **$193** | $366 | ~$174 |
| **g5.xlarge** (A10G, 24 GB) | $724 | **$241** | $456 | ~$217 |
| **g5.2xlarge** (A10G, 24 GB, 8 vCPU) | $873 | **$291** | $550 | ~$262 |
| **g6.2xlarge** (L4, 24 GB, 8 vCPU) | $704 | **$235** | $444 | ~$211 |
| **g6e.xlarge** (L40S, 48 GB) | $1,340 | **$447** | $845 | ~$402 |

### Total Pipeline Cost (Instance + RDS + S3 + Monitoring)

| Tier | GPU Instance | RDS | Other | **Total** |
|------|-------------|-----|-------|-----------|
| **Dev** (g6.xlarge, 8hr/day) | $193 | $47 (t4g.medium) | $8 | **~$248/mo** |
| **Production** (g5.2xlarge, 24/7) | $873 | $187 (r6g.large) | $20 | **~$1,080/mo** |
| **Production Optimized** (g6.2xlarge, Reserved) | $444 | $118 (Reserved) | $15 | **~$577/mo** |
| **High-Scale** (g6e.xlarge, 24/7) | $1,340 | $374 (r6g.xlarge) | $30 | **~$1,744/mo** |

---

## Recommendations

### For Your Use Case (Single PDF Pipeline, Low Concurrency)

**Go with: `g6.2xlarge` + VLM on GPU + Embedding on CPU**

```
Instance:      g6.2xlarge (1× L4, 24 GB VRAM, 8 vCPUs, 32 GB RAM)
VLM:           nemotron-nano-vl-8b (GPU, FP16, 90% VRAM)
Embedding:     nemotron-embed-1b (CPU, ~200ms latency — acceptable)
Reranker:      MiniLM-L-6 (CPU, ~300ms)
BM25:          In-memory (CPU)

Monthly cost:  $704 (on-demand) or $444 (reserved 1yr)
               $235 (8hr/day weekdays only)
```

**Why g6.2xlarge over g5.2xlarge?**
- Same 24 GB VRAM (L4 vs A10G)
- L4 has better power efficiency
- **~$169/month cheaper** ($704 vs $873)
- 8 vCPUs + 32 GB RAM = enough for embedding on CPU + reranker + pipeline

### Checklist Before Deploying vLLM

- [ ] Set `--max-model-len 4096` (your prompts don't need 8K+)
- [ ] Set `--gpu-memory-utilization 0.90` (if embedding on CPU)
- [ ] Enable `--enable-prefix-caching` (reuses system prompt KV cache)
- [ ] Set `--swap-space 4` (offload idle KV to CPU RAM)
- [ ] Set `--max-num-seqs 4` (limit concurrent to prevent OOM)
- [ ] Use `--api-key` for security
- [ ] Monitor GPU utilization via `nvidia-smi` and CloudWatch
- [ ] Set up auto-restart via systemd service

### Pipeline Code Change (Only 1 Line)

```python
# In pipeline.py and query.py, change:
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# To your EC2 instance:
NVIDIA_BASE_URL = "http://<ec2-private-ip>:8000/v1"

# Everything else works as-is (vLLM serves OpenAI-compatible API)
```

---

## Quick Reference: VRAM Budget Cheat Sheet

```
╔══════════════════════════════════════════════════════════════╗
║  VRAM BUDGET FORMULA                                         ║
║                                                              ║
║  Total VRAM needed =                                         ║
║    Model weights (params × bytes_per_param)                  ║
║  + KV cache (concurrent_reqs × context_len × ~130 KB/token) ║
║  + Overhead (~0.5-1.0 GB)                                    ║
║                                                              ║
║  FP16:  8B model = 16 GB weights                             ║
║  INT8:  8B model = 8 GB weights                              ║
║  INT4:  8B model = 4 GB weights                              ║
║                                                              ║
║  KV cache per request:                                       ║
║    2K context = ~0.26 GB                                     ║
║    4K context = ~0.53 GB                                     ║
║    8K context = ~1.07 GB                                     ║
║                                                              ║
║  Rule of thumb:                                              ║
║    Available KV VRAM = Total VRAM - Model Weights - 1 GB     ║
║    Max concurrent = Available KV VRAM ÷ KV per request       ║
╚══════════════════════════════════════════════════════════════╝
```
