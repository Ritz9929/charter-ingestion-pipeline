# AWS Self-Hosting Architecture — Charter Ingestion Pipeline

Complete guide to deploying and self-hosting the Multimodal RAG pipeline on AWS, including architecture, infrastructure choices, justifications, and detailed cost analysis.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Infrastructure Components](#infrastructure-components)
3. [Deployment Architecture](#deployment-architecture)
4. [Component Deep Dives](#component-deep-dives)
5. [Networking & Security](#networking--security)
6. [Cost Analysis](#cost-analysis)
7. [Deployment Tiers](#deployment-tiers)
8. [Monitoring & Operations](#monitoring--operations)

---

## Architecture Overview

### Current vs AWS Self-Hosted

| Aspect | Current (Local + NIM) | AWS Self-Hosted |
|--------|----------------------|-----------------|
| VLM Inference | NVIDIA NIM Cloud (free tier, 40 RPM) | Self-hosted on EC2 GPU instance |
| Embeddings | NVIDIA NIM Cloud (free tier) | Self-hosted on EC2 GPU instance |
| Vector Database | Docker on local machine | Amazon RDS PostgreSQL + pgvector |
| Reranker | Local CPU | EC2 or Lambda |
| File Storage | Local filesystem (`mock_s3_storage/`) | Amazon S3 |
| Rate Limits | 40 RPM, ~1000 credits/key | **None** (you own the GPU) |
| Availability | Limited by your machine uptime | 99.9%+ with proper setup |
| Monthly Cost | $0 (free tier) | $1,200 – $11,000+ |

### High-Level Architecture Diagram

```
                                    AWS Cloud (us-east-1)
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                      │
│   ┌──────────────────────────────────────────────────────────────────────────────┐   │
│   │                          VPC (10.0.0.0/16)                                   │   │
│   │                                                                              │   │
│   │   ┌──────────────────────────┐    ┌──────────────────────────────────────┐   │   │
│   │   │   Public Subnet          │    │   Private Subnet                     │   │   │
│   │   │                          │    │                                      │   │   │
│   │   │  ┌────────────────────┐  │    │  ┌──────────────────────────────┐   │   │   │
│   │   │  │  ALB / API Gateway │  │    │  │  EC2 GPU Instance            │   │   │   │
│   │   │  │  (HTTPS endpoint)  │  │    │  │  (g5.xlarge / g5.2xlarge)    │   │   │   │
│   │   │  └────────┬───────────┘  │    │  │                              │   │   │   │
│   │   │           │              │    │  │  ┌─────────────────────────┐ │   │   │   │
│   │   │           │              │    │  │  │  VLM Model Server      │ │   │   │   │
│   │   │           └──────────────┼────┼──┤  │  (vLLM / TGI)          │ │   │   │   │
│   │   │                          │    │  │  │  nemotron-nano-vl 8B   │ │   │   │   │
│   │   │                          │    │  │  └─────────────────────────┘ │   │   │   │
│   │   │                          │    │  │                              │   │   │   │
│   │   │                          │    │  │  ┌─────────────────────────┐ │   │   │   │
│   │   │                          │    │  │  │  Embedding Server      │ │   │   │   │
│   │   │                          │    │  │  │  nemotron-embed-1b     │ │   │   │   │
│   │   │                          │    │  │  └─────────────────────────┘ │   │   │   │
│   │   │                          │    │  └──────────────────────────────┘   │   │   │
│   │   │                          │    │                                      │   │   │
│   │   │                          │    │  ┌──────────────────────────────┐   │   │   │
│   │   │                          │    │  │  EC2 / ECS (CPU)             │   │   │   │
│   │   │                          │    │  │  Pipeline Application        │   │   │   │
│   │   │                          │    │  │  • pipeline.py               │   │   │   │
│   │   │                          │    │  │  • query.py                  │   │   │   │
│   │   │                          │    │  │  • Cross-Encoder Reranker    │   │   │   │
│   │   │                          │    │  └──────────────────────────────┘   │   │   │
│   │   │                          │    │                                      │   │   │
│   │   │                          │    │  ┌──────────────────────────────┐   │   │   │
│   │   │                          │    │  │  RDS PostgreSQL              │   │   │   │
│   │   │                          │    │  │  (pgvector extension)        │   │   │   │
│   │   │                          │    │  │  db.r6g.large / db.r7g.large │   │   │   │
│   │   │                          │    │  └──────────────────────────────┘   │   │   │
│   │   │                          │    │                                      │   │   │
│   │   └──────────────────────────┘    └──────────────────────────────────────┘   │   │
│   │                                                                              │   │
│   └──────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
│   ┌───────────────┐    ┌───────────────┐    ┌────────────────────┐                   │
│   │  Amazon S3     │    │  CloudWatch    │    │  Secrets Manager   │                   │
│   │  (PDF + Image  │    │  (Logs +       │    │  (API Keys,        │                   │
│   │   Storage)     │    │   Metrics)     │    │   DB Credentials)  │                   │
│   └───────────────┘    └───────────────┘    └────────────────────┘                   │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Infrastructure Components

### 1. EC2 GPU Instance — VLM + Embedding Inference

**What it does:** Runs the VLM model (image summarization) and embedding model.

**Why we need it:** NVIDIA NIM's free tier limits you to 40 RPM and ~1000 credits. Self-hosting removes all rate limits — you can process thousands of images per hour.

#### Instance Options

| Instance | GPU | VRAM | vCPUs | RAM | On-Demand $/hr | Best For |
|----------|-----|------|-------|-----|----------------|----------|
| **g5.xlarge** | 1× A10G | 24 GB | 4 | 16 GB | $1.006 | nemotron-nano-vl (8B) — fits in 24GB |
| **g5.2xlarge** | 1× A10G | 24 GB | 8 | 32 GB | $1.212 | Same GPU, more CPU/RAM for pipeline |
| **g5.4xlarge** | 1× A10G | 24 GB | 16 | 64 GB | $1.624 | Heavy concurrent ingestion |
| **g6.xlarge** | 1× L4 | 24 GB | 4 | 16 GB | $0.805 | Newer GPU, slightly cheaper |
| **g6.2xlarge** | 1× L4 | 24 GB | 8 | 32 GB | $0.978 | Balanced price/performance |
| **g5.12xlarge** | 4× A10G | 96 GB | 48 | 192 GB | $5.672 | qwen3.5-122b (current model) |
| **p4d.24xlarge** | 8× A100 | 320 GB | 96 | 1.1 TB | $32.77 | qwen3.5-397b (largest model) |

> **Recommended:** `g5.xlarge` ($1.006/hr) for `nemotron-nano-vl` (8B model). This is the **sweet spot** — cheapest GPU instance that fits the model.

**Why A10G (g5) over L4 (g6)?**
- A10G has better FP16 performance (31.2 TFLOPS vs 30.3 TFLOPS)
- More mature, better driver/framework support
- G6 is slightly cheaper but newer — less community tooling
- Both have 24GB VRAM — sufficient for 8B models

**Why NOT a larger instance for qwen3.5-122b?**
- `qwen3.5-122b` requires ~240GB VRAM → needs `g5.12xlarge` (4× A10G = 96GB) with model parallelism
- At $5.67/hr = **$4,083/month** — 5× more expensive than nemotron-nano on g5.xlarge
- Quality improvement is ~10-15% — rarely worth the 5× cost increase

#### Model Serving Software

| Software | Description | Why Use It |
|----------|-------------|-----------|
| **vLLM** | High-throughput LLM inference engine | PagedAttention for ~2-4× throughput, continuous batching |
| **NVIDIA TGI (Text Generation Inference)** | HuggingFace's inference server | Easy setup, good for quick deployments |
| **NVIDIA Triton Inference Server** | Production-grade multi-model serving | Serves VLM + embedding model on same GPU, model ensemble |

> **Recommended:** **vLLM** — gives the best throughput (supports continuous batching, PagedAttention), supports vision models, and has native OpenAI-compatible API (no code changes needed in `pipeline.py`).

```bash
# Example: Launch vLLM on g5.xlarge serving nemotron-nano-vl
pip install vllm

python -m vllm.entrypoints.openai.api_server \
    --model nvidia/llama-nemotron-nano-vl \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9
```

Your `pipeline.py` only needs one change:
```python
# Change this:
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# To this:
NVIDIA_BASE_URL = "http://<ec2-private-ip>:8000/v1"
```

---

### 2. Amazon RDS PostgreSQL — Vector Database

**What it does:** Stores document chunks and their embedding vectors with pgvector extension.

**Why we need it:** Your current Docker-based PostgreSQL is not production-ready — no backups, no high availability, no monitoring. RDS provides all of this out of the box.

**Why RDS over self-managed PostgreSQL on EC2?**

| Aspect | RDS (Managed) | EC2 (Self-Managed) |
|--------|--------------|-------------------|
| Automatic backups | ✅ Daily + point-in-time | ❌ Manual setup |
| High availability | ✅ Multi-AZ failover | ❌ Build yourself |
| Patching & updates | ✅ Automatic | ❌ Manual |
| Monitoring | ✅ CloudWatch integrated | ❌ Self-configure |
| Scaling | ✅ Push-button resize | ❌ Manual migration |
| pgvector support | ✅ Native since PG 15 | ✅ Manual install |
| DBA effort needed | Low | High |

#### Instance Options

| Instance | vCPUs | RAM | Storage | On-Demand $/hr | Monthly | Best For |
|----------|-------|-----|---------|----------------|---------|----------|
| **db.t4g.medium** | 2 | 4 GB | gp3 20GB | $0.065 | ~$47 | Dev/testing |
| **db.r6g.large** | 2 | 16 GB | gp3 100GB | $0.260 | ~$187 | Small production (< 1M vectors) |
| **db.r6g.xlarge** | 4 | 32 GB | gp3 500GB | $0.520 | ~$374 | Medium production (1-10M vectors) |
| **db.r7g.2xlarge** | 8 | 64 GB | gp3 1TB | $1.068 | ~$769 | Large production (10M+ vectors) |

> **Recommended:** `db.r6g.large` ($187/month) — 16GB RAM is ideal for pgvector HNSW indexes on up to ~1M vectors. Memory-optimized (r-series) is critical because pgvector indexes are memory-resident.

**Why Memory-Optimized (r-series)?**
- pgvector HNSW indexes live in RAM for fast search
- Each 1024-dim float32 vector = ~4KB
- 1M vectors ≈ 4GB of index data → needs at least 16GB RAM instance
- General-purpose (m-series) would work but with slower search at scale

**Storage calculation:**
```
Per vector: 1024 dims × 4 bytes = 4 KB
Per chunk:  4 KB (vector) + ~1 KB (text + metadata) = ~5 KB
550 chunks: 550 × 5 KB = ~2.75 MB (your current PDF)
1M chunks:  1M × 5 KB = ~5 GB
HNSW index: ~2× data size = ~10 GB for 1M chunks
Total for 1M chunks: ~15 GB storage + 10 GB RAM for index
```

---

### 3. Amazon S3 — File Storage

**What it does:** Stores uploaded PDFs, extracted images, and summary cache files.

**Why we need it:** Replaces your local `mock_s3_storage/` folder. S3 provides:
- **Durability**: 99.999999999% (11 9's) — your files never get lost
- **Scalability**: Store unlimited files
- **Access control**: Fine-grained permissions
- **Cross-service access**: GPU instances and application servers can all access the same files

**Why NOT EBS (Elastic Block Storage)?**
- EBS is attached to one EC2 instance — can't share between VLM server and application server
- S3 is accessible from anywhere in AWS
- S3 is cheaper for infrequently accessed data

#### Cost

| Tier | Storage $/GB/month | Requests (PUT) | Requests (GET) |
|------|-------------------|----------------|----------------|
| S3 Standard | $0.023 | $0.005 per 1K | $0.0004 per 1K |
| S3 Infrequent Access | $0.0125 | $0.01 per 1K | $0.001 per 1K |

**Your estimated usage:**
- 1 PDF × 300 images × ~200KB each = ~60 MB per PDF
- 100 PDFs/month = ~6 GB → **$0.14/month** (negligible)

---

### 4. EC2 / ECS — Application Server

**What it does:** Runs `pipeline.py` (ingestion) and `query.py` (retrieval + reranking), including the cross-encoder reranker model.

**Why separate from GPU instance?**
- The GPU instance should be dedicated to model inference (VLM + embeddings)
- The application logic (PDF parsing, chunking, reranking, orchestration) is CPU-bound
- Separating them allows independent scaling — e.g., stop GPU instance when not ingesting

#### Instance Options

| Instance | vCPUs | RAM | On-Demand $/hr | Monthly | Best For |
|----------|-------|-----|----------------|---------|----------|
| **t3.medium** | 2 | 4 GB | $0.0416 | ~$30 | Dev/testing |
| **c6g.xlarge** | 4 | 8 GB | $0.136 | ~$98 | Production with reranker |
| **c6g.2xlarge** | 8 | 16 GB | $0.272 | ~$196 | High-throughput ingestion |

> **Recommended:** `c6g.xlarge` ($98/month) — ARM-based (Graviton3), 20% cheaper than x86 equivalents. 4 vCPUs handle the cross-encoder reranker + PyMuPDF extraction + chunking comfortably.

**ECS Fargate Alternative:**
- If you prefer containers over EC2, use ECS Fargate
- Pricing: ~$0.04/vCPU-hour + $0.004/GB-hour
- 4 vCPU + 8GB = ~$0.19/hr = ~$137/month
- Advantage: No EC2 management, auto-scaling, pay-per-use
- Disadvantage: Slightly more expensive, cold start latency

---

### 5. AWS Secrets Manager

**What it does:** Securely stores API keys, database credentials, and configuration.

**Why we need it:** Replaces your `.env` file. In production, you should never store secrets in files on disk.

**Cost:** $0.40 per secret/month + $0.05 per 10K API calls ≈ **~$2/month**

---

### 6. Amazon CloudWatch

**What it does:** Collects logs, metrics, and alerts from all components.

**Why we need it:**
- Monitor GPU utilization (are you paying for idle GPU time?)
- Track inference latency and error rates
- Alert on database connection issues or storage growth
- Log all pipeline runs for debugging

**Cost:** ~$5-15/month for basic monitoring

---

## Deployment Architecture

### Option A: Minimal (Dev/Staging)

```
┌─────────────────────┐     ┌──────────────────────┐
│  g5.xlarge           │     │  RDS db.t4g.medium   │
│  • vLLM (VLM)       │     │  • PostgreSQL 16     │
│  • Embedding Server  │◄───►│  • pgvector          │
│  • Pipeline App     │     │  • 20GB gp3           │
│  • Reranker         │     │                        │
│  (everything on 1)  │     │  $47/mo               │
│  $724/mo            │     └──────────────────────┘
└─────────────────────┘
              Total: ~$771/month
```

**Pros:** Simplest setup, lowest cost  
**Cons:** No separation of concerns, GPU idle during queries

---

### Option B: Recommended (Production)

```
┌─────────────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│  g5.xlarge (GPU)     │     │  c6g.xlarge (CPU)     │     │  RDS r6g.large   │
│  • vLLM (VLM)       │     │  • Pipeline App       │     │  • PostgreSQL 16 │
│  • Embedding Server  │◄───►│  • query.py           │◄───►│  • pgvector      │
│                      │     │  • Cross-Encoder      │     │  • HNSW index    │
│  $724/mo             │     │  • PDF Processing     │     │  • 100GB gp3     │
│  (can stop when idle)│     │                        │     │  • Multi-AZ      │
└─────────────────────┘     │  $98/mo                │     │  $374/mo         │
                             └──────────────────────┘     └──────────────────┘
         │
         ▼
┌─────────────────────┐     ┌──────────────────────┐
│  Amazon S3           │     │  CloudWatch +         │
│  • PDFs             │     │  Secrets Manager      │
│  • Extracted Images │     │  • Logs & Metrics     │
│  • Summary Cache    │     │  • API Keys           │
│  ~$1/mo             │     │  ~$7/mo               │
└─────────────────────┘     └──────────────────────┘

              Total: ~$1,204/month (GPU always on)
              Total: ~$600/month (GPU on 8hr/day — stop nights/weekends)
```

**Key optimization:** The GPU instance can be **stopped when not ingesting**. If you only process new PDFs during business hours, you can schedule the GPU instance to run 8 hours/day, 5 days/week — cutting GPU costs by **~75%**.

---

### Option C: High-Scale (Enterprise)

```
┌────────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  g5.2xlarge (GPU)       │     │  ECS Fargate Cluster  │     │  RDS r6g.xlarge      │
│  • vLLM (VLM, batched) │     │  • Auto-scaling       │     │  • Multi-AZ HA       │
│  • Embedding Server    │◄───►│  • Pipeline Workers   │◄───►│  • Read Replicas     │
│  • GPU auto-scaling     │     │  • Query API          │     │  • HNSW index        │
│                          │     │  • Reranker           │     │  • 500GB gp3          │
│  $873/mo                 │     │  ~$200/mo             │     │  $748/mo              │
└────────────────────────┘     └──────────────────────┘     └──────────────────────┘
         │                               │
         ▼                               ▼
┌─────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  Amazon S3           │     │  ElastiCache Redis    │     │  CloudWatch +         │
│  • PDFs              │     │  • Query caching     │     │  Secrets Manager      │
│  ~$1/mo              │     │  • Session store     │     │  ~$20/mo              │
│                      │     │  ~$50/mo             │     │                        │
└─────────────────────┘     └──────────────────────┘     └──────────────────────┘

              Total: ~$1,892/month
```

---

## Cost Analysis

### Detailed Monthly Breakdown

| Component | Dev/Staging | Production | Enterprise |
|-----------|------------|------------|------------|
| **GPU Instance** (VLM + Embeddings) | g5.xlarge: $724 | g5.xlarge: $724 | g5.2xlarge: $873 |
| **App Server** (Pipeline + Reranker) | (on GPU instance) | c6g.xlarge: $98 | ECS Fargate: $200 |
| **Database** (RDS PostgreSQL) | db.t4g.medium: $47 | db.r6g.large: $187 | db.r6g.xlarge: $374 |
|   └── Multi-AZ HA | — | +$187 | +$374 |
|   └── Storage (gp3) | 20GB: $2.30 | 100GB: $11.50 | 500GB: $57.50 |
| **S3** (File Storage) | ~$1 | ~$1 | ~$5 |
| **CloudWatch + Logging** | $5 | $10 | $20 |
| **Secrets Manager** | $2 | $2 | $2 |
| **ElastiCache Redis** | — | — | $50 |
| **Data Transfer** | ~$5 | ~$10 | ~$30 |
| | | | |
| **Total (GPU always on)** | **~$786/mo** | **~$1,230/mo** | **~$1,986/mo** |
| **Total (GPU 8hr/day)** | **~$545/mo** | **~$870/mo** | **~$1,620/mo** |

### Cost Optimization Strategies

| Strategy | Savings | How |
|----------|---------|-----|
| **Spot Instances (GPU)** | Up to 70% | Use for batch ingestion (interruptible) |
| **Reserved Instances (1yr)** | ~37% | Commit to 1-year for steady workloads |
| **Savings Plans (3yr)** | ~50-60% | Long-term commitment for maximum savings |
| **GPU scheduling** | ~65-75% | Stop GPU nights/weekends (only for ingestion) |
| **Right-sizing** | 20-40% | Monitor utilization, downgrade if underused |

**Example with optimizations (Production tier):**
```
GPU: g5.xlarge Reserved Instance (1yr)       $456/mo  (was $724)
GPU: Run only 8hr/day weekdays               $152/mo  (was $456)
App: c6g.xlarge Reserved Instance (1yr)       $62/mo   (was $98)
RDS: db.r6g.large Reserved Instance (1yr)     $118/mo  (was $187)
Others:                                        $24/mo

Optimized Total:                              ~$356/month ✅
```

### Cost Comparison: Self-Hosted vs NVIDIA NIM vs Full Cloud

| Approach | Monthly Cost | Rate Limit | Best For |
|----------|-------------|------------|----------|
| **NVIDIA NIM Free Tier** (current) | $0 | 40 RPM, ~1000 credits | Prototyping, low volume |
| **NVIDIA NIM Enterprise** | ~$1,000-4,000 | Higher limits | Medium volume, no infra management |
| **AWS Self-Hosted (optimized)** | ~$356-870 | **Unlimited** | Production, high volume, full control |
| **AWS Self-Hosted (always on)** | ~$786-1,986 | **Unlimited** | 24/7 operations |

---

## Networking & Security

### VPC Design

```
VPC: 10.0.0.0/16

  Public Subnets (10.0.1.0/24, 10.0.2.0/24):
    • ALB / API Gateway (HTTPS ingress)
    • NAT Gateway (outbound internet for updates)

  Private Subnets (10.0.10.0/24, 10.0.20.0/24):
    • GPU Instance (VLM + Embeddings)
    • App Server (Pipeline + Reranker)
    • RDS PostgreSQL (Multi-AZ)
```

### Security Groups

| Component | Inbound | Outbound |
|-----------|---------|----------|
| ALB | Port 443 (HTTPS) from 0.0.0.0/0 | Port 8080 to App Server SG |
| App Server | Port 8080 from ALB SG | Port 8000 to GPU SG, Port 5432 to RDS SG |
| GPU Instance | Port 8000 from App Server SG | Port 443 to S3 (VPC endpoint) |
| RDS | Port 5432 from App Server SG | — |

### Key Security Practices

1. **No public IPs** on GPU or app servers — all traffic through ALB
2. **IAM roles** instead of access keys for S3 access
3. **Secrets Manager** for database credentials (auto-rotation)
4. **VPC Endpoints** for S3 access (no internet transit)
5. **Encryption**: RDS encryption at rest + SSL in transit

---

## Monitoring & Operations

### Key Metrics to Monitor

| Metric | Source | Alert Threshold | Why |
|--------|--------|----------------|-----|
| GPU Utilization | CloudWatch (NVIDIA DCGM) | < 20% for 1hr → downsize | Avoid paying for idle GPU |
| GPU Memory Used | CloudWatch | > 90% | Model may OOM |
| Inference Latency (P95) | Application logs | > 15s | User experience degradation |
| RDS CPU Utilization | CloudWatch | > 80% for 15min | Need to scale up |
| RDS Free Storage | CloudWatch | < 20% | Risk of running out |
| HNSW Search Latency | Application logs | > 500ms | Need to optimize index |
| S3 Storage Size | CloudWatch | > 1TB | Cost review needed |
| Error Rate (5xx) | ALB metrics | > 1% | Something is broken |

### Operational Runbook

| Task | Frequency | How |
|------|-----------|-----|
| Update model weights | Monthly/Quarterly | Download new model, restart vLLM |
| RDS backup verification | Weekly | Restore test to separate instance |
| Cost review | Monthly | AWS Cost Explorer |
| Security patching | Monthly | SSM Patch Manager |
| Scale review | Quarterly | Check GPU/DB utilization trends |

---

## Migration Steps (From Current to AWS)

### Phase 1: Database Migration (Week 1)

```bash
# 1. Create RDS PostgreSQL instance with pgvector
# 2. Create HNSW index
# 3. Update PG_CONNECTION_STRING in Secrets Manager
# 4. Run pipeline to populate new database
```

### Phase 2: Model Deployment (Week 2)

```bash
# 1. Launch g5.xlarge with Deep Learning AMI
# 2. Install vLLM and download models
# 3. Start inference server
# 4. Update NVIDIA_BASE_URL to point to EC2 private IP
# 5. Test ingestion pipeline end-to-end
```

### Phase 3: Application Deployment (Week 3)

```bash
# 1. Containerize pipeline.py and query.py (Docker)
# 2. Deploy to ECS or EC2 app server
# 3. Set up ALB for HTTPS access
# 4. Configure CloudWatch logging
# 5. Set up GPU scheduling (stop nights/weekends)
```

### Phase 4: Production Hardening (Week 4)

```bash
# 1. Enable RDS Multi-AZ for high availability
# 2. Set up CloudWatch alarms
# 3. Configure auto-scaling policies
# 4. Security audit (VPC, SGs, IAM)
# 5. Load testing and performance tuning
```

---

## Appendix: GPU Instance Quick Reference

| If you want to run... | Minimum Instance | VRAM Needed | Monthly Cost |
|-----------------------|-----------------|------------|-------------|
| nemotron-nano-vl (8B) | g5.xlarge | 16 GB | $724 |
| llama-3.2-11b-vision (11B) | g5.xlarge | 22 GB | $724 |
| qwen3.5-122b-a10b (122B MoE) | g5.12xlarge (4×A10G) | 96 GB | $4,083 |
| qwen3.5-397b-a17b (397B MoE) | p4d.24xlarge (8×A100) | 320 GB | $23,597 |
| nemotron-embed-1b (1B) | g5.xlarge (shared) | 2 GB | (shared with VLM) |
| llama-embed-nemotron-8b (8B) | g5.xlarge (shared) | 16 GB | (shared with VLM) |

> **Key takeaway:** Switching from `qwen3.5-122b` ($4,083/mo) to `nemotron-nano-vl` ($724/mo) saves **$3,359/month** while retaining ~90% quality for document understanding tasks.
