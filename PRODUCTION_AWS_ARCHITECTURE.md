# Production-Grade AWS Architecture — Multimodal RAG Pipeline

Complete production architecture for deploying the Charter Ingestion Pipeline on AWS with high availability, auto-scaling, security, and multi-user support.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Full Architecture Diagram](#full-architecture-diagram)
3. [Component Deep Dive](#component-deep-dive)
4. [Request Flow](#request-flow)
5. [High Availability & Fault Tolerance](#high-availability--fault-tolerance)
6. [Auto-Scaling Strategy](#auto-scaling-strategy)
7. [Networking & Security](#networking--security)
8. [CI/CD Pipeline](#cicd-pipeline)
9. [Monitoring & Alerting](#monitoring--alerting)
10. [Cost Breakdown](#cost-breakdown)
11. [Deployment Checklist](#deployment-checklist)

---

## Architecture Overview

### Prototype vs Production

| Aspect | Your Prototype (Current) | Production (Target) |
|--------|-------------------------|---------------------|
| Users | 1 (you, `query.py` CLI) | 50-500+ concurrent via REST API |
| VLM (Image Summarization) | NVIDIA NIM free tier (40 RPM) | Self-hosted vLLM on EC2 (on-demand, ingestion only) |
| Answer Synthesis | NVIDIA NIM free tier | **External LLM API** (OpenAI / NIM / Azure) |
| Embedding | NVIDIA NIM API | Self-hosted on CPU (zero cost) |
| Reranker | Local CPU (MiniLM-L-6) | Local CPU (no KV cache needed) |
| Search | hybrid (BM25 + semantic) | Same, behind load balancer |
| Database | Docker container on laptop | RDS Multi-AZ (99.95% SLA) |
| File Storage | Local `mock_s3_storage/` | Amazon S3 (11 nines durability) |
| Availability | Your laptop uptime | 99.9%+ (multi-AZ, auto-restart) |
| Scaling | None | Auto-scaling app instances + on-demand GPU |
| Security | `.env` file | Secrets Manager + VPC + IAM |
| Monitoring | Print statements | CloudWatch + Grafana dashboards |
| CI/CD | Manual `git push` | Automated deploy on merge |

---

## Full Architecture Diagram

```
                                  INTERNET
                                     │
                         ┌───────────┴───────────┐
                         ▼                       ▼
            ┌────────────────────────┐  ┌─────────────────────────┐
            │     Amazon Route 53     │  │  External LLM API       │
            │   (DNS + Health Checks)  │  │  (OpenAI / NIM / Azure) │
            │   api.yourcompany.com    │  │  Answer Synthesis       │
            └────────────┬───────────┘  │  No KV cache concern    │
                         │              │  Provider manages all   │
                         ▼              └─────────────────────────┘
            ┌────────────────────────┐           ▲
            │    AWS WAF (Firewall)    │           │ API calls from
            │  • Rate limiting         │           │ Query Containers
            │  • SQL injection block   │           │
            │  • Bot detection          │           │
            └────────────┬───────────┘           │
                         │                       │
                         ▼                       │
╔════════════════════════════════════════════════════════════════════════════════╗
║                            VPC (10.0.0.0/16)                                  ║
║                                                                                ║
║   ┌─────────────────────────────────────────────────────────────────────────┐ ║
║   │                     PUBLIC SUBNETS (2 AZs)                               │ ║
║   │                                                                          │ ║
║   │   ┌──────────────────────────────────────────────────────────────────┐  │ ║
║   │   │              Application Load Balancer (ALB)                     │  │ ║
║   │   │   • HTTPS (443) termination with ACM certificate                │  │ ║
║   │   │   • Path-based routing:                                          │  │ ║
║   │   │     /api/query    → Query Target Group                          │  │ ║
║   │   │     /api/ingest   → Ingestion Target Group                      │  │ ║
║   │   │     /health       → Health Check endpoint                       │  │ ║
║   │   └────────────┬─────────────────────────────┬──────────────────────┘  │ ║
║   │                │                             │                          │ ║
║   │   ┌────────────┴──────┐       ┌──────────────┴──────┐                  │ ║
║   │   │   NAT Gateway     │       │   NAT Gateway       │                  │ ║
║   │   │   (AZ-1)          │       │   (AZ-2)            │                  │ ║
║   │   └───────────────────┘       └─────────────────────┘                  │ ║
║   └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                ║
║   ┌─────────────────────────────────────────────────────────────────────────┐ ║
║   │                    PRIVATE SUBNETS (2 AZs)                               │ ║
║   │                                                                          │ ║
║   │   ┌──────────────────────────────┐  ┌──────────────────────────────┐    │ ║
║   │   │  QUERY SERVICE (ECS)         │  │  GPU TIER (INGESTION ONLY)   │    │ ║
║   │   │  (Auto-scaling, always-on)   │  │  (On-demand — NOT 24/7)      │    │ ║
║   │   │                              │  │                              │    │ ║
║   │   │  ┌────────────────────────┐ │  │  ┌────────────────────────┐ │    │ ║
║   │   │  │ Query Container 1      │ │  │  │  g6.2xlarge             │ │    │ ║
║   │   │  │ • FastAPI server       │ │  │  │  1× L4 GPU (24 GB)     │ │    │ ║
║   │   │  │ • Hybrid search        │ │  │  │                        │ │    │ ║
║   │   │  │   (BM25 + semantic)    │ │  │  │  vLLM (port 8000)      │ │    │ ║
║   │   │  │ • Embed (CPU, 1B)      │ │  │  │  nemotron-nano-vl-8b   │ │    │ ║
║   │   │  │ • Reranker (CPU, 22M)  │ │  │  │  Image summaries only  │ │    │ ║
║   │   │  │ • Answer → External ───┼─┼──┼──┼──► External LLM API    │ │    │ ║
║   │   │  │   LLM API (no GPU)     │ │  │  │                        │ │    │ ║
║   │   │  └────────────────────────┘ │  │  │  ON only during         │ │    │ ║
║   │   │  ┌────────────────────────┐ │  │  │  ingestion (~3 hrs/wk) │ │    │ ║
║   │   │  │ Query Container 2      │ │  │  └────────────────────────┘ │    │ ║
║   │   │  │ (auto-scaled)          │ │  │                              │    │ ║
║   │   │  └────────────────────────┘ │  │  No standby GPU needed —    │    │ ║
║   │   │  ┌────────────────────────┐ │  │  queries don't use GPU!     │    │ ║
║   │   │  │ Query Container N      │ │  │                              │    │ ║
║   │   │  │ (auto-scaled)          │ │  │                              │    │ ║
║   │   │  └────────────────────────┘ │  │                              │    │ ║
║   │   └──────────────────────────────┘  └──────────────────────────────┘    │ ║
║   │                                                                          │ ║
║   │   ┌──────────────────────────────┐  ┌──────────────────────────────┐    │ ║
║   │   │  INGESTION SERVICE           │  │  DATABASE TIER               │    │ ║
║   │   │  (ECS — on-demand)           │  │                              │    │ ║
║   │   │                              │  │  ┌────────────────────────┐ │    │ ║
║   │   │  ┌────────────────────────┐ │  │  │  RDS PostgreSQL (Pri)  │ │    │ ║
║   │   │  │ Ingest Worker          │ │  │  │  db.r6g.xlarge         │ │    │ ║
║   │   │  │ • PDFExtractor        │ │  │  │  • pgvector + HNSW     │ │    │ ║
║   │   │  │ • ImageSummarizer ────┼─┼──┤  │  • 32 GB RAM           │ │    │ ║
║   │   │  │   (→ vLLM on GPU)     │ │  │  │  • Multi-AZ standby    │ │    │ ║
║   │   │  │ • SmartChunker        │ │  │  │  • Automated backups   │ │    │ ║
║   │   │  │ • VectorStore         │ │  │  └──────────┬─────────────┘ │    │ ║
║   │   │  └────────────────────────┘ │  │             │               │    │ ║
║   │   └──────────────────────────────┘  │  ┌──────────▼─────────────┐ │    │ ║
║   │                                      │  │  RDS PostgreSQL (Sby) │ │    │ ║
║   │                                      │  │  (Auto-failover AZ-2) │ │    │ ║
║   │                                      │  └──────────────────────┘ │    │ ║
║   │                                      └──────────────────────────────┘    │ ║
║   └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                ║
║   ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  ┌───────────────┐  ║
║   │  Amazon S3    │  │ ElastiCache  │  │ Secrets Manager│  │  CloudWatch    │  ║
║   │  • PDFs      │  │ Redis        │  │ • LLM API Keys │  │  • Metrics     │  ║
║   │  • Images    │  │ • Query cache│  │ • DB Creds     │  │  • Logs        │  ║
║   │  • Summaries │  │              │  │ • vLLM tokens  │  │  • Alarms      │  ║
║   └──────────────┘  └──────────────┘  └────────────────┘  └───────────────┘  ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
```

---

## Component Deep Dive

### 1. API Layer — Application Load Balancer (ALB)

```
Client Request → Route 53 (DNS) → WAF → ALB → Target Group → ECS Container

Routing Rules:
  POST /api/query      → Query Service (ECS Fargate)
  POST /api/ingest     → Ingestion Service (ECS Fargate)
  GET  /health         → Health Check (returns 200 if all services up)
  GET  /api/status     → Pipeline status (ongoing ingestions, queue depth)
```

| Config | Value | Why |
|--------|-------|-----|
| HTTPS listener | Port 443 | SSL termination at ALB (ACM free certificate) |
| Health check | `/health` every 30s | Remove unhealthy containers from rotation |
| Idle timeout | 120 seconds | LLM API responses can take 5-10s |
| Sticky sessions | Disabled | Each request is independent |

### 2. Query Service — ECS Fargate (Auto-Scaling)

This replaces your current `query.py` CLI with a **REST API**:

```python
# FastAPI web server (query_api.py)
from fastapi import FastAPI
app = FastAPI()

@app.post("/api/query")
async def query(request: QueryRequest):
    # 1. Hybrid search (BM25 + semantic) — CPU only
    # 2. Cross-encoder reranking — CPU only, no KV cache
    # 3. LLM answer synthesis — External API (no GPU needed!)
    return {"answer": answer, "sources": sources}
```

| Config | Value | Why |
|--------|-------|-----|
| Container | 4 vCPU, 8 GB RAM | Enough for reranker + BM25 + embedding (CPU) |
| Min instances | 2 | Always-on for fast response |
| Max instances | 10 | Handles traffic spikes |
| Auto-scale trigger | CPU > 70% for 2 min | Scales out during peak |
| Reranker | MiniLM-L-6-v2 (22M, ~90 MB, CPU) | No KV cache needed — scores in one forward pass |
| Embed model | nemotron-embed-1b (1B, ~2 GB, CPU) | No KV cache needed — encoder model |
| Answer synthesis | External LLM API | No KV cache concern — provider manages all |

> **Key insight:** Nothing in the query container needs KV caching or GPU.
> Reranker and embedding model are encoder-based (single forward pass, no generation).
> Answer synthesis is offloaded to an external API. The query service is 100% CPU.

### 3. GPU Tier — EC2 g6.2xlarge (INGESTION ONLY, On-Demand)

**Critical change:** The GPU is now used **only for image summarization during ingestion** — NOT for answering user queries. This means:
- GPU can be **turned off** when not ingesting (saves ~95% GPU cost)
- KV cache is only needed for short image summary prompts (~500 tokens)
- Concurrency is low and controlled (batch processing, 1-4 images at a time)

```
GPU Instance Internals (ON only during ingestion):
┌──────────────────────────────────────────────┐
│  g6.2xlarge                                   │
│  └── L4 GPU (24 GB VRAM)                     │
│      └── vLLM Server                         │
│          ├── Model: nemotron-nano-vl-8b (16 GB)│
│          ├── KV Cache Pool (~6.5 GB)          │
│          ├── Image summaries: ~500 tok each   │
│          ├── KV per request: ~0.07 GB (tiny!) │
│          ├── Max concurrent: 4 images         │
│          ├── Total KV needed: ~0.28 GB        │
│          └── 6+ GB VRAM headroom              │
│                                               │
│  └── CPU (8 vCPUs, 32 GB RAM)                │
│      └── System processes only               │
│                                               │
│  ⏰ ON: ~3 hrs/week (during PDF ingestion)    │
│  💤 OFF: rest of the time (not needed for     │
│     queries — synthesis via external API)     │
└──────────────────────────────────────────────┘
```

| Config | Value | Why |
|--------|-------|-----|
| Instance | g6.2xlarge | L4 GPU, 24 GB VRAM, 8 vCPU, 32 GB RAM |
| vLLM params | `--gpu-mem-util 0.90 --max-model-len 2048` | Image prompts are short, 2048 is enough |
| Placement | Private subnet | No public internet access |
| Scheduling | **On-demand** — start before ingestion, stop after | Pay only for ~3 hrs/week |
| Auto-scaling | Not needed | Batch ingestion, controlled concurrency |

### 4. Ingestion Service — ECS Fargate (On-Demand)

Runs only when new PDFs are uploaded. Not always-on.

```
S3 Event (new PDF uploaded)
  → SQS Queue
    → ECS Task (ingestion worker)
      → PDFExtractor → ImageSummarizer (→ vLLM) → Chunker → Embed → PGVector
        → S3 (save images + cache)
```

| Config | Value | Why |
|--------|-------|-----|
| Trigger | S3 PUT event via SQS | Automatic ingestion on PDF upload |
| Container | 4 vCPU, 16 GB RAM | PDF parsing is memory-intensive |
| Concurrency | 1-3 simultaneous | Controlled by SQS + GPU capacity |
| Timeout | 30 minutes | One PDF with 300+ images |
| Auto-terminate | After task completion | Pay only for active processing |

### 5. Database — RDS PostgreSQL + pgvector

```
Production Database Setup:

Primary (AZ-1):                    Standby (AZ-2):
┌─────────────────────┐           ┌─────────────────────┐
│ db.r6g.xlarge       │           │ db.r6g.xlarge       │
│ • 4 vCPUs, 32 GB    │──sync──►│ • Auto-failover      │
│ • pgvector + HNSW   │ repl.    │ • Takes over in <60s │
│ • 500 GB gp3 (SSD)  │           │ • Same data          │
│ • Encrypted at rest │           │ • Same region, diff. │
└─────────────────────┘           │   availability zone  │
                                  └─────────────────────┘

HNSW Index (created once):
  CREATE INDEX ON langchain_pg_embedding
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 200);
```

| Config | Value | Why |
|--------|-------|-----|
| Instance | db.r6g.xlarge | 32 GB RAM for HNSW index in memory |
| Multi-AZ | Enabled | Auto-failover if primary dies |
| Storage | 500 GB gp3, 3000 IOPS | Fast reads for vector search |
| Backups | Daily + 7-day retention | Point-in-time recovery |
| Encryption | AES-256 at rest + SSL in transit | Compliance |

### 6. Redis Cache — ElastiCache

Caches query results to avoid redundant LLM API calls (saves cost and latency):

```
User asks: "What is the SLA?"
  → Check Redis: key = hash("What is the SLA?")
    → HIT:  Return cached answer instantly (~5ms) — no API cost!
    → MISS: Run full pipeline (3-5 sec), cache result, return
```

| Config | Value | Why |
|--------|-------|-----|
| Instance | cache.r6g.large (13 GB) | Stores ~10,000 cached query results |
| TTL | 1 hour | Balance freshness vs API cost savings |
| Eviction | LRU (Least Recently Used) | Auto-remove stale entries |

> **Cost impact:** Every Redis HIT saves one LLM API call (~$0.01-0.05). With 50% hit rate on 1000 queries/day, Redis saves ~$150-750/month in API costs.

---

## Request Flow

### Query Flow (User Asks a Question)

```
                    Time
User ──────────────────────────────────────────────────────►

1. HTTP POST /api/query {"question": "What is the SLA?"}
   │
   ▼
2. ALB routes to Query Container (ECS)              [~2ms]
   │
   ▼
3. Check Redis cache                                 [~5ms]
   │
   ├── CACHE HIT → Return instantly                 [total: ~7ms]
   │
   └── CACHE MISS ↓
       │
       ▼
4. Hybrid Search                                     [~200ms]
   ├── BM25 keyword search (in-memory)              [~10ms]
   ├── Semantic search (PGVector + HNSW)             [~50ms]
   │   └── Embed query via CPU (nemotron-embed-1b)   [~150ms]
   └── RRF fusion                                    [~5ms]
       │
       ▼
5. Cross-encoder reranking (CPU, no KV cache)         [~300ms]
   │    MiniLM-L-6-v2: single forward pass → score
   │    No autoregressive generation → no KV cache
   ▼
6. LLM answer synthesis                              [~2-4s]
   └── HTTP call to External LLM API
       └── Provider handles KV cache, concurrency, GPU
       └── No GPU needed on your infrastructure!
       │
       ▼
7. Cache result in Redis, return to user             [~5ms]

Total: ~3-5 seconds (dominated by LLM API call)
```

### Ingestion Flow (New PDF Uploaded)

```
1. Client uploads PDF to S3 bucket                   [~2s]
   │
   ▼
2. S3 event notification → SQS queue                 [~1s]
   │
   ▼
3. ECS ingestion worker picks up task                [~5s]
   │
   ▼
4. PDFExtractor (PyMuPDF)                            [~10s]
   ├── Extract text per page
   ├── Extract embedded images
   └── Render pages with charts/tables
   │
   ▼
5. ImageSummarizer → vLLM (GPU)                      [~10-20min]
   ├── 200-300 images × ~3s each
   ├── Rate: limited by GPU throughput
   └── Cache summaries to S3
   │
   ▼
6. Reassembler + SmartChunker                        [~5s]
   │
   ▼
7. Embedding (CPU — nemotron-embed-1b)               [~2min]
   ├── 500+ chunks × ~200ms each
   └── Batch processing
   │
   ▼
8. Store in PGVector (RDS)                           [~30s]
   └── Rebuild HNSW index
   │
   ▼
9. Invalidate Redis cache                            [~1ms]
   └── Clear stale query results

Total: ~15-25 minutes per PDF
```

---

## High Availability & Fault Tolerance

### What Happens When Things Fail

| Failure | Auto-Recovery | Downtime | How |
|---------|--------------|----------|-----|
| Query container crashes | ✅ | 0 seconds | ALB routes to healthy container, ECS replaces crashed one |
| External LLM API down | ⚠️ | Depends on provider | Fallback: switch API provider or return cached results |
| GPU instance dies (ingestion) | ⚠️ | 2-5 minutes | Relaunch — only affects ingestion, NOT user queries |
| RDS primary fails | ✅ | < 60 seconds | Multi-AZ auto-failover to standby |
| AZ-1 goes down entirely | ✅ | < 2 minutes | All services run in 2 AZs, traffic shifts to AZ-2 |
| Redis fails | ✅ | 0 seconds | Queries bypass cache, call LLM API directly (costs more) |
| Ingestion worker crashes | ✅ | 0 seconds | SQS re-delivers message, new worker picks it up |

> **Key HA improvement:** Since queries no longer depend on your GPU, a GPU failure only affects ingestion (batch, non-urgent) — user-facing queries remain 100% available.

### Multi-AZ Design

```
              Availability Zone 1              Availability Zone 2
          ┌─────────────────────────┐     ┌─────────────────────────┐
          │  Query Container (ECS)  │     │  Query Container (ECS)  │
          │  RDS Primary            │     │  RDS Standby            │
          │  Redis Primary          │     │  Redis Replica          │
          │  NAT Gateway            │     │  NAT Gateway            │
          └─────────────────────────┘     └─────────────────────────┘
                    │                              │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────┐
                    │  ALB (spans both AZs) │
                    └──────────────────────┘

  GPU Instance: NOT in AZ diagram — only runs on-demand for ingestion
  External LLM API: Outside AWS — provider handles HA

  If AZ-1 fails → ALL query traffic goes to AZ-2 automatically
  GPU failure → Only ingestion affected, user queries unimpacted
```

---

## Auto-Scaling Strategy

### Query Service Scaling

```
Auto-Scaling Policy:
  Metric: Average CPU utilization
  Target: 70%
  Min capacity: 2 containers
  Max capacity: 10 containers
  Scale-out cooldown: 60 seconds
  Scale-in cooldown: 300 seconds

Example:
  Normal traffic (10 req/min):   2 containers running
  Peak traffic (100 req/min):    5 containers running (auto-scaled)
  Spike (500 req/min):           10 containers running (max)
```

### GPU Instance Scaling (Ingestion Only)

```
Scheduling Policy (not auto-scaling — ingestion is batch):
  Trigger: SQS message received (new PDF uploaded)
  Action:  Start g6.2xlarge → run vLLM → process PDF → stop instance
  Min: 0 GPU instances (OFF when not ingesting)
  Max: 1 GPU instance

Cost optimization:
  • GPU ON only during active ingestion (~3 hrs/week)
  • Use Spot instance (70% cheaper — ingestion is interruptible)
  • If Spot interrupted, SQS re-delivers message, retry on new Spot
  • Monthly GPU cost: ~$3-12 instead of $704!
```

---

## Networking & Security

### VPC Layout

```
VPC: 10.0.0.0/16

  Public Subnets:
    10.0.1.0/24 (AZ-1) — ALB, NAT Gateway
    10.0.2.0/24 (AZ-2) — ALB, NAT Gateway

  Private Subnets:
    10.0.10.0/24 (AZ-1) — ECS, GPU, RDS Primary
    10.0.20.0/24 (AZ-2) — ECS, GPU, RDS Standby
```

### Security Groups

| Component | Inbound | Outbound |
|-----------|---------|----------|
| ALB | 443 from 0.0.0.0/0 (HTTPS) | 8080 to Query SG |
| Query Service | 8080 from ALB SG | 443 to External LLM API (NAT), 5432 to RDS SG, 6379 to Redis SG |
| GPU Instance | 8000 from Ingestion SG only | 443 to S3 (VPC endpoint) |
| RDS | 5432 from Query SG + Ingestion SG | None |
| Redis | 6379 from Query SG | None |

### Security Checklist

- [x] No public IPs on GPU, app servers, or database
- [x] ALB is the only public-facing component
- [x] IAM roles (not access keys) for all AWS service access
- [x] Secrets Manager for DB credentials, API keys
- [x] S3 VPC endpoint (no internet transit for files)
- [x] RDS encryption at rest (AES-256) + SSL in transit
- [x] AWS WAF on ALB (rate limiting, injection protection)
- [x] CloudTrail for audit logging
- [x] VPC Flow Logs for network monitoring

---

## CI/CD Pipeline

```
Developer pushes code
        │
        ▼
┌───────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  GitHub / GitLab   │───►│  AWS CodePipeline │───►│  CodeBuild       │
│  (source)          │    │  (orchestrator)   │    │  • Run tests     │
│                    │    │                    │    │  • Build Docker  │
│  main branch       │    │  Trigger: on push │    │  • Push to ECR   │
└───────────────────┘    └──────────────────┘    └────────┬─────────┘
                                                          │
                          ┌───────────────────────────────┘
                          ▼
                ┌──────────────────┐    ┌──────────────────┐
                │  ECR (Container   │───►│  ECS Deploy       │
                │  Registry)        │    │  (rolling update) │
                │  • query-api      │    │  • Blue/green     │
                │  • ingest-worker  │    │  • Zero downtime  │
                └──────────────────┘    └──────────────────┘
```

---

## Monitoring & Alerting

### Dashboard Metrics

| Metric | Source | Alert Threshold | Action |
|--------|--------|----------------|--------|
| **LLM API Latency (P95)** | Application logs | > 8 seconds | Check provider status, consider backup API |
| **LLM API Error Rate** | Application logs | > 2% | Switch to backup provider |
| **LLM API Cost** | Provider dashboard | > budget threshold | Review caching, reduce prompt size |
| Query API Latency (P95) | ALB + CloudWatch | > 6 seconds | Scale out ECS |
| Query API Error Rate | ALB metrics | > 1% | Page on-call |
| Redis Hit Rate | CloudWatch | < 50% | Increase TTL or cache size (saves API cost) |
| RDS CPU | CloudWatch | > 80% for 15 min | Scale up instance |
| RDS Connections | CloudWatch | > 80% of max | Check connection leaks |
| SQS Queue Depth | CloudWatch | > 10 messages | Start GPU + scale ingestion workers |
| GPU Utilization (ingestion) | NVIDIA DCGM | Ingestion complete | Stop GPU instance to save cost |
| S3 Storage | CloudWatch | > 1 TB | Cost review |

---

## Cost Breakdown

### Monthly Cost (Production Configuration)

| Component | Instance/Config | On-Demand | Notes |
|-----------|----------------|-----------|-------|
| **GPU (Ingestion ONLY)** (g6.2xlarge) | ~12 hrs/month (Spot) | **~$3-12** | ON only during PDF ingestion |
| **External LLM API** | Usage-based | **~$50-300** | Depends on query volume (see below) |
| **Query Service** (ECS Fargate, 2 containers) | 4 vCPU, 8 GB each | $290 | Always-on, no GPU needed |
| **Ingestion Worker** (ECS Fargate, on-demand) | 4 vCPU, 16 GB, ~40 hrs/mo | $25 | — |
| **Database** (RDS r6g.xlarge, Multi-AZ) | 4 vCPU, 32 GB, 500 GB | $748 | $472 with Reserved |
| **Redis** (ElastiCache r6g.large) | 13 GB | $195 | $123 with Reserved, saves API cost |
| **S3** (Standard, ~50 GB) | Images + PDFs + cache | $2 | — |
| **ALB** | Per-hour + LCU | $25 | — |
| **NAT Gateway** (2×, multi-AZ) | Per-hour + data | $70 | — |
| **CloudWatch + Logs** | Metrics, dashboards, alarms | $30 | — |
| **Secrets Manager** | 5 secrets | $2 | — |
| **Route 53** | Hosted zone + queries | $1 | — |
| **WAF** | 3 rules | $12 | — |
| | | | |
| **Total** | | **~$1,453-1,703/mo** | |
| **Total (Reserved DB + Redis)** | | **~$1,050-1,300/mo** | |

### LLM API Cost Estimation

| Query Volume | Avg Tokens/Query | Monthly API Cost (est.) |
|-------------|-----------------|------------------------|
| 100 queries/day | ~4000 tokens | ~$50/mo |
| 500 queries/day | ~4000 tokens | ~$150/mo |
| 1000 queries/day | ~4000 tokens | ~$300/mo |
| With 50% Redis cache hit rate | — | **Halve the above** |

> Pricing varies by provider. NVIDIA NIM Enterprise ~$1-4K/mo (unlimited). OpenAI GPT-4o-mini ~$0.15/1M input tokens.

### Cost Comparison: Old vs New Architecture

| | Old (GPU 24/7 for everything) | New (API synthesis + on-demand GPU) | Savings |
|---|---|---|---|
| GPU | $704/mo (24/7) | ~$12/mo (on-demand) | **$692** |
| LLM API | $0 | ~$150/mo | -$150 |
| Other infra | $1,400 | $1,400 | $0 |
| **Total** | **~$2,104** | **~$1,562** | **~$542/mo (26%)** |

### Cost Optimization Levers

| Strategy | Monthly Savings | How |
|----------|----------------|-----|
| Reserved Instances (DB + Redis) | ~$348 | 1-year commitment |
| Spot for GPU ingestion | ~$8 | Spot = 70% cheaper than on-demand |
| Redis caching (50% hit rate) | ~$75-150 | Avoid redundant LLM API calls |
| Right-size ECS containers | ~$100 | Monitor CPU/memory, reduce if underused |
| **Fully optimized** | | **~$930-1,080/mo** |

### Per-Ingestion Cost (What Each PDF Costs to Process)

```
One PDF with 300 images:

  GPU time:
    300 images × ~3 sec/image = ~15 minutes of GPU time
    g6.2xlarge: $0.978/hr → 15 min = $0.24
    g5.2xlarge: $1.212/hr → 15 min = $0.30
    Spot (g6):  ~$0.29/hr → 15 min = $0.07

  Embedding (CPU, free — runs inside ECS container):
    550 chunks × ~200ms = ~2 min → $0.00 (already paid for ECS)

  ECS Fargate (ingestion worker):
    4 vCPU, 16 GB × ~25 min = ~$0.04

  Total per PDF:
    On-Demand (g6): ~$0.28
    On-Demand (g5): ~$0.34
    Spot (g6):      ~$0.11

  If you ingest 10 PDFs/month:
    On-Demand: ~$2.80
    Spot:      ~$1.10
```

---

## L4 vs A10G — Which GPU for Your Pipeline?

### Hardware Specs Comparison

| Spec | NVIDIA L4 (g6) | NVIDIA A10G (g5) |
|------|----------------|-----------------|
| **Architecture** | Ada Lovelace (2023) | Ampere (2021) |
| **VRAM** | 24 GB GDDR6 | 24 GB GDDR6X |
| **Memory Bandwidth** | 300 GB/s | 600 GB/s |
| **FP16 TFLOPS** | 121 TFLOPS | 125 TFLOPS |
| **INT8 TOPS** | 242 TOPS | 250 TOPS |
| **TDP (Power)** | **72W** ⚡ | 150W |
| **Tensor Cores** | 4th Gen | 3rd Gen |
| **FP8 Support** | ✅ Yes | ❌ No |
| **Designed For** | Inference | Training + Inference |

### Performance for Your Use Case

```
nemotron-nano-vl-8b (FP16, image summarization):

  L4:   ~2.8 sec/image  (power efficient, newer arch)
  A10G: ~2.5 sec/image  (slightly faster due to higher bandwidth)
  Difference: ~12% faster on A10G

  For 300 images:
    L4:   ~14 min
    A10G: ~12.5 min
    Difference: ~1.5 minutes (negligible for batch processing)
```

### Cost Comparison (Same VRAM, Different Price)

| | L4 (g6.2xlarge) | A10G (g5.2xlarge) | Difference |
|---|---|---|---|
| **Hourly rate** | **$0.978** | $1.212 | L4 is **19% cheaper** |
| **3 hrs/week ingestion** | **$12/mo** | $15/mo | L4 saves $3/mo |
| **3 hrs/week Spot** | **~$3.50/mo** | ~$4.55/mo | L4 saves $1/mo |
| **24/7 On-Demand** | **$704/mo** | $873/mo | L4 saves **$169/mo** |
| **Performance** | Baseline | ~12% faster | A10G slightly faster |
| **Power consumption** | **72W** | 150W | L4 uses **52% less power** |
| **FP8 quantization** | ✅ Supported | ❌ Not supported | L4 can run FP8 |

### With FP8 Quantization (L4 Exclusive Advantage)

```
L4 supports FP8 natively → model weights shrink from 16 GB to 8 GB

  FP16 (both L4 and A10G):
    Model: 16 GB | Free VRAM: 7 GB | KV capacity: ~6 images concurrent

  FP8 (L4 only):
    Model: 8 GB  | Free VRAM: 15 GB | KV capacity: ~14 images concurrent
    Accuracy loss: < 1% (negligible for image summaries)

  This means L4 with FP8 can:
    • Process images 2× faster (more concurrent batches)
    • Or use --max-model-len 8192 for longer context
```

### Decision: L4 or A10G?

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  RECOMMENDATION: L4 (g6.2xlarge)  ✅                        ║
║                                                              ║
║  Why:                                                        ║
║  1. 19% cheaper ($0.978 vs $1.212/hr)                       ║
║  2. FP8 support → halve model VRAM (L4 exclusive)           ║
║  3. 52% less power → lower thermal throttling risk          ║
║  4. Newer architecture (Ada Lovelace vs Ampere)             ║
║  5. Only 12% slower than A10G (irrelevant for batch)        ║
║                                                              ║
║  When to choose A10G instead:                               ║
║  • g6 instances unavailable in your AWS region              ║
║  • You need maximum memory bandwidth (streaming tasks)       ║
║  • Running training workloads (not your case)               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### Summary: Total Monthly Cost with L4

```
Fully optimized production architecture:

  GPU (L4 Spot, ~12 hrs/mo):          $    3
  External LLM API (500 q/day):       $  150
  ECS Query Service (2 containers):   $  290
  ECS Ingestion Worker:               $   25
  RDS PostgreSQL (Reserved):          $  472
  Redis (Reserved):                   $  123
  S3 + ALB + NAT + CloudWatch + misc: $  140
                                      ──────
  TOTAL:                              ~$1,203/month

  vs A10G would be:                   ~$1,206/month (only $3 more)

  → GPU choice barely matters at on-demand ingestion scale!
    But L4 is still better for FP8 support and future-proofing.
```

---

## Deployment Checklist

### Phase 1: Foundation (Week 1)

- [ ] Create VPC with 2 public + 2 private subnets across 2 AZs
- [ ] Set up NAT Gateways (one per AZ)
- [ ] Create S3 bucket for PDFs, images, summary cache
- [ ] Launch RDS PostgreSQL with pgvector, Multi-AZ
- [ ] Create HNSW index on vector table
- [ ] Set up Secrets Manager with all credentials
- [ ] Migrate existing embeddings from local Docker to RDS

### Phase 2: GPU Tier (Week 2)

- [ ] Launch g6.2xlarge in private subnet
- [ ] Install vLLM, download nemotron-nano-vl-8b weights
- [ ] Configure vLLM with production params
- [ ] Set up systemd service for auto-restart
- [ ] Create AMI snapshot (for fast auto-scaling)
- [ ] Test VLM inference latency and throughput

### Phase 3: Application Tier (Week 3)

- [ ] Build Docker images for query-api and ingest-worker
- [ ] Push to ECR (Elastic Container Registry)
- [ ] Create ECS Fargate cluster + task definitions
- [ ] Set up ALB with HTTPS (ACM certificate)
- [ ] Configure auto-scaling policies
- [ ] Deploy ElastiCache Redis
- [ ] Set up SQS queue for ingestion triggers

### Phase 4: Security & Monitoring (Week 4)

- [ ] Enable AWS WAF on ALB
- [ ] Configure CloudWatch dashboards + alarms
- [ ] Set up VPC Flow Logs
- [ ] Enable CloudTrail
- [ ] Security group audit
- [ ] Load testing (50-100 concurrent users)
- [ ] Disaster recovery test (kill AZ-1, verify failover)

### Phase 5: CI/CD & Go-Live (Week 5)

- [ ] Set up CodePipeline for automated deployments
- [ ] Configure blue/green deployment strategy
- [ ] DNS switchover (Route 53)
- [ ] Production smoke tests
- [ ] Hand-off documentation + runbook
- [ ] Go live 🚀
