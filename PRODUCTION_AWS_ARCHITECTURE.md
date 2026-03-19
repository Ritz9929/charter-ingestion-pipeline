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
| VLM | NVIDIA NIM free tier (40 RPM) | Self-hosted vLLM (unlimited) |
| Embedding | NVIDIA NIM API | Self-hosted on CPU (zero cost) |
| Search | hybrid (BM25 + semantic) | Same, behind load balancer |
| Database | Docker container on laptop | RDS Multi-AZ (99.95% SLA) |
| File Storage | Local `mock_s3_storage/` | Amazon S3 (11 nines durability) |
| Availability | Your laptop uptime | 99.9%+ (multi-AZ, auto-restart) |
| Scaling | None | Auto-scaling GPU + app instances |
| Security | `.env` file | Secrets Manager + VPC + IAM |
| Monitoring | Print statements | CloudWatch + Grafana dashboards |
| CI/CD | Manual `git push` | Automated deploy on merge |

---

## Full Architecture Diagram

```
                                  INTERNET
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │     Amazon Route 53     │
                        │   (DNS + Health Checks)  │
                        │   api.yourcompany.com    │
                        └────────────┬───────────┘
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │    AWS WAF (Firewall)    │
                        │  • Rate limiting         │
                        │  • SQL injection block   │
                        │  • Bot detection          │
                        └────────────┬───────────┘
                                     │
                                     ▼
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
║   │   ┌─────────────────────────┐   ┌──────────────────────────────────┐   │ ║
║   │   │  QUERY SERVICE (ECS)    │   │  GPU INFERENCE TIER              │   │ ║
║   │   │  (Auto-scaling)         │   │                                  │   │ ║
║   │   │                         │   │  ┌────────────────────────────┐ │   │ ║
║   │   │  ┌───────────────────┐ │   │  │  g6.2xlarge (Primary)      │ │   │ ║
║   │   │  │ Query Container 1 │ │   │  │  1× L4 GPU (24 GB VRAM)   │ │   │ ║
║   │   │  │ • FastAPI server  │ │   │  │                            │ │   │ ║
║   │   │  │ • Hybrid search   │◄├───┤  │  vLLM Server (port 8000)  │ │   │ ║
║   │   │  │ • BM25 index      │ │   │  │  nemotron-nano-vl-8b      │ │   │ ║
║   │   │  │ • Reranker (CPU)  │ │   │  │  --gpu-mem-util 0.90      │ │   │ ║
║   │   │  │ • Embed (CPU)     │ │   │  │  --max-model-len 4096     │ │   │ ║
║   │   │  └───────────────────┘ │   │  │  --max-num-seqs 8         │ │   │ ║
║   │   │  ┌───────────────────┐ │   │  └────────────────────────────┘ │   │ ║
║   │   │  │ Query Container 2 │ │   │                                  │   │ ║
║   │   │  │ (auto-scaled)     │ │   │  ┌────────────────────────────┐ │   │ ║
║   │   │  └───────────────────┘ │   │  │  g6.2xlarge (Standby/      │ │   │ ║
║   │   │  ┌───────────────────┐ │   │  │  Auto-Scale)               │ │   │ ║
║   │   │  │ Query Container N │ │   │  │  (launches when primary    │ │   │ ║
║   │   │  │ (auto-scaled)     │ │   │  │   GPU > 80% utilization)   │ │   │ ║
║   │   │  └───────────────────┘ │   │  └────────────────────────────┘ │   │ ║
║   │   └─────────────────────────┘   └──────────────────────────────────┘   │ ║
║   │                                                                          │ ║
║   │   ┌─────────────────────────┐   ┌──────────────────────────────────┐   │ ║
║   │   │  INGESTION SERVICE      │   │  DATABASE TIER                   │   │ ║
║   │   │  (ECS — on-demand)      │   │                                  │   │ ║
║   │   │                         │   │  ┌────────────────────────────┐ │   │ ║
║   │   │  ┌───────────────────┐ │   │  │  RDS PostgreSQL (Primary)  │ │   │ ║
║   │   │  │ Ingest Worker     │ │   │  │  db.r6g.xlarge             │ │   │ ║
║   │   │  │ • PDFExtractor   │ │   │  │  • pgvector + HNSW index   │ │   │ ║
║   │   │  │ • ImageSummarizer│─┼───┤  │  • 32 GB RAM               │ │   │ ║
║   │   │  │ • SmartChunker   │ │   │  │  • Multi-AZ standby        │ │   │ ║
║   │   │  │ • VectorStore    │ │   │  │  • Automated backups       │ │   │ ║
║   │   │  └───────────────────┘ │   │  └─────────────┬──────────────┘ │   │ ║
║   │   └─────────────────────────┘   │                │                │   │ ║
║   │                                  │  ┌─────────────▼──────────────┐ │   │ ║
║   │                                  │  │  RDS PostgreSQL (Standby)  │ │   │ ║
║   │                                  │  │  (Auto-failover, AZ-2)     │ │   │ ║
║   │                                  │  └────────────────────────────┘ │   │ ║
║   │                                  └──────────────────────────────────┘   │ ║
║   └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                ║
║   ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  ┌───────────────┐  ║
║   │  Amazon S3    │  │ ElastiCache  │  │ Secrets Manager│  │  CloudWatch    │  ║
║   │  • PDFs      │  │ Redis        │  │ • API Keys     │  │  • Metrics     │  ║
║   │  • Images    │  │ • Query cache│  │ • DB Creds     │  │  • Logs        │  ║
║   │  • Summaries │  │ • BM25 index │  │ • vLLM tokens  │  │  • Alarms      │  ║
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
| Idle timeout | 120 seconds | VLM responses can take 5-10s |
| Sticky sessions | Disabled | Each request is independent |

### 2. Query Service — ECS Fargate (Auto-Scaling)

This replaces your current `query.py` CLI with a **REST API**:

```python
# FastAPI web server (query_api.py)
from fastapi import FastAPI
app = FastAPI()

@app.post("/api/query")
async def query(request: QueryRequest):
    # 1. Hybrid search (BM25 + semantic)
    # 2. Cross-encoder reranking
    # 3. LLM answer synthesis (calls vLLM on GPU instance)
    return {"answer": answer, "sources": sources}
```

| Config | Value | Why |
|--------|-------|-----|
| Container | 4 vCPU, 8 GB RAM | Enough for reranker + BM25 + embedding (CPU) |
| Min instances | 2 | Always-on for fast response |
| Max instances | 10 | Handles traffic spikes |
| Auto-scale trigger | CPU > 70% for 2 min | Scales out during peak |
| Model loaded | Reranker (88 MB) + Embed 1B (2 GB CPU) | Both run on CPU |

### 3. GPU Inference Tier — EC2 g6.2xlarge

Dedicated to **VLM inference only** (nemotron-nano-vl-8b via vLLM):

```
GPU Instance Internals:
┌──────────────────────────────────────────────┐
│  g6.2xlarge                                   │
│  └── L4 GPU (24 GB VRAM)                     │
│      └── vLLM Server                         │
│          ├── Model: nemotron-nano-vl-8b (16 GB)│
│          ├── KV Cache Pool (~6.5 GB)          │
│          ├── Max concurrent: 6-8 requests     │
│          ├── PagedAttention: enabled          │
│          ├── Continuous batching: enabled     │
│          └── Prefix caching: enabled          │
│                                               │
│  └── CPU (8 vCPUs, 32 GB RAM)                │
│      └── System processes only               │
└──────────────────────────────────────────────┘
```

| Config | Value | Why |
|--------|-------|-----|
| Instance | g6.2xlarge | L4 GPU, 24 GB VRAM, 8 vCPU, 32 GB RAM |
| vLLM params | `--gpu-mem-util 0.90 --max-model-len 4096` | Maximizes KV cache for concurrency |
| Placement | Private subnet | No public internet access |
| Auto-scaling | Launch 2nd GPU when utilization > 80% | Handles ingestion bursts |

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

Caches query results to avoid redundant GPU inference:

```
User asks: "What is the SLA?"
  → Check Redis: key = hash("What is the SLA?")
    → HIT:  Return cached answer instantly (~5ms)
    → MISS: Run full pipeline (5-8 sec), cache result, return
```

| Config | Value | Why |
|--------|-------|-----|
| Instance | cache.r6g.large (13 GB) | Stores ~10,000 cached query results |
| TTL | 1 hour | Balance freshness vs GPU savings |
| Eviction | LRU (Least Recently Used) | Auto-remove stale entries |

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
5. Cross-encoder reranking (CPU)                     [~300ms]
   │
   ▼
6. LLM answer synthesis                              [~3-5s]
   └── HTTP call to vLLM (GPU instance, port 8000)
       └── nemotron-nano-vl-8b generates answer
       │
       ▼
7. Cache result in Redis, return to user             [~5ms]

Total: ~4-6 seconds (dominated by LLM generation)
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
| GPU instance dies | ⚠️ | 2-5 minutes | Auto-scaling launches replacement, vLLM loads model |
| RDS primary fails | ✅ | < 60 seconds | Multi-AZ auto-failover to standby |
| AZ-1 goes down entirely | ✅ | < 2 minutes | All services run in 2 AZs, traffic shifts to AZ-2 |
| Redis fails | ✅ | 0 seconds | Queries bypass cache, hit GPU directly (slower but works) |
| Ingestion worker crashes | ✅ | 0 seconds | SQS re-delivers message, new worker picks it up |

### Multi-AZ Design

```
              Availability Zone 1              Availability Zone 2
          ┌─────────────────────────┐     ┌─────────────────────────┐
          │  Query Container (ECS)  │     │  Query Container (ECS)  │
          │  GPU Instance (g6.2xl)  │     │  GPU Instance (standby) │
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

If AZ-1 fails → ALL traffic goes to AZ-2 automatically
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

### GPU Instance Scaling

```
Auto-Scaling Policy:
  Metric: GPU utilization (via CloudWatch custom metric)
  Scale-out: GPU > 80% for 5 minutes → Launch 2nd GPU instance
  Scale-in:  GPU < 20% for 15 minutes → Terminate extra instance
  Min: 1 GPU instance (always on)
  Max: 3 GPU instances

Cost optimization:
  • Use Spot instances for 2nd/3rd GPU (70% cheaper, interruptible)
  • Primary GPU = On-Demand (always available)
  • Extra GPUs = Spot (for burst ingestion load)
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
| Query Service | 8080 from ALB SG | 8000 to GPU SG, 5432 to RDS SG, 6379 to Redis SG |
| GPU Instance | 8000 from Query SG + Ingestion SG | 443 to S3 (VPC endpoint) |
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
| GPU Utilization | NVIDIA DCGM → CloudWatch | > 80% for 5 min | Scale out GPU |
| GPU Memory Used | NVIDIA DCGM → CloudWatch | > 90% | Reduce `--max-num-seqs` |
| VLM Latency (P95) | vLLM metrics | > 10 seconds | Check GPU overload |
| Query API Latency (P95) | ALB + CloudWatch | > 8 seconds | Scale out ECS |
| Query API Error Rate | ALB metrics | > 1% | Page on-call |
| RDS CPU | CloudWatch | > 80% for 15 min | Scale up instance |
| RDS Connections | CloudWatch | > 80% of max | Check connection leaks |
| Redis Hit Rate | CloudWatch | < 50% | Increase TTL or cache size |
| SQS Queue Depth | CloudWatch | > 10 messages | Scale ingestion workers |
| S3 Storage | CloudWatch | > 1 TB | Cost review |

---

## Cost Breakdown

### Monthly Cost (Production Configuration)

| Component | Instance/Config | On-Demand | Reserved (1yr) |
|-----------|----------------|-----------|----------------|
| **GPU Inference** (g6.2xlarge, 1× L4) | 24/7 | $704 | $444 |
| **Query Service** (ECS Fargate, 2 containers) | 4 vCPU, 8 GB each | $290 | — |
| **Ingestion Worker** (ECS Fargate, on-demand) | 4 vCPU, 16 GB, ~40 hrs/mo | $25 | — |
| **Database** (RDS r6g.xlarge, Multi-AZ) | 4 vCPU, 32 GB, 500 GB | $748 | $472 |
| **Redis** (ElastiCache r6g.large) | 13 GB | $195 | $123 |
| **S3** (Standard, ~50 GB) | Images + PDFs + cache | $2 | — |
| **ALB** | Per-hour + LCU | $25 | — |
| **NAT Gateway** (2×, multi-AZ) | Per-hour + data | $70 | — |
| **CloudWatch + Logs** | Metrics, dashboards, alarms | $30 | — |
| **Secrets Manager** | 5 secrets | $2 | — |
| **Route 53** | Hosted zone + queries | $1 | — |
| **WAF** | 3 rules | $12 | — |
| | | | |
| **Total (On-Demand)** | | **~$2,104/mo** | |
| **Total (Reserved+Optimized)** | | | **~$1,474/mo** |

### Cost Optimization Levers

| Strategy | Monthly Savings | How |
|----------|----------------|-----|
| Reserved Instances (GPU + DB) | ~$536 | 1-year commitment |
| GPU off nights/weekends | ~$470 | Run 8hr/day weekdays (for ingestion-only GPU) |
| Spot for extra GPU instances | ~$490 | Use Spot for burst ingestion |
| Right-size ECS containers | ~$100 | Monitor CPU/memory, reduce if underused |
| **Fully optimized** | | **~$950/mo** |

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
