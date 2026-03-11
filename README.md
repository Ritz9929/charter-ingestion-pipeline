# 📄 Multimodal RAG Ingestion & Retrieval Pipeline

A modular Python pipeline that extracts text, images, charts, and tables from PDFs, summarizes visual content using a Vision Language Model (NVIDIA NIM), and stores everything in a persistent PostgreSQL vector database (PGVector) for accurate Retrieval-Augmented Generation (RAG).

---

## 🏗️ Architecture

```
┌────────────┐    ┌─────────────────┐    ┌───────────────────────┐    ┌──────────────┐    ┌──────────────┐
│  PDF Input ├───►│  PDFExtractor   ├───►│  ImageSummarizer      ├───►│  Reassembler ├───►│ SmartChunker │
│            │    │  (PyMuPDF)      │    │  (NVIDIA NIM)         │    │              │    │  (800/100)   │
└────────────┘    │ • Text per page │    │                       │    │ Injects      │    │              │
                  │ • Embedded imgs │    │ qwen3.5-122b-a10b     │    │ [IMAGE_REF]  │    │ Never splits │
                  │ • Rendered pages│    │ (VLM — 122B MoE)      │    │ tags         │    │ image tags   │
                  │   (charts/tables│    │ + Summary caching     │    │              │    │              │
                  └─────────────────┘    └───────────────────────┘    └──────────────┘    └──────┬───────┘
                                                                                                │
                                                                                                ▼
┌────────────────────┐    ┌────────────────────┐    ┌──────────────────────────────────────────────────────┐
│  Answer Synthesis  │◄───│  Cross-Encoder     │◄───│  VectorStore (PGVector — Persistent PostgreSQL)     │
│  (NVIDIA NIM)      │    │  Reranker          │    │                                                      │
│                    │    │  ms-marco-MiniLM   │    │  Embeddings: llama-nemotron-embed-1b-v2 (NVIDIA NIM) │
│ qwen3.5-122b-a10b │    │  (local)           │    │  Asymmetric: passage/query input types               │
└────────────────────┘    └────────────────────┘    └──────────────────────────────────────────────────────┘
       query.py                query.py                              pipeline.py
```

---

## 🤖 Models Used

| Component | Model | Provider | Details |
|-----------|-------|----------|---------|
| **VLM (Image Summarization)** | `qwen/qwen3.5-122b-a10b` | NVIDIA NIM | 122B MoE (10B active), vision-language model, 40 RPM free tier |
| **Embeddings** | `nvidia/llama-nemotron-embed-1b-v2` | NVIDIA NIM | 1B param, asymmetric model (passage/query), retrieval-optimized |
| **Answer Synthesis** | `qwen/qwen3.5-122b-a10b` | NVIDIA NIM | Same VLM for generating coherent answers from retrieved chunks |
| **Cross-Encoder Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local (HuggingFace) | Runs locally, reranks top-20 → top-5 for better relevance |

> **Note**: All API calls use NVIDIA NIM's free tier via OpenAI-compatible endpoints. Each model requires its own API key from [build.nvidia.com](https://build.nvidia.com).

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- Docker Desktop (for PostgreSQL + pgvector)
- NVIDIA NIM API keys (free from [build.nvidia.com](https://build.nvidia.com))

### 2. Setup

```bash
# Navigate to the project
cd Ingestion_prototype

# Create and activate a virtual environment
python -m venv venv
source venv/Scripts/activate    # Windows (Git Bash)
# or: venv\Scripts\activate     # Windows (CMD)
# or: source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 3. Database Setup (One-time)

```bash
# Start a PostgreSQL container with pgvector
docker run --name local-rag-db \
  -e POSTGRES_PASSWORD=mysecretpassword \
  -p 5432:5432 \
  -v pgvector_data:/var/lib/postgresql/data \
  -d pgvector/pgvector:pg17

# Enable the vector extension
docker exec local-rag-db psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 4. Configure API Keys

Create a `.env` file (use `.env.example` as template):

```env
# Get keys from each model's page on https://build.nvidia.com
NVIDIA_VLM_API_KEY=nvapi-xxxxx      # From qwen3.5-122b-a10b model page
NVIDIA_EMBED_API_KEY=nvapi-xxxxx    # From llama-nemotron-embed-1b-v2 model page

# PostgreSQL connection
PG_CONNECTION_STRING=postgresql+psycopg://postgres:mysecretpassword@localhost:5432/postgres
```

### 5. Run

```bash
# Make sure Docker container is running
docker start local-rag-db

# Place your PDF as 'sample.pdf' in the project root

# Run ingestion pipeline
python main.py

# Query interactively (instant — uses existing embeddings)
python query.py
```

---

## 📁 Project Structure

```
Ingestion_prototype/
├── pipeline.py              # Core ingestion pipeline (6 classes + orchestrator)
├── query.py                 # Interactive query tool (reranker + LLM synthesis)
├── main.py                  # Entry point — runs pipeline and prints results
├── requirements.txt         # Python dependencies
├── .env                     # Your API keys (not committed)
├── .env.example             # Template for .env
├── mock_s3_storage/         # Auto-created — extracted images stored here
│   └── summaries_cache.json # Cached VLM summaries (never re-summarize)
└── README.md                # This file
```

---

## 🧩 Pipeline Modules (pipeline.py)

### Step 1: `PDFExtractor`

Parses PDFs using **PyMuPDF (fitz)** with two extraction methods:

| Method | What It Captures | How |
|--------|-----------------|-----|
| **Embedded images** | Photos, logos, raster graphics | `page.get_images()` — extracts raw image data |
| **Rendered pages** | Charts, graphs, tables (vector-drawn) | `page.get_pixmap()` — renders full page at 200 DPI |

**Filters applied:**
- Images < 1KB are skipped (icons/artifacts)
- Duplicate images on the same page are deduplicated by xref
- Pages are only rendered if they have ≥10 vector drawing operations AND no embedded raster images (avoids duplication)

```python
extractor = PDFExtractor(output_dir="./mock_s3_storage", render_dpi=200)
result = extractor.extract("sample.pdf")
# result.page_texts  → {page_num: text}
# result.images      → [ExtractedImage(...), ...]
```

---

### Step 2: `ImageSummarizer`

Sends each image to the **NVIDIA NIM VLM** (`qwen/qwen3.5-122b-a10b`) for structured summarization.

**Key features:**
- **Image resizing**: Images > 1024px are downscaled before sending (reduces payload from ~15MB to ~200KB)
- **JPEG compression**: Non-transparent images are converted to JPEG (quality=85) for smaller payloads
- **Exponential backoff retry**: 5 retries with 5s → 160s waits for rate limit (429), server (502/503) errors
- **Rate limiting**: 1.5s delay between calls (40 RPM compliance)
- **Summary caching**: Summaries are saved to `summaries_cache.json` after each image. On re-run, cached summaries are loaded instantly — no API calls needed

**Structured summary output** (4 Markdown sections):

| Section | Content |
|---------|---------|
| 1. Image Type & Core Subject | Type of image + main title/subject |
| 2. Explicit Text & Labels | All visible text — titles, axis labels, legends, node labels |
| 3. Data & Relationships | Translated visual data — values, trends, flows, connections |
| 4. Semantic Keywords | 5-10 keywords for vector similarity matching |

```python
summarizer = ImageSummarizer(model_name="qwen/qwen3.5-122b-a10b")
summarizer.summarize_all(extraction.images, cache_path="./mock_s3_storage/summaries_cache.json")
```

---

### Step 3: `DocumentReassembler`

Reconstructs the full document text, injecting `IMAGE_REFERENCE` tags at the correct page positions:

```
[IMAGE_REFERENCE | URL: /mock_s3_storage/page2_rendered.png | SUMMARY: The chart shows quarterly revenue ...]
```

- Images are grouped by page and sorted by position index
- Summaries are cleaned (newlines collapsed) so tags stay on one line
- Pages are ordered sequentially with image tags appended after their page's text

---

### Step 4: `SmartChunker`

Wraps LangChain's `RecursiveCharacterTextSplitter` with a **critical guarantee**: `[IMAGE_REFERENCE ...]` tags are **never split in half**.

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `chunk_size` | 800 | Maximum characters per chunk |
| `chunk_overlap` | 100 | Overlap between consecutive chunks for context continuity |

**Chunking strategy:**
1. Pre-split text around `[IMAGE_REFERENCE ...]` tags using regex
2. Each tag becomes an **atomic unit** (never subdivided)
3. Text segments between tags are split recursively using separators:
   - `\n\n` (paragraph break) → `\n` (line break) → `. ` (sentence) → ` ` (word) → `` (character)
4. Image tags are inserted as standalone chunks

```python
chunker = SmartChunker(chunk_size=800, chunk_overlap=100)
chunks = chunker.chunk(reassembled_text)
```

---

### Step 5: `NvidiaEmbeddings` + `VectorStoreManager`

#### Embedding Model: `nvidia/llama-nemotron-embed-1b-v2`

This is an **asymmetric embedding model** — it uses different encoding for documents vs queries:

| Operation | `input_type` | When |
|-----------|-------------|------|
| Storing documents | `"passage"` | During ingestion (`embed_documents`) |
| Searching | `"query"` | During retrieval (`embed_query`) |

The custom `NvidiaEmbeddings` class handles this automatically using the raw OpenAI client (avoids Pydantic/langchain-openai compatibility issues).

**Batching**: Documents are embedded in batches of 50 to avoid payload limits.

#### Vector Store: PGVector (PostgreSQL)

- **Persistent storage** — embeddings survive process restarts and container reboots
- **Collection name**: `multimodal_rag`
- **Pre-delete on re-ingest**: `pre_delete_collection=True` ensures clean slate each run
- **Metadata per chunk**: `chunk_index`, `has_image_ref`, `char_count`

```python
store = VectorStoreManager(collection_name="multimodal_rag")
vectorstore = store.ingest(chunks)        # Ingest (pipeline)
vectorstore = store.connect()              # Connect to existing (query)
```

---

## 🔍 Query Module (query.py)

Interactive query tool with a **3-stage retrieval pipeline**:

### Stage 1: Vector Search
- Retrieves **top 20 candidates** from PGVector via cosine similarity
- Uses `nvidia/llama-nemotron-embed-1b-v2` with `input_type="query"` for the search embedding

### Stage 2: Cross-Encoder Reranking
- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (runs locally)
- Processes each (query, chunk) pair jointly for fine-grained relevance scoring
- Reranks 20 candidates → **top 5** most relevant

### Stage 3: LLM Answer Synthesis
- **Model**: `qwen/qwen3.5-122b-a10b` (via NVIDIA NIM)
- Takes the top 5 chunks as context + user question
- Generates a coherent, cited answer (only from provided context)
- Displays formatted answer + supporting sources with type indicators (📄 text / 🖼️ image)

```bash
python query.py
# 📝 Your question: What are the main features of TechMobile?
```

---

## ⚙️ Configuration

### Swapping the VLM Model

Pass any NVIDIA NIM vision model:

```python
vectorstore, chunks = run_pipeline(
    pdf_path="report.pdf",
    vlm_model_name="qwen/qwen3.5-397b-a17b",  # Larger, slower, more capable
)
```

### Custom Embeddings

Pass any LangChain-compatible embedding model:

```python
from langchain_huggingface import HuggingFaceEmbeddings

custom_emb = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
vectorstore, chunks = run_pipeline("report.pdf", embedding_model=custom_emb)
```

### Adjusting Chunk Size

```python
vectorstore, chunks = run_pipeline("report.pdf", chunk_size=1200, chunk_overlap=200)
```

### Clearing the Summary Cache

Delete the cache to force re-summarization of all images:

```bash
rm mock_s3_storage/summaries_cache.json
```

---

## 🔧 Troubleshooting

### SSL Certificate Errors (Corporate Networks)

The pipeline includes automatic SSL handling via `truststore`, which uses the OS system certificate store. If you still see SSL errors:

```bash
pip install truststore
```

### NVIDIA NIM 401 Unauthorized

- Ensure your API key is from the **correct model page** (each model has its own key)
- `NVIDIA_VLM_API_KEY` → from the `qwen3.5-122b-a10b` page
- `NVIDIA_EMBED_API_KEY` → from the `llama-nemotron-embed-1b-v2` page

### NVIDIA NIM 429 Rate Limit

The pipeline includes automatic retry with exponential backoff (5 retries, 5s → 160s). If you consistently hit limits:
- Wait a few minutes and try again
- Generate a new API key from the same model page
- Increase `DELAY_BETWEEN` in `ImageSummarizer` class (default: 1.5s)

### Docker / PGVector Issues

```bash
# Check if container is running
docker ps

# Start the container
docker start local-rag-db

# Verify pgvector extension
docker exec local-rag-db psql -U postgres -c "SELECT * FROM pg_extension WHERE extname='vector';"
```

### Viewing Data in DBeaver

Navigate to: `postgres` → `Schemas` → `public` → `Tables`

| Table | Contents |
|-------|----------|
| `langchain_pg_collection` | Collection metadata |
| `langchain_pg_embedding` | Chunks + embedding vectors |

Press **F5** to refresh data after a new ingestion run.

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `PyMuPDF` | PDF parsing, text/image extraction, page rendering |
| `openai` | NVIDIA NIM API client (OpenAI-compatible protocol) |
| `langchain` | Core LangChain framework |
| `langchain-text-splitters` | RecursiveCharacterTextSplitter for chunking |
| `langchain-postgres` | PGVector integration for persistent vector storage |
| `psycopg[binary]` | PostgreSQL driver for Python |
| `sentence-transformers` | Cross-encoder reranker (local, query.py) |
| `Pillow` | Image resizing and format conversion |
| `python-dotenv` | Environment variable loading from .env |
| `truststore` | OS-level SSL certificate handling |

---

## 📊 Performance Characteristics

| Metric | Value |
|--------|-------|
| **Image summarization speed** | ~5-10 seconds per image (qwen3.5-122b-a10b) |
| **319 images total time** | ~30-45 minutes (first run), instant (cached re-run) |
| **Embedding speed** | ~1-2 minutes for 550 chunks |
| **Query response time** | ~5-10 seconds (search + rerank + LLM synthesis) |
| **Embedding dimensions** | Model-dependent (llama-nemotron-embed-1b-v2) |
| **NVIDIA NIM rate limit** | 40 RPM per API key |
| **NVIDIA NIM free credits** | ~1000 per API key |

---

## 📄 License

This is a prototype/proof-of-concept for internal use.
