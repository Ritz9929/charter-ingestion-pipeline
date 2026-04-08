# Chunking Strategies Deep Dive

A comprehensive guide to text chunking strategies for RAG pipelines — why we chose 800/100 Recursive Character splitting, and when to upgrade.

---

## Table of Contents

1. [Why Chunking Matters](#why-chunking-matters)
2. [Current Strategy: Recursive Character (800/100)](#current-strategy)
3. [Why 800 Characters?](#why-800-characters)
4. [Why 100 Overlap?](#why-100-overlap)
5. [Strategy 1: Recursive Character Text Splitter (Current)](#strategy-1-recursive-character-text-splitter)
6. [Strategy 2: Semantic Chunking](#strategy-2-semantic-chunking)
7. [Strategy 3: Agentic Chunking (LLM-Based)](#strategy-3-agentic-chunking)
8. [Strategy 4: Document-Structure-Based Chunking](#strategy-4-document-structure-based-chunking)
9. [Strategy 5: Propositionizing (Dense-X Retrieval)](#strategy-5-propositionizing)
10. [Strategy 6: Late Chunking (2024)](#strategy-6-late-chunking)
11. [Strategy 7: Parent-Child Chunking](#strategy-7-parent-child-chunking)
12. [Comparison Table](#comparison-table)
13. [Recommendation & Upgrade Path](#recommendation--upgrade-path)

---

## Why Chunking Matters

Chunking is the single most impactful decision in a RAG pipeline. Bad chunking ruins everything downstream — no amount of reranking or LLM quality can recover from chunks that split topics mid-sentence or merge unrelated content.

```
                    ┌─────────────────────────────────────────────┐
                    │            IMPACT OF CHUNKING                │
                    │                                             │
                    │  Good chunks → Good embeddings              │
                    │             → Good retrieval                │
                    │             → Good answers                  │
                    │                                             │
                    │  Bad chunks  → Noisy embeddings             │
                    │             → Wrong chunks retrieved        │
                    │             → Hallucinated or incomplete    │
                    │               answers                      │
                    │                                             │
                    │  Chunking affects EVERY downstream stage.   │
                    └─────────────────────────────────────────────┘
```

### What Makes a Good Chunk?

```
✅ GOOD CHUNK:
  "The SLA for standard provisioning is 4 hours. Priority requests
   receive a 1-hour SLA. After 2 failed attempts, the system
   escalates to Level 2 support within 30 minutes."

  → Single topic (SLA policy)
  → Complete thought (self-contained, no dangling references)
  → ~200 tokens (within embedding model sweet spot)
  → Answerable: "What is the SLA?" → this chunk has the answer


❌ BAD CHUNK (too small):
  "The SLA is 4 hours."

  → Missing context: SLA for what? Which service?
  → Embedding is vague — many things could have "4 hour" SLAs
  → Query "provisioning SLA" may not match this


❌ BAD CHUNK (too large — mixed topics):
  "The SLA is 4 hours for standard provisioning. Priority
   is 1 hour. The billing cycle runs monthly. Invoices are
   generated on the 1st. Error code ERR-4102 means timeout.
   The backup policy requires daily snapshots. The data
   retention period is 7 years for compliance records..."

  → Mixes SLA + billing + errors + backups + compliance
  → Embedding vector = average of 5 topics = matches nothing well
  → Query "What is the SLA?" retrieves this, but LLM gets distracted
    by billing, errors, backups (noise)


❌ BAD CHUNK (split mid-thought):
  Chunk N:   "...the SLA for standard provisioning is"
  Chunk N+1: "4 hours. Priority requests receive a 1-hour SLA."

  → Chunk N is incomplete — what's the SLA? Cut off.
  → Chunk N+1 starts mid-sentence — no context
  → Query "What is the provisioning SLA?" might find Chunk N
    but the answer (4 hours) is in Chunk N+1
```

---

## Current Strategy

Your pipeline uses **Recursive Character Text Splitter** from LangChain:

```python
# From pipeline.py — SmartChunker class:

class SmartChunker:
    def __init__(self, chunk_size=800, chunk_overlap=100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
```

Additionally, your `SmartChunker` has custom logic to **never split image reference tags**:

```
[IMAGE_REFERENCE: page5_img3]
Summary: This flowchart shows the escalation process with 3 stages...

→ Your SmartChunker ensures this entire block stays in ONE chunk.
  Standard RecursiveCharacterTextSplitter would split it.
```

---

## Why 800 Characters?

```
Your embedding model: nemotron-embed-1b-v2
  Max input tokens: 4096
  Optimal input range: 128-512 tokens (trained on this range)

Character-to-token ratio (English):
  ~4 characters = ~1 token
  800 characters ≈ 200 tokens

Why 200 tokens is the sweet spot:
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  Tokens │ Quality          │ Issue                                       │
│  ───────┼──────────────────┼──────────────────────────────────────────── │
│   50    │ ★★               │ Too short. "SLA is 4 hours." — no context. │
│         │                  │ Embedding is vague. 5× more storage.       │
│  100    │ ★★★              │ Barely enough. 1-2 sentences.              │
│  200    │ ★★★★★            │ Sweet spot. ~1 paragraph. Complete thought.│
│  400    │ ★★★★             │ Good, but starts mixing topics.            │
│  800    │ ★★★              │ Too long. Multiple topics in one chunk.    │
│  2000   │ ★★               │ Entire section. Embedding diluted badly.   │
│                                                                          │
│  800 chars ≈ 200 tokens = sweet spot ✅                                  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘

Research backing:
  • LlamaIndex benchmarks: 256-512 tokens optimal for most embedding models
  • Anthropic's RAG guide: "Chunks of 100-300 tokens perform best"
  • OpenAI cookbook: "Split into paragraphs of ~200 tokens"
  • Your model (nemotron-embed-1b): trained on passages in this range
```

---

## Why 100 Overlap?

```
Overlap = number of characters shared between consecutive chunks.
Purpose: Prevent information loss at chunk boundaries.

Without overlap (0):
  Chunk N:   "The SLA for standard provisioning is 4 hours."
  Chunk N+1: "Priority requests receive a 1-hour SLA."
  
  ❌ If a query is about "standard vs priority SLA," neither chunk
     has both pieces of information.

With 100 char overlap:
  Chunk N:   "The SLA for standard provisioning is 4 hours. Priority requests"
  Chunk N+1: "4 hours. Priority requests receive a 1-hour SLA. After 2 failed"
                        ↑─────── 100 chars shared ──────↑
  
  ✅ Chunk N+1 has context from the end of Chunk N.
  ✅ "standard vs priority SLA" query can match Chunk N+1.

Why 100 and not more?

  Overlap │ % of 800  │ Effect
  ────────┼───────────┼──────────────────────────────────────────
  0       │  0%       │ Hard boundary cuts. Context lost. ❌
  50      │  6.25%    │ ~1 sentence. Minimal but sometimes not enough.
  100     │  12.5%    │ ~2 sentences. Good balance. ✅
  200     │  25%      │ ~1 paragraph. Noticeable duplication.
  400     │  50%      │ Half the chunk is repeated! Storage waste. ❌
  
  100 = 12.5% duplication → negligible storage cost, meaningful context preservation.
  
  At 2 million chunks: 100 overlap adds ~12% more vectors.
  12% more storage is a small price for much better boundary handling.
```

---

## Strategy 1: Recursive Character Text Splitter

**Your current strategy.**

### How It Works

```
Input: A long document string

Separators tried in order: ["\n\n", "\n", " ", ""]

Step 1: Try splitting by double-newline (paragraph breaks)
  "Para 1 about SLA.\n\nPara 2 about errors.\n\nPara 3 about billing."
                       ↑ split                  ↑ split

Step 2: If any paragraph > 800 chars, split by single newline
  "Line 1 of very long paragraph.\nLine 2.\nLine 3.\nLine 4."
                                   ↑ split  ↑ split  ↑ split

Step 3: If any line > 800 chars, split by space
  "This is a very very very long sentence that goes on and on..."
                          ↑ split somewhere here

Step 4: Last resort — split by character (rare, only for 800+ char words)

Result: Chunks of ≤ 800 chars, split at the most natural boundary possible.
```

### Strengths

```
✅ Deterministic — same input always produces same chunks
✅ Fast — O(N) string splitting, no model calls
✅ Free — no API costs
✅ Respects paragraph boundaries first (most natural)
✅ Battle-tested — used in 90%+ of LangChain RAG apps
✅ Your SmartChunker adds image-tag protection on top
```

### Weaknesses

```
❌ No semantic awareness:
   "...the SLA is 4 hours.\n\nIn related news, the SLA for priority is 1 hour."
   → Splits at \n\n even though both paragraphs are about the SAME topic (SLA)
   → These should be ONE chunk, but recursive splitter separates them

❌ Fixed size treats all content equally:
   A 50-char paragraph gets MERGED with the next paragraph to fill 800 chars
   → Two unrelated topics may be forced into one chunk

❌ Character count ≠ token count:
   "ERR-4102-TIMEOUT-PROVISIONING-FAILURE" = 38 chars but ~10 tokens
   "The quick brown fox" = 19 chars but 4 tokens
   → 800 chars can be anywhere from 100-250 tokens depending on content
```

---

## Strategy 2: Semantic Chunking

### How It Works

```
Instead of splitting by character count, split where the TOPIC changes.

Step 1: Split document into individual sentences
  S1: "The SLA for standard provisioning is 4 hours."
  S2: "Priority requests receive a 1-hour SLA."
  S3: "Escalation happens after 2 failed attempts."
  S4: "Error code ERR-4102 indicates a timeout."    ← topic change!
  S5: "ERR-4102 is caused by network congestion."
  S6: "To resolve ERR-4102, restart the service."

Step 2: Embed each sentence
  S1: [0.23, -0.84, 0.13, ...]
  S2: [0.25, -0.81, 0.15, ...]   ← very similar to S1
  S3: [0.20, -0.79, 0.11, ...]   ← similar to S1, S2
  S4: [0.67, 0.45, -0.33, ...]   ← VERY DIFFERENT from S3!
  S5: [0.69, 0.42, -0.31, ...]   ← similar to S4
  S6: [0.65, 0.44, -0.30, ...]   ← similar to S4, S5

Step 3: Compute similarity between consecutive sentences
  sim(S1, S2) = 0.95  → same topic
  sim(S2, S3) = 0.91  → same topic
  sim(S3, S4) = 0.23  → TOPIC CHANGE! ✂️ Split here!
  sim(S4, S5) = 0.94  → same topic
  sim(S5, S6) = 0.92  → same topic

Step 4: Group sentences into chunks
  Chunk 1: [S1, S2, S3]  → "SLA policy" topic
  Chunk 2: [S4, S5, S6]  → "ERR-4102" topic
```

### Implementation

```python
# LangChain experimental:
from langchain_experimental.text_splitter import SemanticChunker

chunker = SemanticChunker(
    embeddings=your_embedding_model,
    breakpoint_threshold_type="percentile",    # or "standard_deviation", "interquartile"
    breakpoint_threshold_amount=75,            # split at 75th percentile of dissimilarity
)

chunks = chunker.split_text(document_text)
```

### Threshold Types Explained

```
Percentile (recommended):
  Calculate similarity between ALL consecutive sentence pairs.
  Split where dissimilarity is in the top 25% (75th percentile).
  Adaptive — adjusts to each document's natural topic boundaries.

Standard Deviation:
  Split where dissimilarity > mean + 1σ
  More aggressive — fewer, larger chunks.

Interquartile:
  Split where dissimilarity > Q3 + 1.5 × IQR
  Most conservative — only splits at extreme topic changes.
```

### Pros & Cons

| Pros | Cons |
|------|------|
| ✅ Chunks are topically coherent | ❌ **Must embed every sentence during ingestion** — slow |
| ✅ No mid-thought splits | ❌ Chunk sizes vary wildly (50 to 3000+ chars) |
| ✅ Better embeddings → better retrieval | ❌ Very short paragraphs may become tiny chunks |
| ✅ Adapts to each document's structure | ❌ Threshold tuning: too high = tiny chunks, too low = huge |
| ✅ No overlap needed (topics don't bleed) | ❌ Extra embedding cost at ingestion time |

### When to Use

```
✅ Documents with diverse topics that change frequently
✅ Unstructured text without clear headings
✅ When retrieval precision is more important than ingestion speed
❌ Not for homogeneous content (same topic throughout)
❌ Not for high-volume ingestion (too slow)
```

---

## Strategy 3: Agentic Chunking (LLM-Based)

### How It Works

```
An LLM reads the document and decides where to split.

Prompt:
  "You are a document chunking expert. Read this text and split it
   into self-contained, topically coherent chunks. Each chunk should:
   - Cover exactly ONE topic or subtopic
   - Be understandable without reading other chunks
   - Have a descriptive title
   
   Text: {document_text}
   
   Output format:
   ---CHUNK: [Title]---
   [chunk content]
   ---CHUNK: [Title]---
   [chunk content]"

Output:
  ---CHUNK: SLA Policy for Provisioning---
  The SLA for standard provisioning is 4 hours. Priority requests
  receive a 1-hour SLA. After 2 failed attempts, the system
  escalates to Level 2 support within 30 minutes.

  ---CHUNK: Error Code ERR-4102---
  ERR-4102 indicates a timeout failure in the provisioning subsystem.
  This error occurs when the backend service does not respond within
  the configured 30-second timeout window. Resolution: restart the
  affected service and check network connectivity.
```

### Pros & Cons

| Pros | Cons |
|------|------|
| ✅ **Best chunk quality** — LLM truly understands content | ❌ **Very slow** — LLM call for every 4000-token window |
| ✅ Adds titles/metadata automatically | ❌ **Expensive** — 10× more tokens consumed than other methods |
| ✅ Handles complex layouts (tables within text, lists) | ❌ **Non-deterministic** — different runs may chunk differently |
| ✅ Can generate summaries per chunk | ❌ Difficult to parallelize (context-dependent) |
| ✅ Perfect for messy, unstructured documents | ❌ LLM rate limits bottleneck ingestion |

### When to Use

```
✅ Small number of critical reference documents (< 50)
✅ Messy documents with no clear structure
✅ When chunk quality is paramount and cost/speed don't matter
❌ Not for bulk ingestion (4000+ documents)
❌ Not when you need deterministic, reproducible chunks
```

---

## Strategy 4: Document-Structure-Based Chunking

### How It Works

```
Uses the document's OWN structure (headings, sections) as chunk boundaries.

Markdown input:
  # Chapter 1: SLA Policy                ← Level 1 boundary
    ## 1.1 Standard SLA                   ← Level 2 boundary
      The SLA is 4 hours...
      (paragraph content)
    ## 1.2 Priority SLA                   ← Level 2 boundary
      Priority SLA is 1 hour...
  # Chapter 2: Error Codes                ← Level 1 boundary
    ## 2.1 ERR-4102                        ← Level 2 boundary
      ERR-4102 is a timeout...

Result:
  Chunk 1: "Standard SLA — The SLA is 4 hours..."
           metadata: {chapter: "SLA Policy", section: "Standard SLA"}
  Chunk 2: "Priority SLA — Priority SLA is 1 hour..."
           metadata: {chapter: "SLA Policy", section: "Priority SLA"}
  Chunk 3: "ERR-4102 — ERR-4102 is a timeout..."
           metadata: {chapter: "Error Codes", section: "ERR-4102"}

If a section > max_size → fall back to recursive splitting WITHIN it.
```

### Implementation

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "chapter"),
        ("##", "section"),
        ("###", "subsection"),
    ]
)
chunks = splitter.split_text(markdown_text)

# Each chunk has rich metadata:
# chunks[0].metadata = {"chapter": "SLA Policy", "section": "Standard SLA"}
```

### For Your Pipeline — Converting PDF to Markdown First

```
Your current pipeline: PDF → PyMuPDF → raw text (no structure)
Upgrade option:        PDF → PyMuPDF4LLM → markdown (preserves headings)

# pip install pymupdf4llm
import pymupdf4llm

md_text = pymupdf4llm.to_markdown("document.pdf")
# Returns markdown with:
#   - Headings preserved as # ## ###
#   - Tables as markdown tables
#   - Bold/italic formatting
#   - Lists preserved
```

### Pros & Cons

| Pros | Cons |
|------|------|
| ✅ Follows author's intended organization | ❌ Only works if document HAS clear headings |
| ✅ Each chunk = one topic by design | ❌ Raw PDF text (PyMuPDF) loses heading structure |
| ✅ Rich metadata (chapter, section, subsection) | ❌ Section sizes vary: 50 chars to 5000 chars |
| ✅ Fast — no model calls | ❌ Requires markdown input (extra conversion step) |
| ✅ Easy to add section titles to chunk metadata | ❌ Some PDFs have NO structure (scanned docs) |

### When to Use

```
✅ Well-structured documents (reports, manuals, policies)
✅ Documents with clear heading hierarchy
✅ When you need section-level metadata in retrieval
❌ Not for flat text with no headings
❌ Not for scanned PDFs (no text layer)
```

---

## Strategy 5: Propositionizing (Dense-X Retrieval)

### How It Works

```
Convert each chunk into atomic factual statements (propositions).
Each proposition = one fact = one embedding.

Original chunk (800 chars):
  "The SLA for standard provisioning is 4 hours. Priority requests
   receive a 1-hour SLA. After 2 failed attempts, the system
   escalates to Level 2 support. The escalation window is 30 minutes.
   Level 2 support operates 24/7."

LLM extracts propositions:
  P1: "The SLA for standard provisioning is 4 hours."
  P2: "Priority provisioning requests receive a 1-hour SLA."
  P3: "Escalation occurs after 2 failed provisioning attempts."
  P4: "The escalation window is 30 minutes."
  P5: "Level 2 support operates 24/7."

Each proposition is embedded independently.
```

### Why This Helps Retrieval

```
Query: "What is the escalation window?"

With 800-char chunks:
  The chunk contains the answer BUT also SLA info, priority info, etc.
  Embedding = average of ALL topics in the chunk → diluted match.
  cosine_similarity ≈ 0.72 (good but not great)

With propositions:
  P4: "The escalation window is 30 minutes."
  Embedding = EXACTLY about escalation windows → precise match.
  cosine_similarity ≈ 0.95 (excellent!)
```

### Pros & Cons

| Pros | Cons |
|------|------|
| ✅ **Maximum retrieval precision** — each embedding = one fact | ❌ **3-5× more vectors** — 550 chunks → 2000+ propositions |
| ✅ No topic dilution in embeddings | ❌ **Slow ingestion** — LLM call for every chunk |
| ✅ Excellent for factual Q&A ("What is X?") | ❌ **Loses context** — "4 hours" without knowing it's about SLA |
| ✅ Works well with any embedding model | ❌ Bad for "explain" or "summarize" queries that need context |
| ✅ Can be combined with parent-child strategy | ❌ Expensive — 3× embedding cost + LLM cost |

### When to Use

```
✅ Factual Q&A systems ("What is the SLA?", "What does ERR-4102 mean?")
✅ When precision matters more than recall
✅ When combined with parent-child (see Strategy 7)
❌ Not for summarization or broad "explain this" queries
❌ Not for high-volume ingestion where cost/speed matter
```

---

## Strategy 6: Late Chunking (2024)

### How It Works

```
Traditional chunking:
  1. Split text into chunks FIRST
  2. Embed each chunk independently
  → Each chunk's embedding has NO knowledge of surrounding chunks

  Document: [Para about SLA] [Para about escalation] [Para about SLA exceptions]
  Chunks:    Chunk 1 (SLA)    Chunk 2 (escalation)    Chunk 3 (SLA exceptions)
  
  Chunk 3 says: "In these cases, the standard timeframe does not apply."
  Embedding for Chunk 3 doesn't know what "standard timeframe" refers to!
  (It was defined in Chunk 1 — but Chunk 3 was embedded independently.)


Late chunking:
  1. Embed the ENTIRE document at once (using long-context embedding model)
     → Every token's embedding has context from the WHOLE document
  2. THEN split into chunks and average the token embeddings per chunk

  Document: [Para about SLA] [Para about escalation] [Para about SLA exceptions]
  
  Full-doc embedding: every token "sees" the entire document context.
  
  Chunk 3 says: "In these cases, the standard timeframe does not apply."
  Chunk 3's embedding KNOWS that "standard timeframe" = "4-hour SLA"
  because it was computed with full document context!
```

### The Difference Visually

```
Traditional:
  Chunk 1: [...] → Embed(Chunk 1) → [v1, v2, v3, ...]     independent
  Chunk 2: [...] → Embed(Chunk 2) → [v1, v2, v3, ...]     independent
  Chunk 3: [...] → Embed(Chunk 3) → [v1, v2, v3, ...]     independent

Late Chunking:
  Full Doc: [...] → Embed(Full Doc) → [t1, t2, ..., t500, ..., t1000, ...]
                                       ↓              ↓              ↓
  Chunk 1:                         avg(t1...t200)                    contextual!
  Chunk 2:                                       avg(t201...t500)   contextual!
  Chunk 3:                                                     avg(t501...t1000) contextual!
```

### Pros & Cons

| Pros | Cons |
|------|------|
| ✅ Each chunk's embedding has full document context | ❌ Requires long-context embedding model (8K+ tokens) |
| ✅ +10-15% recall improvement in benchmarks | ❌ Very new (2024) — limited library support |
| ✅ No overlap needed (context is in the embedding) | ❌ Slower — must process full document per call |
| ✅ Works with any chunk boundaries after embedding | ❌ Only works with specific model architectures |
| ✅ No additional cost vs traditional (same # of embeds) | ❌ Not all models support token-level output |

### Compatible Models

```
Models that support late chunking (as of 2024):
  - Jina Embeddings v2 (jina-embeddings-v2-base-en, 8K context)
  - nomic-embed-text-v1.5 (8K context)
  - Some custom fine-tuned models

Your nemotron-embed-1b → Does NOT support late chunking natively.
You'd need to switch embedding models to use this strategy.
```

### When to Use

```
✅ Documents with many cross-references ("as mentioned above", "the standard timeframe")
✅ Legal/policy documents where context flows across paragraphs
✅ When you can switch to a compatible embedding model
❌ Not if you're locked into a specific embedding model
❌ Not yet mature enough for production-critical systems
```

---

## Strategy 7: Parent-Child Chunking

### How It Works

```
Create TWO levels of chunks from the same document:

Parent chunks (large, ~2000 chars):
  "The SLA for standard provisioning is 4 hours. Priority requests
   receive a 1-hour SLA. After 2 failed attempts, the system
   escalates to Level 2 support. The escalation window is
   30 minutes. Level 2 operates 24/7. If Level 2 cannot resolve
   the issue, it escalates to management within 2 hours..."

Child chunks (small, ~200 chars):
  Child 1: "The SLA for standard provisioning is 4 hours."
  Child 2: "Priority requests receive a 1-hour SLA."
  Child 3: "After 2 failed attempts, escalation to Level 2."
  Child 4: "Escalation window is 30 minutes."
  Child 5: "Level 2 operates 24/7."

Storage:
  Embed and index ONLY the child chunks (for precise retrieval)
  Store parent chunks separately (for context when answering)

Retrieval:
  1. Query matches Child 3 (about escalation)
  2. Look up Child 3's parent → get the FULL 2000-char parent chunk
  3. Send the parent chunk to the LLM (more context for better answers)
```

### Why This Helps

```
Problem with small chunks: Great retrieval but LLM lacks context.
Problem with large chunks: LLM has context but retrieval is imprecise.

Parent-child: Best of both worlds!
  → SEARCH on small chunks (precise matching)
  → ANSWER with large chunks (rich context)

Query: "What happens after escalation?"
  → Child 3 matches precisely (small, focused)
  → Parent chunk gives LLM the full SLA + escalation workflow
  → Answer is both accurate AND contextual
```

### Implementation

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Create parent splitter (large chunks)
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
)

# Create child splitter (small chunks from each parent)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
)

parents = parent_splitter.split_text(document)
for parent in parents:
    children = child_splitter.split_text(parent.text)
    for child in children:
        child.metadata["parent_id"] = parent.id
    # Embed and store children only
    # Store parents in a separate table/dict for lookup
```

### Pros & Cons

| Pros | Cons |
|------|------|
| ✅ Best of both worlds: precise retrieval + rich context | ❌ More complex storage (2 levels) |
| ✅ No need to choose between small and large chunks | ❌ More code to maintain |
| ✅ LLM gets full context without noisy retrieval | ❌ Parent lookup adds a small latency (~5ms) |
| ✅ Works with your existing RecursiveCharacterTextSplitter | ❌ Slightly more storage (parents stored separately) |
| ✅ Easy to implement — just two splitters | |

### When to Use

```
✅ When you need both precision AND context (most RAG pipelines)
✅ When evaluation shows high retrieval precision but low answer quality
✅ Easy upgrade from recursive character splitting
❌ Adds complexity — only if simpler methods aren't good enough
```

---

## Comparison Table

| Strategy | Chunk Quality | Speed | Cost | Complexity | Chunk Size | Overlap Needed? |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Recursive Char** (yours) | ★★★ | ★★★★★ | Free | Low | Fixed | Yes (100) |
| **Semantic** | ★★★★ | ★★★ | Medium | Medium | Variable | No |
| **Agentic (LLM)** | ★★★★★ | ★ | High | High | Variable | No |
| **Structure-Based** | ★★★★ | ★★★★★ | Free | Medium | Variable | Optional |
| **Propositionizing** | ★★★★★ | ★★ | High | High | Very small | No |
| **Late Chunking** | ★★★★★ | ★★★ | Medium | High | Any | No |
| **Parent-Child** | ★★★★ | ★★★★ | Free | Medium | Two levels | Yes (parent) |

### Which Strategy Catches Which Query?

```
Query: "What is the SLA?"
  All strategies: ✅ (simple factoid, any chunking works)

Query: "Compare standard vs priority SLA"
  Recursive:      ⚠️ May split into separate chunks
  Semantic:        ✅ Groups both SLA types together
  Structure:       ✅ If both are under same heading
  Parent-Child:    ✅ Parent chunk has both

Query: "What is ERR-4102?"
  Recursive:      ✅ Exact keyword usually in one chunk
  Propositions:   ✅✅ Each fact about ERR-4102 is its own proposition
  Semantic:        ✅ Groups ERR-4102 sentences together

Query: "Summarize the escalation process"
  Recursive:      ⚠️ Escalation may span multiple chunks
  Parent-Child:   ✅ Parent chunk has the full process
  Agentic:        ✅ LLM creates a coherent "Escalation" chunk

Query: "In those cases, what timeframe applies?"
  Recursive:      ❌ "those cases" reference is lost across chunk boundaries
  Late Chunking:  ✅ Embedding retains cross-chunk context
```

---

## Recommendation & Upgrade Path

```
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║  KEEP Recursive Character (800/100) as your BASE strategy.             ║
║  Your SmartChunker + image-tag protection is already better than       ║
║  vanilla recursive splitting. It works for 90% of use cases.          ║
║                                                                        ║
║  UPGRADE PATH (step by step, based on evaluation scores):             ║
║                                                                        ║
║  Step 1: Add Parent-Child on top of current strategy                   ║
║     Effort: Low (2nd splitter + parent lookup table)                   ║
║     Impact: Better answer quality with context                        ║
║     Try if: Answers are correct but lack context/detail               ║
║                                                                        ║
║  Step 2: Convert PDF → Markdown first (pymupdf4llm)                  ║
║     Then use Structure-Based splitting as primary                     ║
║     Effort: Medium (add pymupdf4llm, new splitting logic)            ║
║     Impact: Heading-aware chunks with rich metadata                   ║
║     Try if: Documents have clear heading structure                    ║
║                                                                        ║
║  Step 3: Semantic Chunking for topic-diverse documents                ║
║     Effort: Medium (needs embedding at chunk time)                    ║
║     Impact: Topic-coherent chunks                                     ║
║     Try if: Context Precision < 0.65 in RAGAS eval                    ║
║                                                                        ║
║  Step 4: Only for critical docs — Agentic chunking                    ║
║     Effort: High (LLM calls at ingestion)                             ║
║     Impact: Highest quality chunks                                    ║
║     Try if: Specific documents keep producing bad answers             ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
```
