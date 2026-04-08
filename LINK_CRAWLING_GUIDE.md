# Inter-Document Link Crawling — Implementation Guide

How to integrate recursive document link following with deduplication into the existing ingestion pipeline.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Proposed Architecture](#proposed-architecture)
3. [Ingestion Registry (Deduplication Core)](#ingestion-registry)
4. [Link Extractor — New Component](#link-extractor)
5. [Crawl Orchestrator — New Component](#crawl-orchestrator)
6. [Integration with Existing Pipeline](#integration-with-existing-pipeline)
7. [Walkthrough with Example](#walkthrough-with-example)
8. [Edge Cases & Safeguards](#edge-cases--safeguards)
9. [Database Changes](#database-changes)
10. [File Changes Summary](#file-changes-summary)

---

## Problem Statement

### Current Behavior

```
Documents to ingest: [Doc1, Doc2, Doc3, ..., Doc10]

Current pipeline processes them independently:
  Doc1 → Extract → Summarize → Chunk → Embed → PGVector
  Doc2 → Extract → Summarize → Chunk → Embed → PGVector
  ...
  Doc10 → Extract → Summarize → Chunk → Embed → PGVector

If Doc1 contains a link to Doc3, the link is IGNORED.
```

### Desired Behavior

```
Documents to ingest: [Doc1, Doc2, Doc3, ..., Doc10]

Start with Doc1:
  Doc1 → Extract → finds links to [Doc3, Doc5]
    → Doc3 → Extract → finds link to [Doc7]
      → Doc7 → Extract → finds link to [Doc9]
        → Doc9 → Extract → finds link to [Doc1] → ALREADY INGESTED, SKIP ✅
    → Doc5 → Extract → finds link to [Doc3] → ALREADY INGESTED, SKIP ✅

Continue with Doc2:
  Doc2 → Extract → finds link to [Doc3] → ALREADY INGESTED, SKIP ✅

Continue with Doc4 (Doc3, Doc5, Doc7, Doc9 already done):
  Doc4 → Extract → no links

Continue with Doc6, Doc8, Doc10:
  → Only ingest what hasn't been ingested yet
```

---

## Proposed Architecture

```
                    ┌─────────────────────────────┐
                    │  Batch Ingestion Entry Point  │
                    │  (new: batch_ingest.py)       │
                    │                               │
                    │  Input: [Doc1..Doc10]          │
                    │  Maintains: ingestion_queue    │
                    └──────────────┬────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │  Crawl Orchestrator (NEW)     │
                    │                               │
                    │  1. Pick next doc from queue  │
                    │  2. Check registry: ingested? │
                    │     → Yes: SKIP               │
                    │     → No: proceed ↓           │
                    │  3. Run existing pipeline      │
                    │  4. Extract links from doc     │
                    │  5. Add linked docs to queue   │
                    │  6. Mark doc as INGESTED       │
                    │  7. Go to step 1              │
                    └──────────────┬────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                              │
                    ▼                              ▼
        ┌─────────────────────┐      ┌──────────────────────────┐
        │  Existing Pipeline   │      │  Ingestion Registry (NEW) │
        │  (pipeline.py)       │      │  (PostgreSQL table)       │
        │                      │      │                           │
        │  extract()           │      │  doc_id | status | links  │
        │  summarize()         │      │  ────────────────────── │
        │  reassemble()        │      │  doc1   | DONE   | [3,5] │
        │  chunk()             │      │  doc3   | DONE   | [7]   │
        │  embed + store()     │      │  doc5   | DONE   | [3]   │
        │                      │      │  doc7   | DONE   | [9]   │
        │  UNCHANGED ✅        │      │  doc9   | DONE   | [1]   │
        └─────────────────────┘      └──────────────────────────┘
                                                ▲
                                                │
                                      ┌─────────┴──────────┐
                                      │  Link Extractor     │
                                      │  (NEW component)    │
                                      │                     │
                                      │  PDF: get_links()   │
                                      │  DOCX: extract rels │
                                      │  CSV: scan for URLs │
                                      │  PPT: extract links │
                                      └────────────────────┘
```

---

## Ingestion Registry

The **core** of deduplication. A PostgreSQL table that tracks what has been ingested.

### Table Schema

```sql
CREATE TABLE IF NOT EXISTS ingestion_registry (
    doc_id          TEXT PRIMARY KEY,     -- unique identifier for the document
    doc_path        TEXT NOT NULL,        -- original file path or URL
    doc_hash        TEXT NOT NULL,        -- SHA-256 hash of file content
    status          TEXT NOT NULL,        -- 'QUEUED' | 'IN_PROGRESS' | 'DONE' | 'FAILED'
    chunk_count     INTEGER DEFAULT 0,   -- number of chunks stored in PGVector
    linked_docs     TEXT[] DEFAULT '{}',  -- list of doc_ids this doc links to
    discovered_by   TEXT,                 -- which doc's link led to this one (NULL if direct)
    ingested_at     TIMESTAMP,           -- when ingestion completed
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Index for fast dedup lookups
CREATE INDEX idx_registry_hash ON ingestion_registry(doc_hash);
CREATE INDEX idx_registry_status ON ingestion_registry(status);
```

### How doc_id is Generated

```
For local files:
  doc_id = normalize(filename)
  Example: "Charter_Agreement_v2.pdf" → "charter_agreement_v2.pdf"

For URLs:
  doc_id = hash(canonical_url)
  Example: "https://company.com/docs/sla.pdf" → "url_a3f8c2d1"

For content dedup (same file, different name):
  doc_hash = SHA-256 of file bytes
  Before ingesting, check: is this hash already in registry?
  If yes → same content, different name → SKIP
```

### Registry Operations

```python
# Pseudocode for registry methods:

class IngestionRegistry:
    
    def is_ingested(self, doc_id: str) -> bool:
        """Check if document was already ingested."""
        # SELECT status FROM ingestion_registry WHERE doc_id = ?
        # Return True if status == 'DONE'
    
    def is_content_ingested(self, file_bytes: bytes) -> bool:
        """Check if identical content was already ingested (different filename)."""
        # doc_hash = sha256(file_bytes)
        # SELECT 1 FROM ingestion_registry WHERE doc_hash = ? AND status = 'DONE'
    
    def mark_queued(self, doc_id: str, doc_path: str, doc_hash: str, discovered_by: str = None):
        """Register a document as queued for ingestion."""
        # INSERT INTO ingestion_registry (doc_id, doc_path, doc_hash, status, discovered_by)
        # VALUES (?, ?, ?, 'QUEUED', ?)
        # ON CONFLICT (doc_id) DO NOTHING  ← skip if already registered
    
    def mark_in_progress(self, doc_id: str):
        """Mark document as currently being ingested."""
        # UPDATE ingestion_registry SET status = 'IN_PROGRESS' WHERE doc_id = ?
    
    def mark_done(self, doc_id: str, chunk_count: int, linked_docs: list[str]):
        """Mark document as successfully ingested."""
        # UPDATE ingestion_registry 
        # SET status = 'DONE', chunk_count = ?, linked_docs = ?, ingested_at = NOW()
        # WHERE doc_id = ?
    
    def mark_failed(self, doc_id: str, error: str):
        """Mark document as failed."""
        # UPDATE SET status = 'FAILED' WHERE doc_id = ?
```

---

## Link Extractor — New Component

A new class that extracts document links from different file types.

### For PDFs (PyMuPDF)

```python
# Pseudocode:

class LinkExtractor:

    def extract_links_from_pdf(self, pdf_path: str) -> list[dict]:
        """Extract all external links from a PDF."""
        doc = fitz.open(pdf_path)
        links = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            for link in page.get_links():
                if link["kind"] == 2:  # URI link (external)
                    links.append({
                        "url": link["uri"],
                        "page": page_num,
                        "type": "uri",
                    })
                elif link["kind"] == 1:  # Internal page link
                    pass  # Skip — same document
        
        doc.close()
        return links
```

### Types of Links to Handle

```
Links found in documents can point to:

1. ANOTHER DOCUMENT in the same batch:
   "See appendix_b.pdf for details"
   → Link text or URL contains a filename that matches another doc in the batch
   → Action: ADD to ingestion queue

2. AN EXTERNAL URL pointing to a document:
   "https://company.sharepoint.com/docs/sla_policy.pdf"
   → A downloadable document (PDF, DOCX, CSV, PPT)
   → Action: DOWNLOAD to temp dir, then ADD to ingestion queue

3. AN EXTERNAL URL pointing to a webpage:
   "https://company.com/policies/data-retention"
   → An HTML page, not a document file
   → Action: CRAWL with requests + BeautifulSoup, extract text,
             create a synthetic "document" from the page content

4. AN INTERNAL LINK (same document):
   "See page 12" or anchor link within the PDF
   → Action: SKIP (already ingesting this document)

5. AN IRRELEVANT LINK:
   "mailto:support@company.com" or "tel:+1234567890"
   → Action: SKIP
```

### Link Resolution Logic

```
For each extracted link:

┌────────────────────────────────┐
│  Raw link from document         │
│  "https://sharepoint/sla.pdf"   │
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│  Is it mailto/tel/javascript?   │──── Yes ──→ SKIP
└────────────┬───────────────────┘
             │ No
             ▼
┌────────────────────────────────┐
│  Does URL end in                │
│  .pdf/.docx/.csv/.pptx?        │──── Yes ──→ DOWNLOAD + QUEUE as document
└────────────┬───────────────────┘
             │ No
             ▼
┌────────────────────────────────┐
│  Does filename match another    │
│  doc in the ingestion batch?    │──── Yes ──→ QUEUE that batch document
└────────────┬───────────────────┘
             │ No
             ▼
┌────────────────────────────────┐
│  Is it an HTTP(S) URL?          │──── Yes ──→ CRAWL as webpage, extract text
└────────────┬───────────────────┘
             │ No
             ▼
           SKIP (unrecognized link type)
```

---

## Crawl Orchestrator — New Component

The **brain** of the system. Manages the ingestion queue with BFS (breadth-first) traversal.

### Algorithm

```
Input: initial_docs = [Doc1, Doc2, Doc3, ..., Doc10]

ALGORITHM: BFS with Deduplication

  ingestion_queue = deque(initial_docs)        # FIFO queue
  registry = IngestionRegistry(db_connection)

  # Pre-register all initial docs as QUEUED
  for doc in initial_docs:
      registry.mark_queued(doc.id, doc.path, hash(doc))

  while ingestion_queue is not empty:
      current_doc = ingestion_queue.popleft()
      
      # ── DEDUP CHECK ──────────────────────────────────
      if registry.is_ingested(current_doc.id):
          log(f"SKIP {current_doc.id} — already ingested")
          continue
      
      if registry.is_content_ingested(current_doc.bytes):
          log(f"SKIP {current_doc.id} — identical content exists under different name")
          continue
      
      # ── INGEST ───────────────────────────────────────
      registry.mark_in_progress(current_doc.id)
      
      try:
          # Run the EXISTING pipeline (unchanged!)
          vectorstore, chunks = run_pipeline(current_doc.path, ...)
          
          # ── EXTRACT LINKS (NEW STEP) ─────────────────
          links = LinkExtractor.extract(current_doc.path)
          discovered_doc_ids = []
          
          for link in links:
              resolved_doc = resolve_link(link, initial_docs)
              
              if resolved_doc is None:
                  continue  # unresolvable link
              
              if not registry.is_ingested(resolved_doc.id):
                  registry.mark_queued(resolved_doc.id, resolved_doc.path,
                                       hash(resolved_doc), discovered_by=current_doc.id)
                  ingestion_queue.append(resolved_doc)
                  discovered_doc_ids.append(resolved_doc.id)
                  log(f"  DISCOVERED: {resolved_doc.id} (via {current_doc.id})")
              else:
                  log(f"  SKIP LINK: {resolved_doc.id} (already ingested)")
          
          # ── MARK DONE ───────────────────────────────
          registry.mark_done(current_doc.id, len(chunks), discovered_doc_ids)
          
      except Exception as e:
          registry.mark_failed(current_doc.id, str(e))
          log(f"FAILED: {current_doc.id} — {e}")
  
  log("ALL DOCUMENTS INGESTED ✅")
```

### BFS vs DFS — Why BFS?

```
DFS (depth-first): Doc1 → Doc3 → Doc7 → Doc9 → ... (goes deep before moving on)
  Problem: If Doc1 links to 50 docs and each links to 50 more,
           you go 10 levels deep before even starting Doc2.

BFS (breadth-first): Doc1 → Doc2 → Doc3 → Doc5 → Doc7 → Doc9 → ...
  Advantage: Processes all initial documents first, then discovered ones.
  Your initial batch (Doc1-Doc10) completes before chasing deep links.

✅ Recommendation: Use BFS (FIFO queue).
```

---

## Integration with Existing Pipeline

### What Changes and What Doesn't

```
pipeline.py:
  PDFExtractor        → UNCHANGED ✅
  ImageSummarizer     → UNCHANGED ✅
  DocumentReassembler → UNCHANGED ✅
  SmartChunker        → UNCHANGED ✅
  VectorStoreManager  → UNCHANGED ✅
  run_pipeline()      → UNCHANGED ✅  (called as-is by the orchestrator)

query.py:
  HybridSearcher      → UNCHANGED ✅
  Reranker             → UNCHANGED ✅
  AnswerSynthesizer    → UNCHANGED ✅

NEW FILES:
  link_extractor.py    → NEW: Extracts links from PDF/DOCX/CSV/PPT
  crawl_orchestrator.py→ NEW: BFS queue + dedup logic
  ingestion_registry.py→ NEW: PostgreSQL registry operations
  batch_ingest.py      → NEW: Entry point (replaces main.py for batch runs)
```

### New Entry Point: batch_ingest.py

```python
# Pseudocode for the new entry point:

def batch_ingest(document_paths: list[str]):
    """
    Ingest multiple documents with link crawling and deduplication.
    
    Usage:
      python batch_ingest.py doc1.pdf doc2.pdf doc3.pdf ... doc10.pdf
      python batch_ingest.py ./documents_folder/
    """
    registry = IngestionRegistry(connection_string)
    orchestrator = CrawlOrchestrator(registry)
    
    # Queue all initial documents
    for path in document_paths:
        orchestrator.add_to_queue(path)
    
    # Process queue (BFS with dedup + link following)
    results = orchestrator.process_all()
    
    # Print summary
    print(f"Ingested: {results.ingested_count} documents")
    print(f"Skipped:  {results.skipped_count} (already ingested)")
    print(f"Failed:   {results.failed_count}")
    print(f"Links discovered: {results.links_found}")
```

---

## Walkthrough with Example

### Your Exact Scenario

```
Input batch: [Doc1, Doc2, Doc3, Doc4, Doc5, Doc6, Doc7, Doc8, Doc9, Doc10]

Step-by-step execution:

═══════════════════════════════════════════════════════════════
ITERATION 1 — Pick Doc1 from queue
═══════════════════════════════════════════════════════════════

  Queue:    [Doc1, Doc2, Doc3, Doc4, Doc5, Doc6, Doc7, Doc8, Doc9, Doc10]
  Registry: (empty)

  1. is_ingested("doc1")? → NO
  2. run_pipeline("doc1.pdf") → 42 chunks stored ✅
  3. extract_links("doc1.pdf") → found links to [Doc3, Doc5]
  4. is_ingested("doc3")? → NO → add to queue
  5. is_ingested("doc5")? → NO → add to queue
  6. mark_done("doc1", chunks=42, links=["doc3", "doc5"])

  Queue:    [Doc2, Doc3, Doc4, Doc5, Doc6, Doc7, Doc8, Doc9, Doc10, Doc3*, Doc5*]
                                                                       ↑ duplicates!
  But Doc3 and Doc5 are ALREADY in the queue from the initial batch,
  so mark_queued() does ON CONFLICT DO NOTHING → no duplicate entry.
  
  Effective queue: [Doc2, Doc3, Doc4, Doc5, Doc6, Doc7, Doc8, Doc9, Doc10]
  Registry: {doc1: DONE}

═══════════════════════════════════════════════════════════════
ITERATION 2 — Pick Doc2 from queue
═══════════════════════════════════════════════════════════════

  1. is_ingested("doc2")? → NO
  2. run_pipeline("doc2.pdf") → 38 chunks stored ✅
  3. extract_links("doc2.pdf") → no links found
  4. mark_done("doc2", chunks=38, links=[])

  Registry: {doc1: DONE, doc2: DONE}

═══════════════════════════════════════════════════════════════
ITERATION 3 — Pick Doc3 from queue
═══════════════════════════════════════════════════════════════

  1. is_ingested("doc3")? → NO
  2. run_pipeline("doc3.pdf") → 55 chunks stored ✅
  3. extract_links("doc3.pdf") → found link to [Doc7]
  4. is_ingested("doc7")? → NO → add to queue
  5. mark_done("doc3", chunks=55, links=["doc7"])

  Registry: {doc1: DONE, doc2: DONE, doc3: DONE}

═══════════════════════════════════════════════════════════════
ITERATIONS 4-6 — Doc4, Doc5, Doc6
═══════════════════════════════════════════════════════════════

  Doc4: Ingested (no links)
  Doc5: Ingested, found link to [Doc3] → ALREADY INGESTED → SKIP ✅
  Doc6: Ingested (no links)

  Registry: {doc1-doc6: all DONE}

═══════════════════════════════════════════════════════════════
ITERATION 7 — Pick Doc7 from queue
═══════════════════════════════════════════════════════════════

  1. is_ingested("doc7")? → NO
  2. run_pipeline("doc7.pdf") → 29 chunks stored ✅
  3. extract_links("doc7.pdf") → found link to [Doc9]
  4. is_ingested("doc9")? → NO → already in queue from initial batch
  5. mark_done("doc7", chunks=29, links=["doc9"])

  Registry: {doc1-doc7: all DONE}

═══════════════════════════════════════════════════════════════
ITERATIONS 8-10 — Doc8, Doc9, Doc10
═══════════════════════════════════════════════════════════════

  Doc8: Ingested (no links)
  Doc9: Ingested, found link to [Doc1] → ALREADY INGESTED → SKIP ✅
  Doc10: Ingested (no links)

  Registry: {doc1-doc10: all DONE} ✅

═══════════════════════════════════════════════════════════════
QUEUE IS EMPTY — ALL DONE!
═══════════════════════════════════════════════════════════════

Final summary:
  Documents ingested: 10
  Documents skipped:  0 (all 10 were new)
  Links found:        Doc1→[3,5], Doc3→[7], Doc5→[3], Doc7→[9], Doc9→[1]
  Links that were already ingested: 3 (Doc3 via Doc5, Doc9 via initial, Doc1 via Doc9)
  Circular reference detected: Doc1→Doc3→Doc7→Doc9→Doc1 (handled correctly ✅)
```

### Second Run (Adding 3 New Documents)

```
Input batch: [Doc1, Doc2, Doc3, Doc11, Doc12, Doc13]

  Doc1:  is_ingested? → YES → SKIP ✅
  Doc2:  is_ingested? → YES → SKIP ✅
  Doc3:  is_ingested? → YES → SKIP ✅
  Doc11: is_ingested? → NO  → INGEST → found link to [Doc3] → SKIP (already done)
  Doc12: is_ingested? → NO  → INGEST
  Doc13: is_ingested? → NO  → INGEST

  Only 3 new docs were ingested! ✅
```

---

## Edge Cases & Safeguards

### 1. Circular References (Doc1 → Doc3 → Doc7 → Doc1)

```
Handled automatically by the registry.
When Doc9 links back to Doc1, the registry check:
  is_ingested("doc1") → YES → SKIP

No special cycle detection needed — the registry IS the cycle breaker.
```

### 2. Maximum Crawl Depth

```
Even though circular references are safe, a long chain can slow things down:
  Doc1 → URL_A → URL_B → URL_C → ... → URL_Z (26 hops!)

Safeguard: Add a max_depth parameter:
  - Initial batch documents: depth = 0
  - Documents discovered via links: depth = parent_depth + 1
  - Skip if depth > MAX_DEPTH (recommended: 2-3)

  max_depth = 2:
    Doc1 (depth=0) → Doc3 (depth=1) → Doc7 (depth=2) → Doc9 (depth=3) → SKIP ⛔
```

### 3. Maximum Queue Size

```
Safeguard: Cap the queue at ~100 documents per run.
If a single document links to 500 external URLs, you don't want to crawl them all.

Config:
  MAX_DISCOVERED_LINKS_PER_DOC = 20  # Max links to follow from one document
  MAX_TOTAL_QUEUE_SIZE = 100         # Total cap across entire run
```

### 4. External URL Failures

```
Link points to: "https://internal.company.com/deprecated-doc.pdf"
  → Server returns 404 or times out

Handling:
  1. Retry 2× with exponential backoff (2s, 4s)
  2. If still failing → mark_failed(doc_id, "HTTP 404")
  3. Continue with next document in queue
  4. Do NOT block the rest of ingestion
```

### 5. Same Content, Different Filename

```
Doc1.pdf links to "SLA_Policy_v2.pdf"
Doc4.pdf links to "SLA_v2_final.pdf"
Both files are identical (same content, same hash)

Handling:
  1. When downloading SLA_Policy_v2.pdf → hash = "abc123"
  2. Ingest it, registry: {doc_hash: "abc123", status: DONE}
  3. When downloading SLA_v2_final.pdf → hash = "abc123"
  4. is_content_ingested("abc123") → YES → SKIP ✅
```

### 6. Re-ingesting Updated Documents

```
Doc3 was ingested last week. Today, Doc3 has been updated (new content).

Option A (simple): Force re-ingest flag
  python batch_ingest.py --force doc3.pdf
  → Bypasses registry check, re-ingests, updates chunk count

Option B (smart): Content-hash based
  Old hash: "abc123" vs New hash: "def456" → different → re-ingest
  1. Delete old chunks from PGVector for this doc_id
  2. Re-run pipeline
  3. Update registry with new hash, new chunk count
```

---

## Database Changes

### New Table in Existing PostgreSQL

```sql
-- Run this ONCE on your existing PGVector database:

CREATE TABLE IF NOT EXISTS ingestion_registry (
    doc_id          TEXT PRIMARY KEY,
    doc_path        TEXT NOT NULL,
    doc_hash        TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'QUEUED',
    chunk_count     INTEGER DEFAULT 0,
    linked_docs     TEXT[] DEFAULT '{}',
    discovered_by   TEXT,
    crawl_depth     INTEGER DEFAULT 0,
    error_message   TEXT,
    ingested_at     TIMESTAMP,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_registry_hash ON ingestion_registry(doc_hash);
CREATE INDEX IF NOT EXISTS idx_registry_status ON ingestion_registry(status);
```

This table coexists with the existing `langchain_pg_embedding` table (your vectors) and `langchain_pg_collection` (your collections). No changes to existing tables.

---

## File Changes Summary

```
EXISTING FILES — NO CHANGES:
  pipeline.py           → Untouched (run_pipeline() called as-is)
  query.py              → Untouched (search/rerank/synthesis unchanged)
  main.py               → Untouched (still works for single-PDF ingestion)

NEW FILES TO CREATE:
  ┌─────────────────────────┬──────────────────────────────────────────────┐
  │ File                    │ Purpose                                      │
  ├─────────────────────────┼──────────────────────────────────────────────┤
  │ ingestion_registry.py   │ IngestionRegistry class — PostgreSQL CRUD    │
  │                         │ for tracking ingested documents               │
  │                         │                                              │
  │ link_extractor.py       │ LinkExtractor class — extracts links from    │
  │                         │ PDF (get_links), DOCX, CSV, PPT              │
  │                         │ Resolves links to doc_ids                     │
  │                         │                                              │
  │ crawl_orchestrator.py   │ CrawlOrchestrator class — BFS queue, dedup  │
  │                         │ logic, calls run_pipeline() + LinkExtractor  │
  │                         │ Handles depth limits, queue caps, retries    │
  │                         │                                              │
  │ batch_ingest.py         │ CLI entry point for batch ingestion          │
  │                         │ python batch_ingest.py doc1.pdf doc2.pdf ... │
  └─────────────────────────┴──────────────────────────────────────────────┘

USAGE:
  # Single document (existing — unchanged):
  python main.py

  # Batch with link crawling (new):
  python batch_ingest.py doc1.pdf doc2.pdf doc3.pdf ... doc10.pdf

  # Batch from folder:
  python batch_ingest.py ./documents/

  # Force re-ingest specific docs:
  python batch_ingest.py --force doc3.pdf doc5.pdf

  # Set max crawl depth:
  python batch_ingest.py --max-depth 2 ./documents/
```
