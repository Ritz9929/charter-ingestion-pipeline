"""
Main entry point for the Multimodal RAG Ingestion Pipeline.
============================================================
Usage:
    Single PDF:
        python main.py sample.pdf

    Multiple PDFs:
        python main.py doc1.pdf doc2.pdf doc3.pdf doc4.pdf doc5.pdf

    All PDFs in current directory:
        python main.py *.pdf

    No arguments (defaults to sample.pdf):
        python main.py
"""

import sys
import time
from pathlib import Path

from pipeline import run_pipeline


def main():
    # ── Collect PDF paths from command-line arguments ─────────────────────
    if len(sys.argv) > 1:
        pdf_paths = sys.argv[1:]
    else:
        pdf_paths = ["sample.pdf"]

    # ── Validate all PDFs exist ───────────────────────────────────────────
    valid_pdfs = []
    for pdf_path in pdf_paths:
        if not Path(pdf_path).exists():
            print(f"  ⚠️  Skipping '{pdf_path}' — file not found")
        elif not pdf_path.lower().endswith(".pdf"):
            print(f"  ⚠️  Skipping '{pdf_path}' — not a PDF file")
        else:
            valid_pdfs.append(pdf_path)

    if not valid_pdfs:
        print("❌ ERROR: No valid PDF files found.")
        print("   Usage: python main.py doc1.pdf doc2.pdf ...")
        sys.exit(1)

    print(f"\n📚 Ingestion Queue: {len(valid_pdfs)} PDF(s)")
    for i, pdf in enumerate(valid_pdfs, 1):
        size_mb = Path(pdf).stat().st_size / (1024 * 1024)
        print(f"   {i}. {pdf} ({size_mb:.1f} MB)")

    # ── Ingest each PDF sequentially ──────────────────────────────────────
    total_chunks = 0
    results = []
    overall_start = time.time()

    for i, pdf_path in enumerate(valid_pdfs, 1):
        print(f"\n{'━' * 60}")
        print(f"📄 [{i}/{len(valid_pdfs)}] Ingesting: {pdf_path}")
        print(f"{'━' * 60}")

        start = time.time()
        vectorstore, chunks = run_pipeline(pdf_path)
        elapsed = time.time() - start

        total_chunks += len(chunks)
        results.append({
            "file": pdf_path,
            "chunks": len(chunks),
            "time": elapsed,
        })

        print(f"  ✅ {pdf_path}: {len(chunks)} chunks in {elapsed:.1f}s")

    # ── Print summary ─────────────────────────────────────────────────────
    total_time = time.time() - overall_start
    print(f"\n{'═' * 60}")
    print(f"📊  INGESTION COMPLETE — SUMMARY")
    print(f"{'═' * 60}")
    print(f"  PDFs ingested     : {len(results)}")
    print(f"  Total chunks      : {total_chunks}")
    print(f"  Total time        : {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"  Avg time per PDF  : {total_time/len(results):.1f}s")
    print()

    for r in results:
        print(f"  📄 {r['file']:40s} │ {r['chunks']:>5} chunks │ {r['time']:>6.1f}s")

    print(f"\n  Stored in PGVector : ✅ (incremental — all PDFs coexist)")
    print(f"  Run 'python query.py' to search across ALL documents.\n")

    # ── Run a quick verification search ───────────────────────────────────
    print("── Verification Search ──")
    query = "What does the document discuss?"
    search_results = vectorstore.similarity_search(query, k=3)
    print(f"  Query: '{query}'")
    print(f"  Top {len(search_results)} results:")
    for j, doc in enumerate(search_results):
        source = doc.metadata.get("source_doc", "unknown")
        preview = doc.page_content[:120] + "..." if len(doc.page_content) > 120 else doc.page_content
        print(f"    [{j+1}] [{source}] {preview}")

    print(f"\n✅ All done! {total_chunks} chunks from {len(results)} PDFs are searchable.")


if __name__ == "__main__":
    main()
