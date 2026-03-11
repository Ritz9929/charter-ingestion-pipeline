"""
Main entry point for the Multimodal RAG Ingestion Pipeline.
============================================================
Usage:
    1. Place a PDF named 'sample.pdf' in this directory.
    2. Create a .env file with:
         NVIDIA_VLM_API_KEY=<your_nvidia_vlm_key>
         NVIDIA_EMBED_API_KEY=<your_nvidia_embed_key>
         PG_CONNECTION_STRING=postgresql+psycopg://postgres:mysecretpassword@localhost:5432/postgres
    3. Start the pgvector Docker container:
         docker start local-rag-db
    4. Install deps:  pip install -r requirements.txt
    5. Run:  python main.py
"""

import sys
from pathlib import Path

from pipeline import run_pipeline


def main():
    pdf_path = "sample.pdf"

    # ── Verify the PDF exists ─────────────────────────────────────────────
    if not Path(pdf_path).exists():
        print(f"❌ ERROR: '{pdf_path}' not found in the current directory.")
        print("   Please place a sample PDF here and try again.")
        sys.exit(1)

    # ── Run the full pipeline ─────────────────────────────────────────────
    vectorstore, chunks = run_pipeline(pdf_path)

    # ── Print results ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊  PIPELINE RESULTS")
    print("=" * 60)
    print(f"  Total chunks produced : {len(chunks)}")
    print(f"  Stored in PGVector    : ✅ (persistent — data survives restarts)")

    # Show a preview of the first few chunks
    print("\n── First 3 Chunks (Preview) ──")
    for i, chunk in enumerate(chunks[:3]):
        preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
        print(f"\n  [Chunk {i}] ({len(chunk)} chars)")
        print(f"  {preview}")

    # ── Run a sample similarity search to verify the vector store ─────────
    print("\n── Sample Similarity Search ──")
    query = "What does the document discuss?"
    results = vectorstore.similarity_search(query, k=3)
    print(f"  Query: '{query}'")
    print(f"  Top {len(results)} results:")
    for i, doc in enumerate(results):
        preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
        print(f"    [{i+1}] {preview}")

    print("\n✅ Pipeline complete! Data is stored in PGVector.")
    print("   Run 'python query.py' to search interactively (instant — no re-ingestion needed).")


if __name__ == "__main__":
    main()
