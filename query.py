"""
Interactive Query Tool for the Multimodal RAG Pipeline
=======================================================
Connects to the EXISTING PGVector database — no re-ingestion needed.

Features:
  - Cross-encoder reranking for better relevance
  - LLM answer synthesis via NVIDIA NIM (generates a coherent answer from chunks)
  - Metadata-aware retrieval (separates text vs image results)

Usage:
    python query.py
"""

import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder
from pipeline import VectorStoreManager

import truststore
truststore.inject_into_ssl()

load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# NVIDIA NIM API base URL
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


# ═══════════════════════════════════════════════════════════════════════════════
#  RERANKER — Cross-Encoder for better relevance scoring
# ═══════════════════════════════════════════════════════════════════════════════

class Reranker:
    """
    Uses a cross-encoder model to rerank retrieved chunks.
    Cross-encoders process (query, document) pairs jointly, giving
    far more accurate relevance scores than bi-encoder similarity.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"  Loading reranker: {model_name} ...")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: list, top_k: int = 5) -> list:
        """
        Rerank documents by relevance to query.

        Args:
            query:  The user's search query.
            docs:   List of (Document, score) tuples from vector search.
            top_k:  Number of top results to return after reranking.

        Returns:
            List of (Document, rerank_score) tuples, sorted best-first.
        """
        if not docs:
            return []

        # Score each (query, chunk) pair with the cross-encoder
        pairs = [(query, doc.page_content) for doc, _ in docs]
        scores = self.model.predict(pairs)

        # Attach rerank scores and sort (higher = more relevant)
        reranked = [(doc, float(score)) for (doc, _), score in zip(docs, scores)]
        reranked.sort(key=lambda x: x[1], reverse=True)

        return reranked[:top_k]


# ═══════════════════════════════════════════════════════════════════════════════
#  ANSWER SYNTHESIZER — NVIDIA NIM LLM generates a coherent response
# ═══════════════════════════════════════════════════════════════════════════════

class AnswerSynthesizer:
    """
    Takes retrieved chunks + user question and generates a coherent
    answer using an LLM via NVIDIA NIM API.
    """

    SYSTEM_PROMPT = (
        "You are a helpful research assistant. Answer the user's question "
        "based ONLY on the provided context. If the context doesn't contain "
        "enough information, say so honestly. Cite specific details from the "
        "context. Be concise but thorough."
    )

    def __init__(self, model_name: str = "qwen/qwen3.5-122b-a10b"):
        nvidia_key = os.environ.get("NVIDIA_VLM_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        self.client = OpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=nvidia_key,
            timeout=120.0,
        )
        self.model_name = model_name

    def synthesize(self, query: str, docs: list) -> str:
        """Generate an answer from the top retrieved chunks."""
        # Build context from retrieved documents
        context_parts = []
        for i, (doc, score) in enumerate(docs):
            source_type = "IMAGE" if "[IMAGE_REFERENCE" in doc.page_content else "TEXT"
            context_parts.append(
                f"[Source {i+1} ({source_type})]:\n{doc.page_content}"
            )
        context = "\n\n---\n\n".join(context_parts)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"---\n\n"
                    f"Question: {query}\n\n"
                    f"Answer based on the context above:"
                ),
            },
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1000,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Error generating answer: {type(e).__name__}: {e}]"


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — Interactive Query Loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Initialize components ─────────────────────────────────────────────
    print("🔗 Initializing RAG query engine ...")
    print("  Connecting to PGVector database ...")
    store = VectorStoreManager()
    vectorstore = store.connect()

    print("  Loading cross-encoder reranker ...")
    reranker = Reranker()

    print("  Connecting to NVIDIA NIM LLM for answer synthesis ...")
    synthesizer = AnswerSynthesizer()

    print("\n✅ Ready!")
    print("=" * 60)
    print("🔍 ASK ANYTHING ABOUT YOUR DOCUMENT")
    print("   Type your question and press Enter.")
    print("   Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    # ── Interactive query loop ────────────────────────────────────────────
    while True:
        try:
            query = input("\n📝 Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            print("👋 Bye!")
            break

        # Step 1: Retrieve top-20 candidates via vector search
        print("\n  🔎 Searching vector database ...")
        raw_results = vectorstore.similarity_search_with_score(query, k=20)

        if not raw_results:
            print("  No results found.")
            continue

        # Step 2: Rerank with cross-encoder → top 5
        print("  🏆 Reranking with cross-encoder ...")
        reranked = reranker.rerank(query, raw_results, top_k=5)

        # Step 3: Synthesize answer with LLM
        print("  🤖 Generating answer ...\n")
        answer = synthesizer.synthesize(query, reranked)

        # ── Display the answer ────────────────────────────────────────────
        print("═" * 60)
        print("💡 ANSWER")
        print("─" * 60)
        # Word-wrap the answer for clean display
        import textwrap
        for paragraph in answer.split("\n"):
            if paragraph.strip():
                wrapped = textwrap.fill(paragraph.strip(), width=80, initial_indent="  ", subsequent_indent="  ")
                print(wrapped)
            else:
                print()
        print("═" * 60)

        # ── Show the supporting sources ───────────────────────────────────
        print(f"\n── Supporting Sources ({len(reranked)} chunks) ──\n")
        for i, (doc, score) in enumerate(reranked):
            content = doc.page_content
            chunk_idx = doc.metadata.get("chunk_index", "?")
            is_image = "[IMAGE_REFERENCE" in content

            if is_image:
                # Extract filename and summary from the image tag
                import re
                url_match = re.search(r"URL:\s*(\S+)", content)
                summary_match = re.search(r"SUMMARY:\s*(.+?)(?:\]|$)", content)
                filename = url_match.group(1) if url_match else "unknown"
                summary = summary_match.group(1)[:150] + "..." if summary_match else "No summary"
                print(f"  🖼️  Source {i+1}  (score: {score:.2f})  [Chunk #{chunk_idx}]")
                print(f"      Image: {filename}")
                print(f"      Summary: {summary}")
            else:
                # Text source — show clean preview
                preview = " ".join(content.split())[:200]
                if len(content) > 200:
                    preview += " ..."
                print(f"  📄 Source {i+1}  (score: {score:.2f})  [Chunk #{chunk_idx}]")
                print(f"      {preview}")

            if i < len(reranked) - 1:
                print()


if __name__ == "__main__":
    main()
