"""
Multimodal RAG Ingestion Pipeline
==================================
A modular pipeline that:
  1. Extracts text and images from a PDF (PyMuPDF)
     - Embedded raster images (photos, logos)
     - Vector-drawn visuals (charts, graphs, tables) via page rendering
  2. Summarizes images via a Vision Language Model (NVIDIA NIM API)
  3. Reassembles the document with image reference tags
  4. Chunks the text intelligently (never splitting image tags)
  5. Stores chunks in a PGVector database (persistent)
"""

import os
import re
import json
import base64
import time
import logging
from pathlib import Path
from io import BytesIO
from dataclasses import dataclass, field

import fitz  # PyMuPDF
from PIL import Image
from dotenv import load_dotenv

from openai import OpenAI
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector

# NVIDIA NIM API config
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

# ── Fix SSL certificate issues (common in corporate/enterprise networks) ──────
# truststore makes Python use the OS system certificate store (Windows/macOS/Linux)
# This properly handles corporate proxy CA certificates that aren't in certifi.
import truststore
truststore.inject_into_ssl()

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)s │ %(message)s")
logger = logging.getLogger(__name__)

# Suppress noisy HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ExtractedImage:
    """Represents a single image extracted from the PDF."""
    filename: str          # e.g. "page3_img1.png"
    filepath: str          # full path inside mock_s3_storage
    page_number: int       # 0-indexed page the image came from
    position_index: int    # insertion order on that page
    source_type: str = "embedded"  # "embedded" or "rendered" (for vector graphics)
    summary: str = ""      # filled in by the summarizer


@dataclass
class ExtractionResult:
    """Complete extraction output for one PDF."""
    page_texts: dict = field(default_factory=dict)   # {page_num: raw_text}
    images: list = field(default_factory=list)        # list[ExtractedImage]


# ═══════════════════════════════════════════════════════════════════════════════
#  1. PDF EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

class PDFExtractor:
    """
    Uses PyMuPDF (fitz) to parse a PDF file.

    Extraction methods:
      - Embedded raster images via get_images() (photos, logos, etc.)
      - Vector-drawn visuals (charts, graphs, tables) by detecting pages
        with drawing commands and rendering them as page screenshots.
    """

    # Minimum number of vector drawing operations on a page to consider it
    # as having a chart/graph/table worth rendering.
    MIN_DRAWINGS_THRESHOLD = 10

    def __init__(self, output_dir: str = "./mock_s3_storage", render_dpi: int = 200):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.render_dpi = render_dpi

    def _has_vector_graphics(self, page) -> bool:
        """Check if a page has significant vector drawings (charts/graphs/tables)."""
        try:
            drawings = page.get_drawings()
            return len(drawings) >= self.MIN_DRAWINGS_THRESHOLD
        except Exception:
            return False

    def _render_page_as_image(self, page, page_num: int, img_index: int) -> ExtractedImage:
        """Render an entire page as a PNG image (captures vector graphics)."""
        zoom = self.render_dpi / 72  # 72 DPI is the PDF default
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        filename = f"page{page_num}_rendered.png"
        filepath = self.output_dir / filename
        pix.save(str(filepath))

        return ExtractedImage(
            filename=filename,
            filepath=str(filepath),
            page_number=page_num,
            position_index=img_index,
            source_type="rendered",
        )

    def extract(self, pdf_path: str) -> ExtractionResult:
        """Parse the PDF and return text + image metadata."""
        doc = fitz.open(pdf_path)
        result = ExtractionResult()

        logger.info(f"Opened PDF: {pdf_path} ({len(doc)} pages)")

        for page_num in range(len(doc)):
            page = doc[page_num]

            # ── Extract text ──────────────────────────────────────────────
            result.page_texts[page_num] = page.get_text("text")

            # ── Extract embedded raster images ────────────────────────────
            image_list = page.get_images(full=True)
            has_embedded_images = False
            seen_xrefs = set()  # Deduplicate images by xref
            saved_img_count = 0
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]

                # Skip duplicate image references on the same page
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                try:
                    base_image = doc.extract_image(xref)

                    if base_image is None:
                        continue

                    image_bytes = base_image["image"]

                    # Skip tiny images (< 1KB — likely icons or artifacts)
                    if len(image_bytes) < 1024:
                        continue

                    image_ext = base_image.get("ext", "png")
                    filename = f"page{page_num}_img{saved_img_count}.{image_ext}"
                    filepath = self.output_dir / filename

                    # Save image to disk (simulating S3 upload)
                    with open(filepath, "wb") as f:
                        f.write(image_bytes)

                    result.images.append(
                        ExtractedImage(
                            filename=filename,
                            filepath=str(filepath),
                            page_number=page_num,
                            position_index=saved_img_count,
                            source_type="embedded",
                        )
                    )
                    has_embedded_images = True
                    saved_img_count += 1
                    logger.info(f"  Saved embedded image: {filename}")

                except Exception as e:
                    logger.warning(f"  Skipping image xref {xref} on page {page_num}: {e}")

            # ── Render pages with vector graphics (charts/graphs/tables) ──
            # Only render if the page has drawings AND doesn't already have
            # many embedded images (to avoid duplicating content).
            if self._has_vector_graphics(page) and not has_embedded_images:
                next_index = len(image_list)
                rendered = self._render_page_as_image(page, page_num, next_index)
                result.images.append(rendered)
                logger.info(f"  Rendered page {page_num} as image (vector graphics detected)")

        doc.close()

        embedded_count = sum(1 for img in result.images if img.source_type == "embedded")
        rendered_count = sum(1 for img in result.images if img.source_type == "rendered")
        logger.info(
            f"Extraction complete — {len(result.page_texts)} pages, "
            f"{embedded_count} embedded images, {rendered_count} rendered pages"
        )
        return result


# ═══════════════════════════════════════════════════════════════════════════════
#  2. IMAGE SUMMARIZER (NVIDIA NIM API — OpenAI-compatible)
# ═══════════════════════════════════════════════════════════════════════════════

class ImageSummarizer:
    """
    Sends each extracted image to a Vision Language Model via the
    NVIDIA NIM Inference API and returns a detailed, factual summary.

    Uses NVIDIA NIM's OpenAI-compatible API — 40 RPM free tier.
    The model is swappable via the constructor.
    """

    DEFAULT_PROMPT = (
        "You are an expert data analyst and document parsing specialist. "
        "Your task is to analyze the provided image and extract its contents "
        "into a dense, highly detailed text summary.\n\n"
        "This summary will be embedded into a Vector Database for a "
        "Retrieval-Augmented Generation (RAG) system. It is critical that "
        "your output contains all exact keywords, numbers, and relationships "
        "present in the image so that semantic search can accurately find it later.\n\n"
        "Please analyze the image and output your response strictly using "
        "the following Markdown structure:\n\n"
        "### 1. Image Type & Core Subject\n"
        "[State the type of image: e.g., Bar Chart, Line Graph, Flowchart, "
        "Architecture Diagram, Financial Table, or Photograph. State the main "
        "title or core subject in one sentence.]\n\n"
        "### 2. Explicit Text & Labels\n"
        "[Extract and list all literal text visible in the image. This includes:\n"
        "- Main titles and subtitles\n"
        "- X and Y axis labels (including units of measurement)\n"
        "- Legend items\n"
        "- Node labels in flowcharts\n"
        "- Column and row headers in tables]\n\n"
        "### 3. Data & Relationships (The \"Meat\")\n"
        "[Translate the visual data into descriptive text.\n"
        "- For charts: State the specific values, trends, peaks, and valleys "
        "(e.g., \"Revenue peaked in Q3 2023 at $4.5M, a 15% increase from Q2\").\n"
        "- For flowcharts/diagrams: Describe the step-by-step flow, connections, "
        "and logic (e.g., \"The API Gateway routes traffic to the Auth Service, "
        "which then queries the Redis Cache\").\n"
        "- For tables: Summarize the key data points or anomalies; if the table "
        "is small, represent it entirely in Markdown format.]\n\n"
        "### 4. Semantic Keywords\n"
        "[Provide a comma-separated list of 5-10 highly specific keywords, "
        "jargon, or entities found in the image to aid in vector similarity matching.]"
    )

    MAX_RETRIES = 5          # Number of retry attempts on errors
    INITIAL_BACKOFF = 5.0    # Initial wait (seconds) before first retry
    DELAY_BETWEEN = 1.5      # Delay (seconds) between consecutive API calls (40 RPM = 1.5s/req)

    def __init__(self, model_name: str = "qwen/qwen3.5-122b-a10b"):
        """
        Args:
            model_name: NVIDIA NIM model ID for a vision-language model.
                        Default uses Qwen 3.5 VLM (400B MoE).
        """
        nvidia_key = os.environ.get("NVIDIA_VLM_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        if not nvidia_key:
            raise ValueError(
                "NVIDIA_VLM_API_KEY (or NVIDIA_API_KEY) not found in environment. "
                "Set it in .env"
            )

        self.client = OpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=nvidia_key,
            timeout=120.0,  # 120s timeout to prevent indefinite hangs
        )
        self.model_name = model_name
        logger.info(f"ImageSummarizer initialized with model: {model_name} (NVIDIA NIM)")

    def _image_to_base64_url(self, filepath: str, max_size: int = 1024) -> str:
        """Convert an image file to a base64 data URL, resizing if too large."""
        img = Image.open(filepath)

        # Resize large images to speed up API calls
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.LANCZOS)
            logger.info(f"    Resized image to {img.size[0]}x{img.size[1]}")

        # Convert to JPEG for smaller payload (unless PNG with transparency)
        buffer = BytesIO()
        if img.mode == "RGBA":
            img.save(buffer, format="PNG")
            mime_type = "image/png"
        else:
            img = img.convert("RGB")
            img.save(buffer, format="JPEG", quality=85)
            mime_type = "image/jpeg"

        image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:{mime_type};base64,{image_data}"

    def summarize(self, image: ExtractedImage) -> str:
        """Send an image to the VLM and return the summary string."""
        logger.info(f"  Summarizing {image.filename} ...")

        image_url = self._image_to_base64_url(image.filepath)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": self.DEFAULT_PROMPT},
                ],
            }
        ]

        # Retry with exponential backoff for transient errors
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=1500,
                    temperature=0.2,
                    timeout=120.0,
                )
                summary = response.choices[0].message.content.strip()
                logger.info(f"    ✓ Summary length: {len(summary)} chars")
                return summary
            except Exception as e:
                error_str = str(e)
                is_retryable = (
                    "429" in error_str
                    or "503" in error_str
                    or "502" in error_str
                    or "rate" in error_str.lower()
                    or "overloaded" in error_str.lower()
                )

                if is_retryable and attempt < self.MAX_RETRIES:
                    wait_time = self.INITIAL_BACKOFF * (2 ** attempt)
                    logger.warning(
                        f"    ⏳ Error (attempt {attempt + 1}/{self.MAX_RETRIES}): "
                        f"{type(e).__name__}. Waiting {wait_time:.0f}s ..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"    ✗ Failed to summarize {image.filename}: {e}")
                    return f"[Error: could not summarize image — {type(e).__name__}]"

    def summarize_all(
        self,
        images: list[ExtractedImage],
        cache_path: str = "./mock_s3_storage/summaries_cache.json",
    ) -> list[ExtractedImage]:
        """
        Summarize every image, with disk caching.
        Already-summarized images are loaded from cache instantly.
        New summaries are saved to cache after each image.
        """
        # Load existing cache
        cache = {}
        cache_file = Path(cache_path)
        if cache_file.exists():
            try:
                cache = json.loads(cache_file.read_text(encoding="utf-8"))
                logger.info(f"  Loaded {len(cache)} cached summaries from {cache_path}")
            except Exception:
                cache = {}

        total = len(images)
        cached_count = 0
        for idx, img in enumerate(images):
            # Check cache first
            if img.filename in cache:
                img.summary = cache[img.filename]
                cached_count += 1
                logger.info(f"  [{idx + 1}/{total}] {img.filename} — cached ✓")
                continue

            logger.info(f"  [{idx + 1}/{total}] Processing {img.filename}")
            img.summary = self.summarize(img)

            # Save to cache immediately (so progress is never lost)
            cache[img.filename] = img.summary
            cache_file.write_text(json.dumps(cache, indent=2), encoding="utf-8")

            # Rate-limit delay
            if idx < total - 1:
                time.sleep(self.DELAY_BETWEEN)

        logger.info(f"  Summary complete: {cached_count} cached, {total - cached_count} new")
        return images


# ═══════════════════════════════════════════════════════════════════════════════
#  3. DOCUMENT REASSEMBLER
# ═══════════════════════════════════════════════════════════════════════════════

class DocumentReassembler:
    """
    Reconstructs the full document text, injecting IMAGE_REFERENCE tags
    at the positions where images were extracted.

    Tag format:
      [IMAGE_REFERENCE | URL: /mock_s3_storage/{filename} | SUMMARY: {summary}]
    """

    @staticmethod
    def reassemble(extraction: ExtractionResult) -> str:
        """Return the full document text with image references injected."""
        full_text_parts = []

        # Group images by page
        images_by_page: dict[int, list[ExtractedImage]] = {}
        for img in extraction.images:
            images_by_page.setdefault(img.page_number, []).append(img)

        # Sort pages
        for page_num in sorted(extraction.page_texts.keys()):
            page_text = extraction.page_texts[page_num].strip()

            # Append the page text
            if page_text:
                full_text_parts.append(page_text)

            # Append image references for this page (after the page text)
            if page_num in images_by_page:
                # Sort by position index to maintain original order
                page_images = sorted(images_by_page[page_num], key=lambda x: x.position_index)
                for img in page_images:
                    # Clean summary: collapse internal newlines so the tag stays on one line
                    clean_summary = re.sub(r"\s+", " ", img.summary).strip()
                    tag = (
                        f"[IMAGE_REFERENCE | URL: /mock_s3_storage/{img.filename} "
                        f"| SUMMARY: {clean_summary}]"
                    )
                    full_text_parts.append(tag)

        return "\n\n".join(full_text_parts)


# ═══════════════════════════════════════════════════════════════════════════════
#  4. SMART CHUNKER
# ═══════════════════════════════════════════════════════════════════════════════

class SmartChunker:
    """
    Wraps LangChain's RecursiveCharacterTextSplitter.

    Key guarantee: [IMAGE_REFERENCE ...] tags are NEVER split in half.
    This is achieved by pre-splitting on image-reference boundaries so that
    each tag is always an atomic unit passed to the recursive splitter.
    """

    # Regex matching our injected image reference tags (greedily match the whole tag)
    IMAGE_TAG_PATTERN = re.compile(r"\[IMAGE_REFERENCE\s*\|[^\]]+\]")

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # The recursive splitter uses these separators in order.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",    # double newline (paragraph break)
                "\n",      # single newline
                ". ",      # sentence boundary
                " ",       # word boundary
                "",        # character-level fallback
            ],
            keep_separator=True,
            is_separator_regex=False,
        )

    def chunk(self, text: str) -> list[str]:
        """
        Split text into chunks, guaranteeing image tags stay intact.

        Strategy:
          1. Split the text into segments around IMAGE_REFERENCE tags.
          2. Each tag becomes its own atomic segment.
          3. We then feed segments to the recursive splitter, but any segment
             that is a complete image tag is kept whole (even if > chunk_size).
        """
        # Split into interleaved [text, tag, text, tag, ...] segments
        segments = self.IMAGE_TAG_PATTERN.split(text)
        tags = self.IMAGE_TAG_PATTERN.findall(text)

        # Interleave segments and tags
        parts = []
        for i, segment in enumerate(segments):
            if segment.strip():
                parts.append(("text", segment.strip()))
            if i < len(tags):
                parts.append(("tag", tags[i]))

        # Now chunk each text segment and keep tags whole
        final_chunks = []
        text_buffer = ""

        for part_type, content in parts:
            if part_type == "tag":
                # Flush any accumulated text first
                if text_buffer.strip():
                    final_chunks.extend(self.splitter.split_text(text_buffer.strip()))
                    text_buffer = ""
                # Add tag as its own chunk (never split)
                final_chunks.append(content)
            else:
                text_buffer += "\n\n" + content if text_buffer else content

        # Flush remaining text
        if text_buffer.strip():
            final_chunks.extend(self.splitter.split_text(text_buffer.strip()))

        logger.info(f"Chunking complete — {len(final_chunks)} chunks produced")
        return final_chunks


# ═══════════════════════════════════════════════════════════════════════════════
#  5. VECTOR STORE MANAGER (PGVector — Persistent PostgreSQL Storage)
# ═══════════════════════════════════════════════════════════════════════════════

# Default connection string — override via PG_CONNECTION_STRING in .env
DEFAULT_PG_CONNECTION = os.environ.get(
    "PG_CONNECTION_STRING",
    "postgresql+psycopg://langchain:langchain@localhost:5432/langchain"
)


class NvidiaEmbeddings(Embeddings):
    """
    Custom embeddings class for NVIDIA NIM asymmetric models.
    Passes input_type='passage' for documents, 'query' for queries.
    Uses the raw OpenAI client — no Pydantic conflicts.
    """

    def __init__(self, model: str, base_url: str, api_key: str, batch_size: int = 50):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)
        self.model = model
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents (passages)."""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model,
                extra_body={"input_type": "passage"},
            )
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        response = self.client.embeddings.create(
            input=[text],
            model=self.model,
            extra_body={"input_type": "query"},
        )
        return response.data[0].embedding


class VectorStoreManager:
    """
    Stores embeddings in a PostgreSQL database with the pgvector extension.
    Data persists across runs — no need to re-ingest every time.

    Uses NVIDIA NIM embedding model (free with your NVIDIA API key).
    """

    def __init__(
        self,
        collection_name: str = "multimodal_rag",
        connection_string: str = DEFAULT_PG_CONNECTION,
        embedding_model=None,
    ):
        """
        Args:
            collection_name:   Name for the pgvector collection.
            connection_string: PostgreSQL connection string.
            embedding_model:   Any LangChain Embeddings instance.
                               Defaults to NVIDIA NIM llama-nemotron-embed-1b-v2.
        """
        if embedding_model is None:
            nvidia_key = os.environ.get("NVIDIA_EMBED_API_KEY") or os.environ.get("NVIDIA_API_KEY", "")
            self.embedding = NvidiaEmbeddings(
                model="nvidia/llama-nemotron-embed-1b-v2",
                base_url=NVIDIA_BASE_URL,
                api_key=nvidia_key,
            )
        else:
            self.embedding = embedding_model

        self.collection_name = collection_name
        self.connection_string = connection_string

    def ingest(self, chunks: list[str]) -> PGVector:
        """Insert chunks into the PGVector collection (persistent)."""
        logger.info(f"Connecting to PGVector (collection: {self.collection_name})")

        # Create metadata for each chunk
        metadatas = []
        for i, chunk in enumerate(chunks):
            metadatas.append({
                "chunk_index": i,
                "has_image_ref": "[IMAGE_REFERENCE" in chunk,
                "char_count": len(chunk),
            })

        vectorstore = PGVector.from_texts(
            texts=chunks,
            embedding=self.embedding,
            metadatas=metadatas,
            collection_name=self.collection_name,
            connection=self.connection_string,
            pre_delete_collection=True,  # Fresh ingest each run
        )

        logger.info(f"Ingested {len(chunks)} chunks into PGVector")
        return vectorstore

    def connect(self) -> PGVector:
        """Connect to an existing PGVector collection (for querying)."""
        logger.info(f"Connecting to existing PGVector collection: {self.collection_name}")
        return PGVector(
            embeddings=self.embedding,
            collection_name=self.collection_name,
            connection=self.connection_string,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  6. PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    pdf_path: str,
    output_dir: str = "./mock_s3_storage",
    vlm_model_name: str = "qwen/qwen3.5-122b-a10b",
    embedding_model=None,
    connection_string: str = DEFAULT_PG_CONNECTION,
    collection_name: str = "multimodal_rag",
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> tuple[PGVector, list[str]]:
    """
    End-to-end pipeline: Extract → Summarize → Reassemble → Chunk → Store.

    Args:
        pdf_path:          Path to the input PDF.
        output_dir:        Directory for extracted images.
        vlm_model_name:    NVIDIA NIM model ID for VLM (swappable).
        embedding_model:   Optional custom embedding model.
        connection_string: PostgreSQL connection string.
        collection_name:   Name for the pgvector collection.
        chunk_size:        Chunk size for the text splitter.
        chunk_overlap:     Overlap for the text splitter.

    Returns:
        (vectorstore, chunks) — the PGVector store and the raw chunk list.
    """
    logger.info("=" * 60)
    logger.info("MULTIMODAL RAG INGESTION PIPELINE — START")
    logger.info("=" * 60)

    # ── Step 1: Extract ───────────────────────────────────────────────────
    logger.info("\n📄 STEP 1: Extracting text and images from PDF ...")
    extractor = PDFExtractor(output_dir=output_dir)
    extraction = extractor.extract(pdf_path)

    # ── Step 2: Summarize images ──────────────────────────────────────────
    logger.info("\n🔍 STEP 2: Summarizing images via VLM ...")
    if extraction.images:
        summarizer = ImageSummarizer(model_name=vlm_model_name)
        summarizer.summarize_all(extraction.images)
    else:
        logger.info("  No images found — skipping summarization.")

    # ── Step 3: Reassemble document ───────────────────────────────────────
    logger.info("\n📝 STEP 3: Reassembling document with image references ...")
    reassembled_text = DocumentReassembler.reassemble(extraction)
    logger.info(f"  Reassembled text length: {len(reassembled_text)} chars")

    # ── Step 4: Chunk ─────────────────────────────────────────────────────
    logger.info("\n✂️  STEP 4: Chunking text (preserving image tags) ...")
    chunker = SmartChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk(reassembled_text)

    # ── Step 5: Vector Store (PGVector — Persistent) ──────────────────────
    logger.info("\n🗄️  STEP 5: Ingesting chunks into PGVector ...")
    store_manager = VectorStoreManager(
        collection_name=collection_name,
        connection_string=connection_string,
        embedding_model=embedding_model,
    )
    vectorstore = store_manager.ingest(chunks)

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE ✅")
    logger.info("=" * 60)

    return vectorstore, chunks
