
"""Extract PDF pages and save page-preserving chunks."""

import fitz

from core.models import ManualChunk


def extract_pages(pdf_path):
    """Return non-empty PDF pages with 1-based page numbers."""
    pages = []

    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page_number": index + 1,
                    "text": text,
                })

    return pages


def chunk_page(text, max_chars=1200, overlap=150):
    """Split text into chunks without crossing page boundaries."""
    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = end - overlap

    return chunks


def process_manual(manual):
    """Extract a manual's PDF and persist its chunks."""
    manual.processing_status = "processing"
    manual.save(update_fields=["processing_status"])

    try:
        if not manual.pdf:
            raise ValueError("No PDF file is attached to this manual.")

        pages = extract_pages(manual.pdf.path)

        if not pages:
            raise ValueError(
                "No extractable text found. The PDF may be empty, "
                "image-only, or scanned."
            )

        # Build the new chunks before replacing existing ones.
        new_chunks = []

        for page in pages:
            for index, chunk_text in enumerate(chunk_page(page["text"])):
                new_chunks.append(
                    ManualChunk(
                        manual=manual,
                        page_number=page["page_number"],
                        chunk_index=index,
                        text=chunk_text,
                    )
                )

        if not new_chunks:
            raise ValueError("No text chunks were generated.")

        # Replace old chunks only after successful extraction.
        ManualChunk.objects.filter(manual=manual).delete()
        ManualChunk.objects.bulk_create(new_chunks)

        manual.processing_status = "processed"
        manual.save(update_fields=["processing_status"])

        return len(new_chunks)

    except Exception:
        manual.processing_status = "failed"
        manual.save(update_fields=["processing_status"])
        raise
