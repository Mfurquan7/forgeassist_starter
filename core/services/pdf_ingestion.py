
"""Extract PDF pages and save page-preserving chunks."""
import re

import pymupdf

from core.models import ManualChunk


# ============================================================
# HEADING DETECTION
# ============================================================

# Patterns that indicate a line is a section heading.
_HEADING_PATTERNS = [
    re.compile(
        r"^\d+(?:\.\d+)*\s+[A-Z]",  # "1.2 Safety Precautions"
    ),
    re.compile(
        r"^(?:SECTION|CHAPTER|PART|APPENDIX)\s+",  # "SECTION 4"
        re.IGNORECASE,
    ),
]

# Minimum ratio of uppercase letters to qualify as an ALL-CAPS heading.
_UPPER_RATIO = 0.7


def _is_heading_line(line):
    """Detect whether a line is likely a section heading."""
    stripped = line.strip()

    if not stripped or len(stripped) < 3:
        return False

    # Too long to be a heading.
    if len(stripped) > 120:
        return False

    # Numbered section heading.
    for pattern in _HEADING_PATTERNS:
        if pattern.match(stripped):
            return True

    # ALL-CAPS line (short enough to be a heading).
    alpha_chars = [c for c in stripped if c.isalpha()]
    if (
        len(alpha_chars) >= 3
        and len(stripped) <= 80
        and sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        >= _UPPER_RATIO
    ):
        return True

    return False


# ============================================================
# TABLE DETECTION
# ============================================================

_TABLE_SEP_RE = re.compile(r"\||\t")
_TABLE_HEADER_RE = re.compile(
    r"(?:code|error|fault|alarm|meaning|description|response|action|"
    r"parameter|value|unit|interval|task|item|check|frequency|"
    r"symptom|cause|remedy|component)",
    re.IGNORECASE,
)


def _is_table_line(line):
    """Check whether a line looks like a table row."""
    stripped = line.strip()
    if not stripped:
        return False

    # Pipe-separated.
    if "|" in stripped:
        cells = [c.strip() for c in stripped.split("|") if c.strip()]
        return len(cells) >= 2

    # Tab-separated.
    if "\t" in stripped:
        cells = [c.strip() for c in stripped.split("\t") if c.strip()]
        return len(cells) >= 2

    return False


def _is_table_separator(line):
    """Check if a line is a markdown-style table separator."""
    stripped = line.strip().strip("|")
    return bool(
        stripped
        and all(
            re.fullmatch(r":?-{2,}:?", cell.strip())
            for cell in stripped.split("|")
            if cell.strip()
        )
    )


def _detect_chunk_type(text):
    """Classify a chunk as text, table, or error_table."""
    lines = text.strip().splitlines()
    lower = text.lower()

    # Detect vertical error tables starting with Code / Meaning / Response
    if "code\nmeaning\ndocumented response" in lower or "code\nac-90 meaning\ndocumented response" in lower or "code\nmeaning\nresponse" in lower:
        return "error_table"
    if any(lower.startswith(header) for header in ["code\nmeaning", "code\nac-90 meaning", "alarm codes\ncode"]):
        return "error_table"

    table_lines = sum(1 for line in lines if _is_table_line(line))

    if table_lines >= 2 and table_lines / max(len(lines), 1) >= 0.3:
        # Check if it looks like an error code table.
        if any(
            term in lower
            for term in (
                "error code",
                "fault code",
                "alarm code",
                "documented response",
            )
        ):
            return "error_table"
        return "table"

    return "text"


# ============================================================
# STRUCTURED PAGE EXTRACTION
# ============================================================

def extract_pages(pdf_path):
    """
    Return non-empty PDF pages with 1-based page numbers.

    Uses PyMuPDF's text extraction with line structure preserved.
    Compatible with the original interface.
    """
    pages = []

    with pymupdf.open(pdf_path) as document:
        for index, page in enumerate(document):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page_number": index + 1,
                    "text": text,
                })

    return pages


# ============================================================
# SECTION-AWARE CHUNKING
# ============================================================

def chunk_page(text, max_chars=1200, overlap=150):
    """
    Split text into chunks while preserving structure.

    Improvements over original:
    - Headings stay attached to their content.
    - Table rows stay grouped together.
    - Error-code blocks stay intact.
    - Returns list of dicts with text, chunk_type, heading_context.

    For backward compatibility, when called from code expecting
    a list of strings, the returned objects also work as strings
    via __str__.
    """
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return []

    # Phase 1: Identify sections and group lines.
    sections = []
    current_heading = ""
    current_lines = []
    current_is_table = False
    in_table = False

    for line in lines:
        is_heading = _is_heading_line(line)
        is_table = _is_table_line(line)
        is_sep = _is_table_separator(line)

        if is_sep:
            # Skip markdown table separators but stay in table mode.
            continue

        if is_heading:
            # Flush current section.
            if current_lines:
                sections.append({
                    "heading": current_heading,
                    "lines": current_lines,
                    "is_table": current_is_table,
                })

            current_heading = line
            current_lines = [line]
            current_is_table = False
            in_table = False
            continue

        if is_table:
            if not in_table and current_lines and not current_is_table:
                # Start a new table section; flush text before it.
                sections.append({
                    "heading": current_heading,
                    "lines": current_lines,
                    "is_table": False,
                })
                current_lines = []

            current_is_table = True
            in_table = True

        elif in_table:
            # End of table: flush table section.
            if current_lines:
                sections.append({
                    "heading": current_heading,
                    "lines": current_lines,
                    "is_table": True,
                })
                current_lines = []
            in_table = False
            current_is_table = False

        current_lines.append(line)

    # Flush remaining lines.
    if current_lines:
        sections.append({
            "heading": current_heading,
            "lines": current_lines,
            "is_table": current_is_table,
        })

    # Phase 2: Build chunks from sections, respecting max_chars.
    chunks = []

    for section in sections:
        heading = section["heading"]
        section_lines = section["lines"]
        is_table = section["is_table"]

        # For tables: keep all rows together if possible.
        section_text = "\n".join(section_lines)

        if len(section_text) <= max_chars:
            chunks.append(_make_chunk_dict(
                section_text,
                heading,
                is_table,
            ))
            continue

        # Split large sections using the original overlap algorithm.
        sub_chunks = _split_lines_with_overlap(
            section_lines,
            max_chars,
            overlap,
            heading,
        )

        for sub_text in sub_chunks:
            chunks.append(_make_chunk_dict(
                sub_text,
                heading,
                is_table,
            ))

    return chunks


def _make_chunk_dict(text, heading, is_table):
    """Create a chunk info dict that also works as a string."""
    detected = _detect_chunk_type(text)
    if detected in ("error_table", "table"):
        chunk_type = detected
    elif is_table:
        chunk_type = "table"
    elif _is_heading_line(text.split("\n")[0] if text else ""):
        if len(text.splitlines()) <= 2:
            chunk_type = "heading"
        else:
            chunk_type = "section"
    else:
        chunk_type = "text"

    return _ChunkResult(
        text=text,
        chunk_type=chunk_type,
        heading_context=heading,
    )


class _ChunkResult(str):
    """
    A string subclass that carries extra metadata.

    This preserves backward compatibility — existing code that
    iterates over chunk_page() results and treats them as strings
    will continue to work. New code can access .chunk_type and
    .heading_context.
    """

    def __new__(cls, text, chunk_type="text", heading_context=""):
        instance = super().__new__(cls, text)
        instance.chunk_type = chunk_type
        instance.heading_context = heading_context
        return instance


def _split_lines_with_overlap(lines, max_chars, overlap, heading):
    """
    Split lines into chunks with overlap.
    Preserves the original algorithm for backward compatibility.
    """
    chunks = []
    current_lines = []
    current_length = 0

    # Prepend heading to first chunk if it's not already there.
    heading_prefix = ""
    if heading and lines and lines[0] != heading:
        heading_prefix = heading + "\n"

    for line in lines:
        line_length = len(line) + 1

        if current_lines and current_length + line_length > max_chars:
            text = "\n".join(current_lines)
            if heading_prefix and not chunks:
                text = heading_prefix + text
            chunks.append(text)

            # Preserve recent complete lines as overlap.
            overlap_lines = []
            overlap_length = 0

            for previous_line in reversed(current_lines):
                if overlap_length + len(previous_line) + 1 > overlap:
                    break
                overlap_lines.insert(0, previous_line)
                overlap_length += len(previous_line) + 1

            current_lines = overlap_lines
            current_length = overlap_length
            heading_prefix = ""  # Only prepend to first chunk.

        current_lines.append(line)
        current_length += line_length

    if current_lines:
        text = "\n".join(current_lines)
        if heading_prefix and not chunks:
            text = heading_prefix + text
        chunks.append(text)

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
                        text=str(chunk_text),
                        chunk_type=getattr(chunk_text, "chunk_type", "text"),
                        heading_context=getattr(chunk_text, "heading_context", ""),
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
