"""PDF text + structure extraction using PyMuPDF (fitz)."""
import logging
import statistics
from dataclasses import dataclass, field
from typing import List, Union

import pymupdf as fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class DocumentParseError(Exception):
    """Raised for any document that can't be read or has no usable content."""


@dataclass
class PageContent:
    page_number: int  # 1-indexed
    text: str
    headings: List[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    filename: str
    doc_type: str  # "pdf" or "pptx"
    pages: List[PageContent]
    page_count: int
    word_count: int


_BOLD_FLAG = 1 << 4  # PyMuPDF span flag bit for bold text


def _extract_headings(page: "fitz.Page", size_ratio: float = 1.15) -> List[str]:
    """Heuristic heading detection: spans notably larger than the page's median
    font size, or bold, and short enough to plausibly be a heading."""
    try:
        raw = page.get_text("dict")
    except Exception:
        return []

    sizes = []
    spans_info = []
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                txt = span.get("text", "").strip()
                if not txt:
                    continue
                sizes.append(span["size"])
                spans_info.append((span["size"], span.get("flags", 0), txt))

    if not sizes:
        return []

    median_size = statistics.median(sizes)
    headings: List[str] = []
    for size, flags, txt in spans_info:
        is_bold = bool(flags & _BOLD_FLAG)
        if (size >= median_size * size_ratio or is_bold) and 0 < len(txt.split()) <= 15:
            if txt not in headings:
                headings.append(txt)
    return headings[:10]


def parse_pdf(file_source: Union[bytes, str], filename: str) -> ParsedDocument:
    """Parse a PDF from raw bytes or a file path."""
    try:
        if isinstance(file_source, (bytes, bytearray)):
            doc = fitz.open(stream=bytes(file_source), filetype="pdf")
        else:
            doc = fitz.open(file_source)
    except Exception as e:
        raise DocumentParseError(f"Could not open PDF '{filename}': {e}") from e

    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise DocumentParseError(
                f"'{filename}' is password-protected and cannot be read."
            )

        if doc.page_count == 0:
            raise DocumentParseError(f"'{filename}' has no pages.")

        pages: List[PageContent] = []
        total_words = 0
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = (page.get_text("text") or "").strip()
            headings = _extract_headings(page)
            pages.append(PageContent(page_number=i + 1, text=text, headings=headings))
            total_words += len(text.split())

        if total_words == 0:
            raise DocumentParseError(
                f"'{filename}' contains no extractable text "
                f"(it may be a scanned/image-only PDF)."
            )

        return ParsedDocument(
            filename=filename,
            doc_type="pdf",
            pages=pages,
            page_count=len(pages),
            word_count=total_words,
        )
    finally:
        doc.close()
