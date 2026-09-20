"""Splits parsed pages/slides into fixed-size, overlapping word-window chunks.

Chunking is done per page/slide (never across a page boundary), so every
chunk unambiguously retains a single source page/slide number.
"""
from dataclasses import dataclass
from typing import List

from app.parser import ParsedDocument


@dataclass
class Chunk:
    chunk_id: int
    doc_name: str
    page_number: int
    section: str  # nearest detected heading, or "" if none
    text: str
    word_count: int


def chunk_document(doc: ParsedDocument, chunk_size: int = 220, overlap: int = 40) -> List[Chunk]:
    """Word-window chunking with overlap, scoped to a single page/slide per chunk."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    chunks: List[Chunk] = []
    chunk_id = 0

    for page in doc.pages:
        words = page.text.split()
        if not words:
            continue
        section = page.headings[0] if page.headings else ""

        start = 0
        while start < len(words):
            end = start + chunk_size
            window = words[start:end]
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_name=doc.filename,
                    page_number=page.page_number,
                    section=section,
                    text=" ".join(window),
                    word_count=len(window),
                )
            )
            chunk_id += 1
            if end >= len(words):
                break
            start = end - overlap

    return chunks
