"""Map-reduce style summarization built on the local LLM backend."""
import logging
from typing import List

from app.llm import generate
from app.parser import ParsedDocument

logger = logging.getLogger(__name__)

_SEGMENT_WORD_LIMIT = 500  # keep prompts small enough for local models


def _segments(doc: ParsedDocument) -> List[str]:
    """Group consecutive pages into ~500-word segments for map-reduce summarization."""
    segments: List[str] = []
    buffer: List[str] = []
    word_count = 0
    for page in doc.pages:
        if not page.text.strip():
            continue
        buffer.append(page.text)
        word_count += len(page.text.split())
        if word_count >= _SEGMENT_WORD_LIMIT:
            segments.append("\n".join(buffer))
            buffer, word_count = [], 0
    if buffer:
        segments.append("\n".join(buffer))
    return segments or [""]


def summarize_short(doc: ParsedDocument, max_new_tokens: int = 120) -> str:
    """A short, 3-5 sentence summary of the whole document."""
    segments = _segments(doc)
    partials = [
        generate(f"Summarize the following text in 2-3 sentences:\n\n{seg[:3000]}", max_new_tokens=100)
        for seg in segments
    ]
    if len(partials) == 1:
        return partials[0]

    combined = "\n".join(partials)
    return generate(
        f"Combine these notes into one short 3-5 sentence summary:\n\n{combined[:3000]}",
        max_new_tokens=max_new_tokens,
    )


def summarize_detailed(doc: ParsedDocument, max_new_tokens: int = 150) -> str:
    """A longer, per-page/section outline of the document."""
    outline_lines = []
    label = "Slide" if doc.doc_type == "pptx" else "Page"
    for page in doc.pages:
        if not page.text.strip():
            continue
        prompt = f"Summarize the key points of this text in 2-4 bullet points:\n\n{page.text[:2000]}"
        bullets = generate(prompt, max_new_tokens=max_new_tokens)
        heading = page.headings[0] if page.headings else f"{label} {page.page_number}"
        outline_lines.append(f"**{heading}** ({label.lower()} {page.page_number})\n{bullets}")
    return "\n\n".join(outline_lines)
