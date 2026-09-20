import pytest

from app.chunker import chunk_document
from app.parser import PageContent, ParsedDocument


def make_doc(pages_text, doc_type="pdf"):
    pages = [
        PageContent(page_number=i + 1, text=text, headings=[f"Heading {i + 1}"] if text else [])
        for i, text in enumerate(pages_text)
    ]
    word_count = sum(len(t.split()) for t in pages_text)
    return ParsedDocument(
        filename="test.pdf", doc_type=doc_type, pages=pages,
        page_count=len(pages), word_count=word_count,
    )


def test_chunk_never_spans_two_pages():
    doc = make_doc(["word " * 300, "other " * 300])
    chunks = chunk_document(doc, chunk_size=100, overlap=20)
    assert all(c.page_number in (1, 2) for c in chunks)
    page1_chunks = [c for c in chunks if c.page_number == 1]
    page2_chunks = [c for c in chunks if c.page_number == 2]
    assert len(page1_chunks) > 1
    assert len(page2_chunks) > 1
    # no chunk mixes vocabulary from both pages
    assert all("other" not in c.text for c in page1_chunks)
    assert all("word" not in c.text for c in page2_chunks)


def test_chunk_overlap_present():
    doc = make_doc([" ".join(f"tok{i}" for i in range(300))])
    chunks = chunk_document(doc, chunk_size=100, overlap=20)
    assert len(chunks) >= 2
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    # last 20 words of chunk 1 should reappear at the start of chunk 2
    assert first_words[-20:] == second_words[:20]


def test_empty_pages_produce_no_chunks():
    doc = make_doc(["", "   ", ""])
    chunks = chunk_document(doc)
    assert chunks == []


def test_section_uses_first_heading():
    doc = make_doc(["some content here"])
    chunks = chunk_document(doc)
    assert chunks[0].section == "Heading 1"


def test_chunk_size_must_be_positive():
    doc = make_doc(["content"])
    with pytest.raises(ValueError):
        chunk_document(doc, chunk_size=0)
