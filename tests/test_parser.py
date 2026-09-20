import pymupdf as fitz  # PyMuPDF
import pytest

from app.parser import DocumentParseError, parse_document
from app.parser.pdf_parser import parse_pdf


def make_pdf_bytes(pages_text):
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=14)
    data = doc.tobytes()
    doc.close()
    return data


def test_parse_pdf_extracts_pages_and_word_count():
    pdf_bytes = make_pdf_bytes(["Hello world, this is page one.", "Second page content here."])
    parsed = parse_pdf(pdf_bytes, "sample.pdf")
    assert parsed.page_count == 2
    assert parsed.doc_type == "pdf"
    assert parsed.word_count > 0
    assert parsed.pages[0].page_number == 1
    assert "Hello" in parsed.pages[0].text


def test_parse_pdf_raises_on_empty_document():
    doc = fitz.open()
    doc.new_page()  # blank page, no text
    pdf_bytes = doc.tobytes()
    doc.close()
    with pytest.raises(DocumentParseError):
        parse_pdf(pdf_bytes, "blank.pdf")


def test_parse_document_dispatches_by_extension():
    pdf_bytes = make_pdf_bytes(["Some text"])
    parsed = parse_document(pdf_bytes, "sample.pdf")
    assert parsed.doc_type == "pdf"


def test_parse_document_rejects_unsupported_extension():
    with pytest.raises(DocumentParseError):
        parse_document(b"not a real file", "sample.txt")


def test_parse_document_rejects_legacy_ppt():
    with pytest.raises(DocumentParseError):
        parse_document(b"not a real file", "sample.ppt")
