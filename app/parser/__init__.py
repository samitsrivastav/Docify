"""Unified document parsing entry point."""
from app.parser.pdf_parser import DocumentParseError, PageContent, ParsedDocument, parse_pdf
from app.parser.pptx_parser import parse_pptx

__all__ = [
    "DocumentParseError",
    "PageContent",
    "ParsedDocument",
    "parse_pdf",
    "parse_pptx",
    "parse_document",
]


def parse_document(file_bytes: bytes, filename: str) -> ParsedDocument:
    """Dispatch to the right parser based on file extension."""
    if "." not in filename:
        raise DocumentParseError(f"'{filename}' has no file extension.")

    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return parse_pdf(file_bytes, filename)
    if ext == "pptx":
        return parse_pptx(file_bytes, filename)
    if ext == "ppt":
        raise DocumentParseError(
            "Legacy .ppt files aren't supported — please save as .pptx and re-upload."
        )
    raise DocumentParseError(f"Unsupported file type: .{ext}")
