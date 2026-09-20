"""Small shared text/hashing helpers."""
import hashlib


def hash_bytes(data: bytes) -> str:
    """Short, stable content hash used to cache per-document indexes."""
    return hashlib.sha256(data).hexdigest()[:16]


def clean_text(text: str) -> str:
    """Collapse whitespace/newlines into single spaces."""
    return " ".join(text.split())


def word_count(text: str) -> int:
    return len(text.split())
