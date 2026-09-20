from app.utils.text_utils import clean_text, hash_bytes, word_count


def test_hash_bytes_deterministic_and_short():
    h1 = hash_bytes(b"hello world")
    h2 = hash_bytes(b"hello world")
    h3 = hash_bytes(b"different")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 16


def test_clean_text_collapses_whitespace():
    assert clean_text("  hello   \n\n world  ") == "hello world"


def test_word_count():
    assert word_count("one two three") == 3
    assert word_count("") == 0
