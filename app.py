"""Docify — Streamlit UI entry point.

Upload a PDF/PPTX, get document info + summaries, and ask grounded questions
answered by a local RAG pipeline (no paid LLM APIs).
"""
import logging
import os

import streamlit as st

# On Streamlit Community Cloud, config is set via the dashboard's "Secrets" panel
# (TOML), which only populates `st.secrets` — it does NOT touch the process
# environment. Our settings module reads plain env vars (so the same code works
# locally via .env), so bridge secrets into os.environ before anything imports
# app.config.settings. Locally, with no secrets.toml, this is just a no-op.
try:
    for _secret_key, _secret_value in st.secrets.items():
        os.environ.setdefault(_secret_key, str(_secret_value))
except Exception:
    pass

from app.chunker import chunk_document
from app.config.settings import settings
from app.embeddings import embed_texts
from app.llm import LLMError
from app.parser import DocumentParseError, parse_document
from app.qa import answer_question
from app.summarizer import summarize_detailed, summarize_short
from app.utils.logging_config import setup_logging
from app.utils.text_utils import hash_bytes
from app.vector_store import VectorStore

setup_logging()
logger = logging.getLogger("docify.app")

st.set_page_config(page_title="Docify", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "doc": None,
    "store": None,
    "doc_hash": None,
    "chat_history": [],
    "short_summary": None,
    "detailed_summary": None,
}
for _key, _value in _DEFAULTS.items():
    st.session_state.setdefault(_key, _value)


def build_index(file_bytes: bytes, filename: str):
    doc = parse_document(file_bytes, filename)
    chunks = chunk_document(doc, chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)
    if not chunks:
        raise DocumentParseError("No content could be chunked from this document.")
    embeddings = embed_texts([c.text for c in chunks])
    store = VectorStore(dim=embeddings.shape[1])
    store.add(embeddings, chunks)
    return doc, store


def get_or_build_index(file_bytes: bytes, filename: str):
    file_hash = hash_bytes(file_bytes)
    cache_path = settings.INDEX_CACHE_DIR / file_hash
    if (cache_path / "index.faiss").exists():
        logger.info("Loading cached index for %s", filename)
        store = VectorStore.load(cache_path)
        doc = parse_document(file_bytes, filename)  # re-parse: cheap, needed for metadata/summaries
        return doc, store

    logger.info("Building new index for %s", filename)
    doc, store = build_index(file_bytes, filename)
    store.save(cache_path)
    return doc, store


def render_sources(sources, doc_type: str) -> None:
    if not sources:
        return
    label = "slide" if doc_type == "pptx" else "page"
    pages = sorted({c.page_number for c in sources})
    with st.expander(f"Sources: {label}s " + ", ".join(str(p) for p in pages)):
        for c in sources:
            heading = f" — {c.section}" if c.section else ""
            st.markdown(f"**{label.capitalize()} {c.page_number}**{heading}")
            preview = c.text[:500] + ("..." if len(c.text) > 500 else "")
            st.caption(preview)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Docify")
    st.caption("Local, private document Q&A — no cloud LLM APIs.")
    st.markdown("---")
    st.subheader("Settings")
    st.text(f"Embedding model:\n{settings.EMBEDDING_MODEL}")
    st.text(f"LLM backend: {settings.LLM_BACKEND}")
    st.text(f"LLM model:\n{settings.LLM_MODEL}")
    top_k = st.slider("Chunks to retrieve", min_value=2, max_value=10, value=settings.TOP_K)
    st.markdown("---")
    if st.button("Clear document", use_container_width=True):
        for key, value in _DEFAULTS.items():
            st.session_state[key] = value
        st.rerun()

# ---------------------------------------------------------------------------
# Header + upload
# ---------------------------------------------------------------------------
st.title("📄 Docify")
st.caption("Upload a PDF or PPTX, get instant summaries, and ask questions grounded in the document.")

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "pptx"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = hash_bytes(file_bytes)
    if file_hash != st.session_state.doc_hash:
        with st.spinner(f"Processing {uploaded_file.name}... this may take a moment on first run."):
            try:
                doc, store = get_or_build_index(file_bytes, uploaded_file.name)
                st.session_state.doc = doc
                st.session_state.store = store
                st.session_state.doc_hash = file_hash
                st.session_state.chat_history = []
                st.session_state.short_summary = None
                st.session_state.detailed_summary = None
                st.success(f"Processed {uploaded_file.name}")
            except DocumentParseError as e:
                st.error(str(e))
                logger.warning("Parse error for %s: %s", uploaded_file.name, e)
            except Exception as e:
                st.error(f"Unexpected error while processing the document: {e}")
                logger.exception("Unexpected error processing %s", uploaded_file.name)

doc = st.session_state.doc
store = st.session_state.store

if doc is None:
    st.info("👆 Upload a PDF or PPTX to get started.")
    st.stop()

# ---------------------------------------------------------------------------
# Document info
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Document", doc.filename)
col2.metric("Slides" if doc.doc_type == "pptx" else "Pages", doc.page_count)
col3.metric("Word count", f"{doc.word_count:,}")

st.markdown("---")

tab_summary, tab_qa = st.tabs(["📝 Summary", "💬 Ask Questions"])

# ---------------------------------------------------------------------------
# Summary tab
# ---------------------------------------------------------------------------
with tab_summary:
    sc1, sc2 = st.columns(2)

    with sc1:
        if st.button("Generate short summary"):
            with st.spinner("Summarizing..."):
                try:
                    st.session_state.short_summary = summarize_short(doc)
                except LLMError as e:
                    st.error(str(e))
        if st.session_state.short_summary:
            st.subheader("Short summary")
            st.write(st.session_state.short_summary)

    with sc2:
        if st.button("Generate detailed summary / outline"):
            with st.spinner("Building detailed outline... this can take a while for long documents."):
                try:
                    st.session_state.detailed_summary = summarize_detailed(doc)
                except LLMError as e:
                    st.error(str(e))
        if st.session_state.detailed_summary:
            st.subheader("Detailed summary / outline")
            st.markdown(st.session_state.detailed_summary)

# ---------------------------------------------------------------------------
# Q&A tab
# ---------------------------------------------------------------------------
with tab_qa:
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            render_sources(turn["sources"], doc.doc_type)

    question = st.chat_input("Ask a question about the document...")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = answer_question(store, question, top_k=top_k)
                    st.write(result.answer)
                    render_sources(result.sources, doc.doc_type)
                    st.session_state.chat_history.append(
                        {"question": question, "answer": result.answer, "sources": result.sources}
                    )
                except LLMError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Something went wrong answering that question: {e}")
                    logger.exception("QA failure")
