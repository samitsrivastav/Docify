# Docify — Document Intelligence Tool

Upload a PDF or PPTX, get an instant summary, and ask questions about it — answered by a
local Retrieval-Augmented Generation (RAG) pipeline. No OpenAI, Gemini, or any other paid
LLM API is used; everything runs on your machine.

## How it works

```
PDF / PPTX -> Parser -> Chunker -> Embeddings -> FAISS index
                                                     |
                                    Question --------+-------- Summary
                                       |                           |
                                Question embedding          Map-reduce over
                                       |                     page segments
                                FAISS retrieval                    |
                                       |                            |
                                Relevant chunks                     |
                                       |                            |
                                   Local LLM  <----------------------
                                       |
                                Answer + source pages
```

- **Parsing**: [PyMuPDF](https://pymupdf.readthedocs.io/) for PDFs, [python-pptx](https://python-pptx.readthedocs.io/) for
  PPTX. Extracts text per page/slide plus a heuristic heading (largest/boldest text on
  the page, or the slide title).
- **Chunking**: word-window chunks (default 220 words, 40-word overlap), always scoped
  to a single page/slide so every chunk keeps an unambiguous source reference.
- **Embeddings**: [Sentence-Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2` by default).
- **Vector store**: [FAISS](https://github.com/facebookresearch/faiss) (`IndexFlatIP` over normalized
  embeddings = cosine similarity), cached to disk per document (keyed by content hash) so
  re-uploading the same file skips reprocessing.
- **Generation**: pluggable local LLM backend — see [Choosing an LLM backend](#choosing-an-llm-backend).
- **UI**: Streamlit, with document info, short/detailed summaries, and a chat-style Q&A
  panel that shows the source page/slide numbers behind every answer.

## Project structure

```
Docify/
├── app/
│   ├── parser/        # PDF/PPTX text + heading extraction
│   ├── chunker/        # page-scoped word-window chunking
│   ├── embeddings/     # Sentence-Transformers wrapper
│   ├── vector_store/   # FAISS index + on-disk cache
│   ├── retriever/      # query embedding -> top-k chunk lookup
│   ├── llm/            # local LLM backend (transformers / ollama)
│   ├── summarizer/     # short + detailed map-reduce summarization
│   ├── qa/             # RAG prompt assembly + answer generation
│   ├── utils/          # logging, hashing, text helpers
│   └── config/         # settings loaded from .env
├── tests/               # pytest unit tests (self-contained, no fixture files needed)
├── uploads/             # uploaded files + cached FAISS indexes (.index_cache/, gitignored)
├── logs/                # rotating application log (gitignored)
├── app.py               # Streamlit entry point
├── requirements.txt
└── .env.example
```

## Setup

```powershell
# from the Docify/ folder
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # adjust if needed — defaults work out of the box
```

## Run

```powershell
venv\Scripts\activate
streamlit run app.py
```

The first time you upload a document or ask a question, the embedding model and local
LLM are downloaded from Hugging Face automatically (a few hundred MB to ~1 GB depending
on the model) — subsequent runs are fast and fully offline.

## Choosing an LLM backend

Set `LLM_BACKEND` in `.env`:

- **`transformers` (default)** — runs a small Hugging Face model in-process
  (`google/flan-t5-base` by default). Zero extra setup, works immediately, but answer
  quality is modest since it's a small (~250M parameter) model.
- **`ollama`** — routes generation through a locally running
  [Ollama](https://ollama.com) server for noticeably better answers/summaries. Requires:
  1. Install Ollama and run `ollama serve`.
  2. Pull a model, e.g. `ollama pull llama3.2`.
  3. In `.env`: `LLM_BACKEND=ollama` and `LLM_MODEL=llama3.2`.

Both backends implement the same `generate(prompt, max_new_tokens)` interface
(`app/llm/local_llm.py`), so switching is just a config change.

## Deploying to Streamlit Community Cloud

The app is set up to deploy as-is:

1. Push this repo to GitHub (public, or private on an account/plan that Streamlit
   Community Cloud can access).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick the repo,
   branch, and set the main file path to `app.py`.
3. Open **Advanced settings → Secrets** and paste:
   ```toml
   LLM_BACKEND = "transformers"
   LLM_MODEL = "google/flan-t5-small"
   EMBEDDING_MODEL = "all-MiniLM-L6-v2"
   ```
   `flan-t5-small` (~80M params) is used here instead of the local default
   `flan-t5-base` (~250M) to comfortably fit the free tier's RAM limit — swap it back
   to `flan-t5-base` if you're on a paid tier with more memory. The `ollama` backend
   **cannot** be used on Streamlit Community Cloud (no way to run a persistent Ollama
   server there) — it only works for local or self-hosted (Docker/VPS) deployments.
4. Deploy. `app.py` bridges these Secrets into environment variables automatically, so
   the same `app/config/settings.py` code path is used both locally (via `.env`) and on
   Community Cloud (via Secrets) — no code changes needed between the two.

Notes specific to this platform:
- `requirements.txt` pins pip to PyTorch's CPU-only wheel index
  (`--index-url https://download.pytorch.org/whl/cpu`) — the default PyPI Linux wheel
  bundles unused CUDA libraries and is far larger, which matters on a resource-capped
  free tier.
- The first upload/question after a cold start will be slow (downloading the embedding
  + LLM models from Hugging Face into the container's ephemeral disk); subsequent
  requests in the same running instance are fast.
- `.streamlit/config.toml` caps uploads at 50 MB, which is generally plenty for the
  PDFs/slide decks this tool targets — raise it there if you need larger files.

## Running tests

```powershell
venv\Scripts\activate
pytest
```

Tests build their own throwaway PDF/PPTX files in-memory (via PyMuPDF / python-pptx), so
no sample documents are required, and they don't load the embedding/LLM models — they
cover parsing, chunking, and utility logic only.

## Notes / limitations (MVP)

- Scanned/image-only PDFs aren't supported (no OCR yet) — they're rejected with a clear
  error since no text can be extracted.
- Heading/section detection is a font-size/bold heuristic, not a true layout model.
- The default `transformers` backend is CPU-friendly but limited in reasoning ability;
  switch to the `ollama` backend for meaningfully better answers.
