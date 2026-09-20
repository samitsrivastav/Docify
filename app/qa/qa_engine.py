"""Retrieval-augmented question answering over a document's vector store."""
import logging
from dataclasses import dataclass
from typing import List

from app.chunker import Chunk
from app.llm import generate
from app.retriever import retrieve
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """Answer the question using ONLY the context below. If the answer is not \
contained in the context, say "I couldn't find this in the document."

Context:
{context}

Question: {question}
Answer:"""


@dataclass
class QAResult:
    answer: str
    sources: List[Chunk]
    scores: List[float]


def answer_question(store: VectorStore, question: str, top_k: int = 5) -> QAResult:
    results = retrieve(store, question, top_k=top_k)
    if not results:
        return QAResult(
            answer="The document index is empty — please upload a document first.",
            sources=[],
            scores=[],
        )

    context = "\n\n".join(f"[Page {c.page_number}] {c.text}" for c, _ in results)
    prompt = _PROMPT_TEMPLATE.format(context=context[:4000], question=question)
    answer = generate(prompt, max_new_tokens=200)

    chunks = [c for c, _ in results]
    scores = [s for _, s in results]
    return QAResult(answer=answer, sources=chunks, scores=scores)
