"""Turns a natural-language query into the top-k most relevant chunks."""
from typing import List, Tuple

from app.chunker import Chunk
from app.embeddings import embed_query
from app.vector_store import VectorStore


def retrieve(store: VectorStore, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
    query_embedding = embed_query(query)
    return store.search(query_embedding, top_k=top_k)
