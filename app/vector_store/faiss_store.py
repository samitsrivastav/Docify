"""FAISS-backed vector store holding chunk embeddings + their metadata."""
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from app.chunker import Chunk


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine sim
        self.chunks: List[Chunk] = []

    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        if embeddings.shape[0] == 0:
            return
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if self.index.ntotal == 0:
            return []
        query_embedding = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        store = cls(dim=index.d)
        store.index = index
        store.chunks = chunks
        return store
