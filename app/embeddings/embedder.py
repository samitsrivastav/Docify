"""Local sentence-embedding generation via Sentence-Transformers."""
import logging
from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config.settings import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Encode a batch of texts into L2-normalized float32 embeddings (cosine-ready)."""
    if not texts:
        model = _load_model()
        dim = model.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype="float32")

    model = _load_model()
    embeddings = model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    return embeddings.astype("float32")


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]
