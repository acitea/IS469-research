"""Dense retrieval: query the conditioned and contextual ChromaDB indexes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from contextrag.config import (
    CONDITIONED_COLLECTION,
    CONTEXTUAL_COLLECTION,
    DATABASE_DIR,
    DENSE_CANDIDATES,
)
from contextrag.index.dense_store import query_conditioned_index, query_contextual_index
from contextrag.models import RetrievalHit


def retrieve_conditioned(
    query_embedding: np.ndarray,
    persist_dir: Path | None = None,
    collection_name: str = CONDITIONED_COLLECTION,
    top_k: int = DENSE_CANDIDATES,
) -> list[RetrievalHit]:
    """Query the context-conditioned dense index."""
    persist_dir = persist_dir or DATABASE_DIR / "conditioned"
    return query_conditioned_index(query_embedding, persist_dir, collection_name, top_k)


def retrieve_contextual(
    query: str,
    persist_dir: Path | None = None,
    collection_name: str = CONTEXTUAL_COLLECTION,
    top_k: int = DENSE_CANDIDATES,
) -> list[RetrievalHit]:
    """Query the Anthropic-style contextual dense index."""
    persist_dir = persist_dir or DATABASE_DIR / "contextual"
    return query_contextual_index(query, persist_dir, collection_name, top_k)
