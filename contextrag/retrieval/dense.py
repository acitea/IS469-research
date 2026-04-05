"""Dense retrieval: query the dense and contextual ChromaDB indexes."""

from __future__ import annotations

from pathlib import Path

from contextrag.config import (
    CONTEXTUAL_COLLECTION,
    DATABASE_DIR,
    DENSE_CANDIDATES,
    DENSE_COLLECTION,
    EMBEDDER_MODE,
)
from contextrag.index.dense_store import query_contextual_index, query_dense_index
from contextrag.models import RetrievalHit


def retrieve_contextual(
    query: str,
    persist_dir: Path | None = None,
    collection_name: str = CONTEXTUAL_COLLECTION,
    top_k: int = DENSE_CANDIDATES,
) -> list[RetrievalHit]:
    """Query the contextual dense index (LLM-based context retrieval)."""
    persist_dir = persist_dir or DATABASE_DIR / "contextual"
    return query_contextual_index(query, persist_dir, collection_name, top_k)


def retrieve_dense(
    query: str,
    persist_dir: Path | None = None,
    collection_name: str = DENSE_COLLECTION,
    top_k: int = DENSE_CANDIDATES,
    embedder_mode: str = EMBEDDER_MODE,
) -> list[RetrievalHit]:
    """Query the dense index (custom embedder embeddings)."""
    persist_dir = persist_dir or DATABASE_DIR / "dense"
    return query_dense_index(query, persist_dir, collection_name, top_k, embedder_mode)
