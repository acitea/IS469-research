"""Dense retrieval: query the dense and contextual ChromaDB indexes."""

from __future__ import annotations

from pathlib import Path

from contextrag.config import (
    CONTEXTUAL_COLLECTION,
    DATABASE_DIR,
    DENSE_CANDIDATES,
)
from contextrag.index.dense_store import query_contextual_index
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
