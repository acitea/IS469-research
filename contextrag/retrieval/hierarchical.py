"""Hierarchical retrieval: query the RAPTOR index across all levels."""

from __future__ import annotations

from pathlib import Path

from contextrag.config import DATABASE_DIR, RAPTOR_CANDIDATES_PER_LEVEL, RAPTOR_COLLECTION
from contextrag.models import RetrievalHit


def retrieve_hierarchical(
    query: str,
    persist_dir: Path | None = None,
    collection_name: str = RAPTOR_COLLECTION,
    top_k_per_level: int = RAPTOR_CANDIDATES_PER_LEVEL,
) -> list[RetrievalHit]:
    """Query the RAPTOR hierarchical index."""
    from contextrag.hierarchy.raptor import RaptorBackend

    backend = RaptorBackend()
    persist_dir = persist_dir or DATABASE_DIR / "raptor"
    return backend.query(query, persist_dir, collection_name, top_k_per_level=top_k_per_level)
