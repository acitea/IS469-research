"""Lexical retrieval: query the COIL/BM25 index."""

from __future__ import annotations

from pathlib import Path

from contextrag.config import DATABASE_DIR, LEXICAL_CANDIDATES
from contextrag.index.bm25_store import query_bm25_index
from contextrag.models import RetrievalHit


def retrieve_lexical(
    query: str,
    persist_path: Path | None = None,
    top_k: int = LEXICAL_CANDIDATES,
) -> list[RetrievalHit]:
    """Query the COIL/BM25 lexical index."""
    persist_path = persist_path or DATABASE_DIR / "bm25_coil.pkl"
    return query_bm25_index(query, persist_path, top_k=top_k)
