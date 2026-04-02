"""BM25 index build, serialize, load, and query."""

from __future__ import annotations

import logging
from pathlib import Path

from contextrag.index.persistence import load_bm25, save_bm25
from contextrag.lexical.coil import build_coil_index, query_coil
from contextrag.models import ContextualChunk, RetrievalHit

logger = logging.getLogger(__name__)


def create_bm25_index(
    ctx_chunks: list[ContextualChunk],
    persist_path: Path,
    force_rebuild: bool = False,
) -> None:
    """Build and persist the COIL/BM25 index."""
    if persist_path.exists() and not force_rebuild:
        logger.info("BM25 index already exists at %s, skipping", persist_path)
        return

    bm25, chunk_ids, chunk_anchors = build_coil_index(ctx_chunks)

    # We need to persist the anchors too for query-time scoring
    import pickle
    persist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(persist_path, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "chunk_ids": chunk_ids,
            "chunk_anchors": [list(a) for a in chunk_anchors],
        }, f)

    logger.info("Saved COIL/BM25 index (%d chunks) to %s", len(chunk_ids), persist_path)


def query_bm25_index(
    query: str,
    persist_path: Path,
    top_k: int = 20,
) -> list[RetrievalHit]:
    """Load the BM25 index and query it."""
    import pickle

    if not persist_path.exists():
        logger.warning("BM25 index not found at %s", persist_path)
        return []

    with open(persist_path, "rb") as f:
        data = pickle.load(f)

    bm25 = data["bm25"]
    chunk_ids = data["chunk_ids"]
    chunk_anchors = [set(a) for a in data["chunk_anchors"]]

    return query_coil(query, bm25, chunk_ids, chunk_anchors, top_k=top_k)
