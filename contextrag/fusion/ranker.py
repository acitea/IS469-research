"""Reciprocal Rank Fusion (RRF) across retrieval signals."""

from __future__ import annotations

from contextrag.config import RRF_K
from contextrag.models import RetrievalHit


def reciprocal_rank_fusion(
    hit_lists: list[list[RetrievalHit]],
    k: int = RRF_K,
) -> list[tuple[str, float, list[RetrievalHit]]]:
    """Combine multiple ranked lists into a single ranking using RRF.

    RRF score for chunk *d*::

        score(d) = sum over all signals L of: 1 / (k + rank_L(d))

    Args:
        hit_lists: One list of ``RetrievalHit`` per retrieval signal.
        k: RRF constant (default 60).

    Returns:
        Sorted list of ``(chunk_id, fused_score, contributing_hits)``.
    """
    scores: dict[str, float] = {}
    provenance: dict[str, list[RetrievalHit]] = {}

    for hits in hit_lists:
        for hit in hits:
            rrf_contrib = 1.0 / (k + hit.rank)
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + rrf_contrib
            provenance.setdefault(hit.chunk_id, []).append(hit)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(cid, score, provenance[cid]) for cid, score in ranked]
