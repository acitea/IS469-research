"""COIL-style contextual lexical retrieval.

Approximates COIL's token-level contextualized matching using:
1. Anchor term extraction (dates, amounts, tickers, SEC refs, etc.)
2. Weighted BM25 where anchor tokens are boosted
3. Context-aware: anchors inherited from preceding context
4. Combined score: alpha * BM25 + (1-alpha) * anchor_overlap
"""

from __future__ import annotations

import logging
import re
from typing import Any

from rank_bm25 import BM25Okapi

from contextrag.config import ANCHOR_BOOST, COIL_ALPHA
from contextrag.models import ContextualChunk, RetrievalHit

logger = logging.getLogger(__name__)

# Patterns for financial document anchor terms
_ANCHOR_PATTERNS = [
    r"\b\d{4}\b",                                        # years: 2022, 2023
    r"\$[\d,]+\.?\d*\s*(?:million|billion|thousand)?",   # dollar amounts
    r"\b(?:Item\s+\d+[A-Z]?)\b",                        # SEC item references
    r"\b[A-Z]{2,5}\b",                                   # tickers, abbreviations
    r"\bFY\d{2,4}\b",                                    # fiscal year references
    r"\b\d+[,.]?\d*\s*%",                                # percentages
    r"\b(?:Q[1-4]\s*\d{4})\b",                          # quarter references
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",                     # dates
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _ANCHOR_PATTERNS]


def extract_anchor_terms(text: str) -> set[str]:
    """Extract anchor terms (dates, amounts, identifiers) from text."""
    anchors: set[str] = set()
    for pattern in _COMPILED_PATTERNS:
        for match in pattern.finditer(text):
            anchors.add(match.group().strip().lower())
    return anchors


def tokenize_with_anchor_boost(
    text: str,
    anchor_terms: set[str] | None = None,
    boost: int = ANCHOR_BOOST,
) -> list[str]:
    """Tokenize text, duplicating anchor terms to boost their BM25 weight."""
    base_tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
    if anchor_terms is None:
        anchor_terms = extract_anchor_terms(text)

    boosted: list[str] = []
    for token in base_tokens:
        boosted.append(token)
        if token in anchor_terms:
            boosted.extend([token] * (boost - 1))
    return boosted


def build_coil_index(
    ctx_chunks: list[ContextualChunk],
) -> tuple[BM25Okapi, list[str], list[set[str]]]:
    """Build a weighted BM25 index with anchor boosting.

    Returns (bm25_index, chunk_ids, chunk_anchor_sets).
    """
    corpus_tokens: list[list[str]] = []
    chunk_ids: list[str] = []
    chunk_anchors: list[set[str]] = []

    for ctx in ctx_chunks:
        # Extract anchors from both chunk and preceding context
        chunk_text_anchors = extract_anchor_terms(ctx.chunk.text)
        context_anchors = extract_anchor_terms(ctx.preceding_context[:2000]) if ctx.preceding_context else set()
        combined_anchors = chunk_text_anchors | context_anchors

        # Tokenize chunk text with anchor boosting (include context anchors)
        tokens = tokenize_with_anchor_boost(ctx.chunk.text, combined_anchors)
        corpus_tokens.append(tokens)
        chunk_ids.append(ctx.chunk.chunk_id)
        chunk_anchors.append(combined_anchors)

    bm25 = BM25Okapi(corpus_tokens)
    logger.info("Built COIL/BM25 index: %d chunks", len(chunk_ids))
    return bm25, chunk_ids, chunk_anchors


def query_coil(
    query: str,
    bm25: BM25Okapi,
    chunk_ids: list[str],
    chunk_anchors: list[set[str]],
    top_k: int = 20,
    alpha: float = COIL_ALPHA,
) -> list[RetrievalHit]:
    """Query the COIL index and return ranked hits.

    Score = alpha * normalized_bm25 + (1-alpha) * anchor_overlap.
    """
    query_anchors = extract_anchor_terms(query)
    query_tokens = tokenize_with_anchor_boost(query, query_anchors)

    if not query_tokens:
        return []

    # BM25 scores
    bm25_scores = bm25.get_scores(query_tokens)
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

    # Combine with anchor overlap
    combined: list[tuple[int, float]] = []
    for idx in range(len(chunk_ids)):
        norm_bm25 = bm25_scores[idx] / max_bm25

        # Jaccard-like anchor overlap
        if query_anchors and chunk_anchors[idx]:
            overlap = len(query_anchors & chunk_anchors[idx])
            union = len(query_anchors | chunk_anchors[idx])
            anchor_score = overlap / union if union > 0 else 0.0
        else:
            anchor_score = 0.0

        score = alpha * norm_bm25 + (1 - alpha) * anchor_score
        combined.append((idx, score))

    # Sort by score descending
    combined.sort(key=lambda x: x[1], reverse=True)

    hits: list[RetrievalHit] = []
    for rank, (idx, score) in enumerate(combined[:top_k], start=1):
        hits.append(
            RetrievalHit(
                chunk_id=chunk_ids[idx],
                signal_name="coil",
                rank=rank,
                score=score,
            )
        )

    return hits
