"""Contrastive dataset for training the context-conditioned encoder.

Generates positive/negative pairs from Wikipedia articles:
  - Positive: neighboring chunks from the same document share context
  - Negative: same chunk text paired with context from a different document
  - Additional in-batch negatives via InfoNCE loss
"""

from __future__ import annotations

import logging
import random
from typing import Iterator

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from contextrag.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    TRAIN_MIN_ARTICLE_CHARS,
    TRAIN_NUM_ARTICLES,
)
from contextrag.corpus.chunker import chunk_text_simple
from contextrag.models import Chunk

logger = logging.getLogger(__name__)


def load_wikipedia_articles(
    num_articles: int = TRAIN_NUM_ARTICLES,
    min_chars: int = TRAIN_MIN_ARTICLE_CHARS,
    seed: int = 42,
) -> list[str]:
    """Load and sample Wikipedia articles from HuggingFace.

    Returns a list of article full texts, filtered by minimum length.
    """
    from datasets import load_dataset

    logger.info(
        "Loading Wikipedia dataset (sampling %d articles, min %d chars)...",
        num_articles,
        min_chars,
    )

    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    articles: list[str] = []
    rng = random.Random(seed)
    # Sample with reservoir sampling over streaming dataset
    # We take more than needed then filter by length
    target = num_articles * 3  # oversample to account for short articles
    reservoir: list[str] = []
    scan_limit = target * 5
    for i, row in enumerate(tqdm(ds, total=scan_limit, desc="Scanning Wikipedia", unit="art")):
        text = row.get("text", "")
        if len(text) < min_chars:
            continue
        if len(reservoir) < target:
            reservoir.append(text)
        else:
            j = rng.randint(0, i)
            if j < target:
                reservoir[j] = text
        # Stop after scanning enough
        if i >= scan_limit:
            break

    rng.shuffle(reservoir)
    articles = reservoir[:num_articles]
    logger.info("Selected %d articles (scanned %d candidates)", len(articles), len(reservoir))
    return articles


def _build_chunk_pairs(
    articles: list[str],
    context_window_tokens: int,
) -> list[tuple[str, str, str]]:
    """Build (chunk_text, preceding_context, doc_label) triples from articles.

    Each triple represents one chunk with its preceding document context.
    The ``context_window_tokens`` parameter determines how much preceding text
    to include (in approximate word tokens, since exact tokenization depends
    on the model).
    """
    # Approximate: 1 token ≈ 4 chars for English text
    context_window_chars = context_window_tokens * 4 if context_window_tokens > 0 else 0

    triples: list[tuple[str, str, str]] = []
    for doc_idx, article in enumerate(tqdm(articles, desc="Building chunk pairs", unit="doc")):
        label = f"wiki_{doc_idx}"
        chunks = chunk_text_simple(article, source_label=label)
        for chunk in chunks:
            if context_window_tokens == 0:
                ctx = ""
            elif context_window_tokens < 0:
                # Full preceding text
                ctx = article[: chunk.char_offset_start]
            else:
                start = max(0, chunk.char_offset_start - context_window_chars)
                ctx = article[start : chunk.char_offset_start]
            triples.append((chunk.text, ctx, label))

    return triples


class ContrastiveChunkDataset(Dataset):
    """PyTorch dataset that yields (anchor_chunk, anchor_context, positive_chunk, positive_context).

    Positive pairs: neighboring chunks from the same document (within 3 positions).
    The InfoNCE loss treats other items in the batch as negatives.
    """

    def __init__(
        self,
        articles: list[str],
        context_window_tokens: int = 512,
        neighbor_range: int = 3,
        seed: int = 42,
    ) -> None:
        self.rng = random.Random(seed)
        self.context_window_tokens = context_window_tokens

        # Group triples by document
        triples = _build_chunk_pairs(articles, context_window_tokens)
        self._by_doc: dict[str, list[tuple[str, str]]] = {}  # doc_label → [(chunk, ctx)]
        for chunk_text, ctx, label in triples:
            self._by_doc.setdefault(label, []).append((chunk_text, ctx))

        # Build flat list of (doc_label, index_within_doc) for iteration
        self._items: list[tuple[str, int]] = []
        for label, pairs in self._by_doc.items():
            if len(pairs) < 2:
                continue
            for idx in range(len(pairs)):
                self._items.append((label, idx))

        self.neighbor_range = neighbor_range
        logger.info(
            "ContrastiveChunkDataset: %d items from %d documents",
            len(self._items),
            len(self._by_doc),
        )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[str, str, str, str]:
        label, idx = self._items[index]
        pairs = self._by_doc[label]
        anchor_chunk, anchor_ctx = pairs[idx]

        # Pick a neighboring chunk as positive
        lo = max(0, idx - self.neighbor_range)
        hi = min(len(pairs) - 1, idx + self.neighbor_range)
        candidates = [i for i in range(lo, hi + 1) if i != idx]
        pos_idx = self.rng.choice(candidates) if candidates else idx
        pos_chunk, pos_ctx = pairs[pos_idx]

        return anchor_chunk, anchor_ctx, pos_chunk, pos_ctx


def collate_fn(
    batch: list[tuple[str, str, str, str]],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Collate a batch of string tuples into four lists."""
    anchor_chunks, anchor_ctxs, pos_chunks, pos_ctxs = zip(*batch)
    return list(anchor_chunks), list(anchor_ctxs), list(pos_chunks), list(pos_ctxs)
