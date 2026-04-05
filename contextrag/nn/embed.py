"""Batch inference for producing embeddings using pluggable embedders."""

from __future__ import annotations

import logging
import numpy as np
from tqdm import tqdm

from contextrag.config import DATABASE_DIR, EMBEDDER_MODE
from contextrag.models import Chunk, ContextualChunk
from contextrag.nn.embedders import get_embedder

logger = logging.getLogger(__name__)


def prepare_ctx_chunks(
    chunks: list[Chunk],
    doc_texts: dict[str, str],
) -> list[ContextualChunk]:
    """Wrap raw Chunks with preceding context — no model inference."""
    ctx_chunks: list[ContextualChunk] = []
    for chunk in chunks:
        full_text = doc_texts.get(chunk.doc_file_name, "")
        preceding = full_text[: chunk.char_offset_start] if full_text else ""
        ctx_chunks.append(ContextualChunk(
            chunk=chunk,
            preceding_context=preceding,
            conditioned_embedding_text=f"{preceding}\n\n{chunk.text}" if preceding else chunk.text,
        ))
    return ctx_chunks


def embed_chunks(
    chunks: list[Chunk],
    doc_texts: dict[str, str],
    embedder_mode: str = EMBEDDER_MODE,
    batch_size: int = 16,
) -> tuple[list[ContextualChunk], np.ndarray]:
    """Produce embeddings for all chunks using the specified embedder.

    Supports three modes:
    - "default": chunks embedded standalone (OpenAI text-embedding-3-small)
    - "contextual": chunks + semantic pooling of preceding context (OpenAI + pooling)
    - "voyage": chunks with native contextualization (Voyage API)

    Returns (contextual_chunks, embeddings_array) where embeddings_array
    has shape ``(N, 512)`` (all embedders output 512 dimensions).
    """
    logger.info(f"Embedding {len(chunks)} chunks using embedder: {embedder_mode}")

    # Get embedder instance
    embedder = get_embedder(embedder_mode)
    logger.info(f"Using embedder: {embedder.__class__.__name__}")

    # Build contextual chunks with preceding context metadata
    ctx_chunks = prepare_ctx_chunks(chunks, doc_texts)

    # Batch embedding using the embedder
    all_embeddings: list[np.ndarray] = []
    num_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(chunks), batch_size), total=num_batches, desc="Embedding chunks", unit="batch"):
        batch_chunks = chunks[i : i + batch_size]
        
        # Get batch of embeddings from the embedder
        batch_embeddings = embedder.embed_chunks_with_context(batch_chunks, doc_texts)
        all_embeddings.append(batch_embeddings)

    embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info("Produced %d embeddings of dim %d", embeddings.shape[0], embeddings.shape[1])
    return ctx_chunks, embeddings
