"""Batch inference for producing context-conditioned embeddings."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from contextrag.config import DATABASE_DIR, DEFAULT_CONTEXT_WINDOW, MODEL_DIR
from contextrag.models import Chunk, ContextualChunk
from contextrag.nn.encoder import ContextConditionedEncoder

logger = logging.getLogger(__name__)


def _get_preceding_context(
    chunk: Chunk,
    doc_texts: dict[str, str],
    context_window_tokens: int,
) -> str:
    """Extract preceding document text for a chunk."""
    full_text = doc_texts.get(chunk.doc_file_name, "")
    if context_window_tokens == 0 or not full_text:
        return ""
    preceding = full_text[: chunk.char_offset_start]
    if context_window_tokens > 0:
        # Approximate: 1 token ≈ 4 chars
        max_chars = context_window_tokens * 4
        preceding = preceding[-max_chars:]
    return preceding


def embed_chunks(
    chunks: list[Chunk],
    doc_texts: dict[str, str],
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    batch_size: int = 16,
    model_path: Path | None = None,
    device_name: str | None = None,
) -> tuple[list[ContextualChunk], np.ndarray]:
    """Produce context-conditioned embeddings for all chunks.

    Returns (contextual_chunks, embeddings_array) where embeddings_array
    has shape ``(N, 768)``.
    """
    if device_name:
        device = torch.device(device_name)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load model
    model = ContextConditionedEncoder()
    model_path = model_path or MODEL_DIR / "best_model.pt"
    if model_path.exists():
        model.load_trainable(str(model_path), map_location=str(device))
        logger.info("Loaded trained weights from %s", model_path)
    else:
        logger.warning("No trained model at %s — using random init", model_path)
    model.to(device)
    model.eval()

    # Build contextual chunks with preceding context
    ctx_chunks: list[ContextualChunk] = []
    for chunk in chunks:
        preceding = _get_preceding_context(chunk, doc_texts, context_window)
        ctx_chunk = ContextualChunk(
            chunk=chunk,
            preceding_context=preceding,
            conditioned_embedding_text=f"{preceding}\n\n{chunk.text}" if preceding else chunk.text,
        )
        ctx_chunks.append(ctx_chunk)

    # Batch inference
    all_embeddings: list[np.ndarray] = []
    num_batches = (len(ctx_chunks) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(ctx_chunks), batch_size), total=num_batches, desc="Embedding chunks", unit="batch"):
        batch = ctx_chunks[i : i + batch_size]
        chunk_texts = [c.chunk.text for c in batch]
        context_texts = [c.preceding_context for c in batch]

        with torch.no_grad():
            if context_window == 0:
                emb = model.encode_no_context(chunk_texts)
            else:
                emb = model(chunk_texts, context_texts)

        all_embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info("Produced %d embeddings of dim %d", embeddings.shape[0], embeddings.shape[1])
    return ctx_chunks, embeddings
