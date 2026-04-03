"""Anthropic-style contextual retrieval: LLM-generated chunk context descriptions."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from contextrag.config import DATABASE_DIR, LLM_MODEL
from contextrag.models import ContextualChunk
from contextrag.openai_utils import invoke_llm_parallel

logger = logging.getLogger(__name__)

_CONTEXT_PROMPT = """Here is the document:
<document>
{doc_excerpt}
</document>

Here is a specific chunk from that document:
<chunk>
{chunk_text}
</chunk>

Give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval. Answer only with the context, no preamble."""

_CACHE_FILE = DATABASE_DIR / "llm_context_cache.json"


def _load_cache() -> dict[str, str]:
    if _CACHE_FILE.exists():
        return json.loads(_CACHE_FILE.read_text())
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def generate_llm_contexts(
    ctx_chunks: list[ContextualChunk],
    doc_texts: dict[str, str],
    max_doc_excerpt_chars: int = 8000,
    batch_save_interval: int = 50,
) -> list[ContextualChunk]:
    """Generate LLM context descriptions for each chunk and update in place.

    Caches results to disk so subsequent runs skip already-processed chunks.
    Requires ``OPENAI_API_KEY`` environment variable.
    """
    cache = _load_cache()

    # Identify uncached chunks and build their prompts
    uncached: list[tuple[int, str]] = []  # (original index, chunk_id)
    prompts: list[str] = []

    for i, ctx in enumerate(ctx_chunks):
        chunk_id = ctx.chunk.chunk_id
        if chunk_id in cache:
            ctx.llm_context = cache[chunk_id]
            ctx.contextual_embedding_text = f"{ctx.llm_context}\n\n{ctx.chunk.text}"
        else:
            full_text = doc_texts.get(ctx.chunk.doc_file_name, "")
            start = max(0, ctx.chunk.char_offset_start - max_doc_excerpt_chars // 2)
            end = min(len(full_text), ctx.chunk.char_offset_end + max_doc_excerpt_chars // 2)
            doc_excerpt = full_text[start:end]
            prompts.append(_CONTEXT_PROMPT.format(doc_excerpt=doc_excerpt, chunk_text=ctx.chunk.text))
            uncached.append((i, chunk_id))

    if not uncached:
        logger.info("LLM context generation complete: 0 new, %d cached", len(ctx_chunks))
        return ctx_chunks

    logger.info("Generating LLM contexts for %d uncached chunks...", len(uncached))
    responses = invoke_llm_parallel(prompts, max_tokens=150, desc="LLM contexts")

    for (orig_idx, chunk_id), context_desc in zip(uncached, responses):
        ctx = ctx_chunks[orig_idx]
        ctx.llm_context = context_desc
        ctx.contextual_embedding_text = f"{context_desc}\n\n{ctx.chunk.text}" if context_desc else ctx.chunk.text
        cache[chunk_id] = context_desc

    _save_cache(cache)
    logger.info(
        "LLM context generation complete: %d new, %d cached",
        len(uncached), len(ctx_chunks) - len(uncached),
    )
    return ctx_chunks
