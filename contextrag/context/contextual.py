"""Anthropic-style contextual retrieval: LLM-generated chunk context descriptions."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from contextrag.config import DATABASE_DIR, LLM_MODEL
from contextrag.models import ContextualChunk

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
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, max_tokens=150)
    cache = _load_cache()
    new_count = 0

    for i, ctx in enumerate(ctx_chunks):
        chunk_id = ctx.chunk.chunk_id

        # Use cache if available
        if chunk_id in cache:
            ctx.llm_context = cache[chunk_id]
            ctx.contextual_embedding_text = f"{ctx.llm_context}\n\n{ctx.chunk.text}"
            continue

        # Build prompt with truncated document excerpt
        full_text = doc_texts.get(ctx.chunk.doc_file_name, "")
        # Use text around the chunk for context
        start = max(0, ctx.chunk.char_offset_start - max_doc_excerpt_chars // 2)
        end = min(len(full_text), ctx.chunk.char_offset_end + max_doc_excerpt_chars // 2)
        doc_excerpt = full_text[start:end]

        prompt = _CONTEXT_PROMPT.format(
            doc_excerpt=doc_excerpt,
            chunk_text=ctx.chunk.text,
        )

        try:
            response = llm.invoke(prompt)
            context_desc = response.content.strip()
        except Exception as e:
            logger.warning("LLM context generation failed for %s: %s", chunk_id, e)
            context_desc = ""

        ctx.llm_context = context_desc
        ctx.contextual_embedding_text = f"{context_desc}\n\n{ctx.chunk.text}" if context_desc else ctx.chunk.text
        cache[chunk_id] = context_desc
        new_count += 1

        # Periodic save
        if new_count % batch_save_interval == 0:
            _save_cache(cache)
            logger.info("Generated %d/%d LLM contexts (saved cache)", i + 1, len(ctx_chunks))

    # Final save
    if new_count > 0:
        _save_cache(cache)

    logger.info(
        "LLM context generation complete: %d new, %d cached",
        new_count, len(ctx_chunks) - new_count,
    )
    return ctx_chunks
