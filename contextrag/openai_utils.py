"""Thin wrappers around the OpenAI client for parallel embedding and LLM calls.

Bypasses LangChain's internal re-batching and token-counting overhead.
Both functions use ThreadPoolExecutor so callers stay simple.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar

from tqdm import tqdm

from contextrag.config import EMBEDDING_MODEL, LLM_MODEL

logger = logging.getLogger(__name__)

# OpenAI embedding API limits
_MAX_INPUTS_PER_REQUEST = 2048
_MAX_TOKENS_PER_INPUT = 8192
_MAX_TOKENS_PER_REQUEST = 300_000

T = TypeVar("T")


def _count_tokens(text: str) -> int:
    """Token count for a string using the cl100k_base encoding (used by all OpenAI embedding models)."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def batch_for_embed(texts: list[str]) -> list[list[str]]:
    """Split ``texts`` into batches that each satisfy all OpenAI embedding API limits:
    - at most ``_MAX_INPUTS_PER_REQUEST`` texts per batch
    - at most ``_MAX_TOKENS_PER_REQUEST`` tokens summed per batch
    - each text must be at most ``_MAX_TOKENS_PER_INPUT`` tokens (raises if violated)

    Returns a list of batches, each ready to pass directly to ``embed_texts``.
    """
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    token_counts = [len(enc.encode(t)) for t in texts]

    for i, (text, count) in enumerate(zip(texts, token_counts)):
        if count > _MAX_TOKENS_PER_INPUT:
            raise ValueError(
                f"Text at index {i} has {count} tokens, exceeding the per-input limit "
                f"of {_MAX_TOKENS_PER_INPUT}. Truncate before embedding."
            )

    batches: list[list[str]] = []
    current_batch: list[str] = []
    current_tokens = 0

    for text, count in zip(texts, token_counts):
        if (
            current_batch
            and (
                len(current_batch) >= _MAX_INPUTS_PER_REQUEST
                or current_tokens + count > _MAX_TOKENS_PER_REQUEST
            )
        ):
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(text)
        current_tokens += count

    if current_batch:
        batches.append(current_batch)

    return batches


def run_parallel(
    items: list[T],
    fn: Callable[[T], str],
    desc: str = "Processing",
) -> list[str]:
    """Run ``fn`` over each item in parallel via ThreadPoolExecutor.

    Returns results in the same order as ``items``.
    On per-item failure the exception propagates from ``future.result()``.
    """
    results: dict[int, str] = {}
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(fn, item): i for i, item in enumerate(items)}
        for future in tqdm(as_completed(futures), total=len(items), desc=desc, unit="item"):
            i = futures[future]
            results[i] = future.result()
    return [results[i] for i in range(len(items))]


def embed_texts(
    texts: list[str],
    model: str = EMBEDDING_MODEL,
    desc: str = "Embedding",
) -> list[list[float]]:
    """Embed a list of texts in a single OpenAI API call.

    Validates that the batch satisfies all API limits (input count, per-text
    token count, total token count). Raises ``ValueError`` on violation.
    Call ``batch_for_embed`` first when ``texts`` may exceed these limits.
    Returns embeddings in the same order as ``texts``.
    """
    if len(texts) > _MAX_INPUTS_PER_REQUEST:
        raise ValueError(
            f"embed_texts received {len(texts)} texts; max per request is {_MAX_INPUTS_PER_REQUEST}. "
            "Use batch_for_embed() to split first."
        )

    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    total_tokens = 0
    for i, text in enumerate(texts):
        count = len(enc.encode(text))
        if count > _MAX_TOKENS_PER_INPUT:
            raise ValueError(
                f"Text at index {i} has {count} tokens, exceeding the per-input limit "
                f"of {_MAX_TOKENS_PER_INPUT}. Truncate before embedding."
            )
        total_tokens += count
    if total_tokens > _MAX_TOKENS_PER_REQUEST:
        raise ValueError(
            f"Batch total is {total_tokens} tokens, exceeding the per-request limit "
            f"of {_MAX_TOKENS_PER_REQUEST}. Use batch_for_embed() to split first."
        )

    from openai import OpenAI
    client = OpenAI()
    logger.debug("%s: embedding %d texts (%d tokens)", desc, len(texts), total_tokens)
    response = client.embeddings.create(input=texts, model=model)
    return [d.embedding for d in response.data]


def embed_query(query: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """Embed a single query string. Returns the embedding vector."""
    return embed_texts([query], model=model, desc="Query embedding")[0]


def invoke_llm_parallel(
    prompts: list[str],
    model: str = LLM_MODEL,
    max_tokens: int = 512,
    temperature: float = 0,
    desc: str = "LLM calls",
) -> list[str]:
    """Call the OpenAI chat completions API for each prompt in parallel.

    Returns responses in the same order as ``prompts``.
    On per-prompt failure, returns an empty string and logs a warning.
    """
    from openai import OpenAI
    client = OpenAI()

    def _call(prompt: str) -> str:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return ""

    return run_parallel(prompts, _call, desc=desc)
