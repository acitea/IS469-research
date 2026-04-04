"""Thin wrappers around the OpenAI client for parallel embedding and LLM calls.

Bypasses LangChain's internal re-batching and token-counting overhead.
Both functions use ThreadPoolExecutor so callers stay simple.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import TypeVar

from tqdm import tqdm

from contextrag.config import DATABASE_DIR, EMBEDDING_MODEL, LLM_MODEL

logger = logging.getLogger(__name__)

# OpenAI embedding API limits
_MAX_INPUTS_PER_REQUEST = 2048
_MAX_TOKENS_PER_INPUT = 8192
_MAX_TOKENS_PER_REQUEST = 100_000

# Proactive embedding TPM guard. Override via environment variables when needed.
_EMBEDDING_TPM_LIMIT = int(os.getenv("OPENAI_EMBEDDING_TPM_LIMIT", "4000000"))
_EMBEDDING_TPM_HEADROOM = int(os.getenv("OPENAI_EMBEDDING_TPM_HEADROOM", "200000"))
_EMBEDDING_TPM_WINDOW_SECONDS = 60.0
_EMBEDDING_TPM_EFFECTIVE_LIMIT = max(1, _EMBEDDING_TPM_LIMIT - _EMBEDDING_TPM_HEADROOM)

_EMBEDDING_DISK_CACHE_ENABLED = os.getenv("OPENAI_EMBEDDING_DISK_CACHE_ENABLED", "1").strip().lower() not in {
    "0", "false", "no", "off"
}
_EMBEDDING_DISK_CACHE_PATH = Path(
    os.getenv("OPENAI_EMBEDDING_CACHE_PATH", str(DATABASE_DIR / "embedding_cache.sqlite3"))
)

T = TypeVar("T")

_embed_tpm_lock = Lock()
_embed_tpm_events: deque[tuple[float, int]] = deque()
_embed_tpm_used = 0

_embedding_disk_cache_lock = Lock()
_embedding_disk_cache_ready = False


def _embedding_cache_key(model: str, text: str) -> tuple[str, int, str]:
    """Stable cache key for one model/text pair without retaining full text as key."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return model, len(text), digest


def _ensure_embedding_disk_cache() -> None:
    """Initialize on-disk embedding cache table if enabled."""
    global _embedding_disk_cache_ready
    if not _EMBEDDING_DISK_CACHE_ENABLED or _embedding_disk_cache_ready:
        return

    with _embedding_disk_cache_lock:
        if _embedding_disk_cache_ready:
            return
        _EMBEDDING_DISK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(_EMBEDDING_DISK_CACHE_PATH) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    model TEXT NOT NULL,
                    text_length INTEGER NOT NULL,
                    text_hash TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (model, text_length, text_hash)
                )
                """
            )
        _embedding_disk_cache_ready = True


def _embedding_cache_get_many(model: str, texts: list[str]) -> dict[str, list[float]]:
    """Read embeddings from persistent disk cache."""
    if not texts:
        return {}

    if not _EMBEDDING_DISK_CACHE_ENABLED:
        return {}

    _ensure_embedding_disk_cache()
    found: dict[str, list[float]] = {}

    with _embedding_disk_cache_lock:
        with sqlite3.connect(_EMBEDDING_DISK_CACHE_PATH) as conn:
            cursor = conn.cursor()
            for text in texts:
                model_key, text_length, text_hash = _embedding_cache_key(model, text)
                row = cursor.execute(
                    """
                    SELECT embedding_json
                    FROM embeddings
                    WHERE model = ? AND text_length = ? AND text_hash = ?
                    """,
                    (model_key, text_length, text_hash),
                ).fetchone()
                if row is None:
                    continue
                try:
                    embedding_raw = json.loads(row[0])
                    embedding = [float(v) for v in embedding_raw]
                except Exception:
                    continue
                found[text] = embedding

    return found


def _embedding_cache_set_many(model: str, values: dict[str, list[float]]) -> None:
    """Write embeddings to persistent disk cache."""
    if not values:
        return

    if not _EMBEDDING_DISK_CACHE_ENABLED:
        return

    _ensure_embedding_disk_cache()
    rows: list[tuple[str, int, str, str, float]] = []
    now = time.time()
    for text, embedding in values.items():
        model_key, text_length, text_hash = _embedding_cache_key(model, text)
        rows.append(
            (
                model_key,
                text_length,
                text_hash,
                json.dumps(embedding, separators=(",", ":")),
                now,
            )
        )

    with _embedding_disk_cache_lock:
        with sqlite3.connect(_EMBEDDING_DISK_CACHE_PATH) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embeddings (
                    model, text_length, text_hash, embedding_json, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )


def _prune_embed_tpm_window(now: float) -> None:
    """Drop token-usage events older than the rolling TPM window."""
    global _embed_tpm_used
    cutoff = now - _EMBEDDING_TPM_WINDOW_SECONDS
    while _embed_tpm_events and _embed_tpm_events[0][0] <= cutoff:
        _, tokens = _embed_tpm_events.popleft()
        _embed_tpm_used -= tokens
    if _embed_tpm_used < 0:
        _embed_tpm_used = 0


def _acquire_embed_tpm_budget(request_tokens: int, desc: str) -> None:
    """Block until enough embedding TPM budget is available in the rolling window."""
    global _embed_tpm_used
    if request_tokens <= 0:
        return

    if request_tokens > _EMBEDDING_TPM_EFFECTIVE_LIMIT:
        raise ValueError(
            f"Single embedding request needs {request_tokens} tokens, but effective TPM budget is "
            f"{_EMBEDDING_TPM_EFFECTIVE_LIMIT}. Lower batch size or adjust "
            "OPENAI_EMBEDDING_TPM_LIMIT/OPENAI_EMBEDDING_TPM_HEADROOM."
        )

    while True:
        now = time.monotonic()
        with _embed_tpm_lock:
            _prune_embed_tpm_window(now)
            available = _EMBEDDING_TPM_EFFECTIVE_LIMIT - _embed_tpm_used
            if request_tokens <= available:
                _embed_tpm_events.append((now, request_tokens))
                _embed_tpm_used += request_tokens
                return

            oldest_ts = _embed_tpm_events[0][0]
            wait_seconds = max(oldest_ts + _EMBEDDING_TPM_WINDOW_SECONDS - now, 0.05)
            used_snapshot = _embed_tpm_used

        logger.info(
            "%s: waiting %.2fs for embedding TPM budget (%d requested, %d used / %d limit)",
            desc,
            wait_seconds,
            request_tokens,
            used_snapshot,
            _EMBEDDING_TPM_LIMIT,
        )
        time.sleep(wait_seconds)


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
        if text == "":
            raise ValueError(
                f"Text at index {i} is an empty string, which is not allowed by the OpenAI embeddings API."
            )
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
    """Embed texts while automatically handling request-size limits.

    - validates per-input token limits
    - pre-splits by OpenAI documented per-request limits
    - proactively throttles embedding TPM to reduce rate-limit failures
    - deduplicates identical inputs and reuses persistent cache hits
    - persists embeddings to disk for reuse across runs

    Returns embeddings in the same order as ``texts``.
    """
    if not texts:
        return []

    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    total_tokens = 0
    text_to_count: dict[str, int] = {}
    unique_texts: list[str] = []
    for i, text in enumerate(texts):
        if text == "":
            raise ValueError(
                f"Text at index {i} is an empty string, which is not allowed by the OpenAI embeddings API."
            )
        count = len(enc.encode(text))
        if count > _MAX_TOKENS_PER_INPUT:
            raise ValueError(
                f"Text at index {i} has {count} tokens, exceeding the per-input limit "
                f"of {_MAX_TOKENS_PER_INPUT}. Truncate before embedding."
            )
        total_tokens += count
        if text not in text_to_count:
            text_to_count[text] = count
            unique_texts.append(text)

    embeddings_by_text: dict[str, list[float]] = {}
    cached_by_text = _embedding_cache_get_many(model, unique_texts)
    embeddings_by_text.update(cached_by_text)
    uncached_unique_texts = [text for text in unique_texts if text not in cached_by_text]
    cache_hits = len(cached_by_text)

    batches = batch_for_embed(uncached_unique_texts)

    from openai import OpenAI
    client = OpenAI()

    logger.debug(
        "%s: embedding %d texts (%d unique, %d cache hits) across %d batch(es) (%d tokens total)",
        desc,
        len(texts),
        len(unique_texts),
        cache_hits,
        len(batches),
        total_tokens,
    )

    for batch in batches:
        batch_tokens = sum(text_to_count[text] for text in batch)
        _acquire_embed_tpm_budget(batch_tokens, desc)
        response = client.embeddings.create(input=batch, model=model)
        if len(response.data) != len(batch):
            raise RuntimeError(
                f"Embedding response size mismatch: expected {len(batch)}, got {len(response.data)}"
            )
        fetched: dict[str, list[float]] = {}
        for text, embedded in zip(batch, response.data):
            embedding = embedded.embedding
            embeddings_by_text[text] = embedding
            fetched[text] = embedding
        _embedding_cache_set_many(model, fetched)

    return [embeddings_by_text[text] for text in texts]


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
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return ""

    return run_parallel(prompts, _call, desc=desc)
