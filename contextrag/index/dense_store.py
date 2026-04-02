"""ChromaDB wrapper for creating, loading, and querying dense indexes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from contextrag.config import CHROMA_BATCH_SIZE
from contextrag.models import ContextualChunk, RetrievalHit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom embedding function that wraps pre-computed numpy arrays
# ---------------------------------------------------------------------------


class PrecomputedEmbeddingFunction:
    """A ChromaDB-compatible embedding function backed by pre-computed vectors."""

    def __init__(self, dim: int = 768) -> None:
        self._dim = dim
        self._store: dict[str, list[float]] = {}

    def register(self, text: str, vector: np.ndarray) -> None:
        self._store[text] = vector.tolist()

    def __call__(self, input: list[str]) -> list[list[float]]:
        results = []
        for text in input:
            if text in self._store:
                results.append(self._store[text])
            else:
                # Return zero vector for unknown texts (shouldn't happen in practice)
                results.append([0.0] * self._dim)
        return results


# ---------------------------------------------------------------------------
# Conditioned index (custom model embeddings)
# ---------------------------------------------------------------------------


def create_conditioned_index(
    ctx_chunks: list[ContextualChunk],
    embeddings: np.ndarray,
    persist_dir: Path,
    collection_name: str,
    force_rebuild: bool = False,
) -> None:
    """Store context-conditioned embeddings in ChromaDB.

    Uses raw chromadb client for direct embedding injection.
    """
    import chromadb

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    if force_rebuild:
        try:
            client.delete_collection(collection_name)
            logger.info("Deleted existing collection: %s", collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0 and not force_rebuild:
        logger.info("Collection %s already has %d items, skipping", collection_name, collection.count())
        return

    # Batch upsert
    for i in range(0, len(ctx_chunks), CHROMA_BATCH_SIZE):
        batch_chunks = ctx_chunks[i : i + CHROMA_BATCH_SIZE]
        batch_embs = embeddings[i : i + CHROMA_BATCH_SIZE]

        ids = [c.chunk.chunk_id for c in batch_chunks]
        documents = [c.chunk.text for c in batch_chunks]
        metadatas = [
            {
                "doc_file_name": c.chunk.doc_file_name,
                "section_title": c.chunk.section_title,
                "chunk_index": c.chunk.chunk_index,
                "char_offset_start": c.chunk.char_offset_start,
                "char_offset_end": c.chunk.char_offset_end,
            }
            for c in batch_chunks
        ]

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=batch_embs.tolist(),
            metadatas=metadatas,
        )

    logger.info("Indexed %d chunks in %s", len(ctx_chunks), collection_name)


def query_conditioned_index(
    query_embedding: np.ndarray,
    persist_dir: Path,
    collection_name: str,
    top_k: int = 20,
) -> list[RetrievalHit]:
    """Query the conditioned ChromaDB index with a pre-computed query embedding."""
    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(collection_name)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[RetrievalHit] = []
    if results["ids"] and results["ids"][0]:
        for rank, (cid, doc, dist) in enumerate(
            zip(results["ids"][0], results["documents"][0], results["distances"][0]),
            start=1,
        ):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            score = 1.0 - dist  # Convert to similarity
            hits.append(
                RetrievalHit(
                    chunk_id=cid,
                    signal_name="conditioned",
                    rank=rank,
                    score=score,
                    text_preview=doc[:200] if doc else "",
                )
            )

    return hits


# ---------------------------------------------------------------------------
# Contextual index (OpenAI embeddings with LLM context)
# ---------------------------------------------------------------------------


def create_contextual_index(
    ctx_chunks: list[ContextualChunk],
    persist_dir: Path,
    collection_name: str,
    embedding_model: str = "text-embedding-3-small",
    force_rebuild: bool = False,
) -> Chroma:
    """Store Anthropic-style contextual embeddings using OpenAI embeddings."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    embeddings = OpenAIEmbeddings(model=embedding_model)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    existing_count = vector_store._collection.count()
    if existing_count > 0 and not force_rebuild:
        logger.info("Collection %s already has %d items, skipping", collection_name, existing_count)
        return vector_store

    if force_rebuild and existing_count > 0:
        vector_store._collection.delete(where={})
        logger.info("Cleared existing collection: %s", collection_name)

    # Build LangChain documents with contextual embedding text
    documents: list[Document] = []
    for ctx in ctx_chunks:
        text = ctx.contextual_embedding_text or ctx.chunk.text
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "chunk_id": ctx.chunk.chunk_id,
                    "doc_file_name": ctx.chunk.doc_file_name,
                    "section_title": ctx.chunk.section_title,
                    "chunk_index": ctx.chunk.chunk_index,
                    "original_text": ctx.chunk.text[:500],
                },
            )
        )

    # Batch add
    for i in range(0, len(documents), CHROMA_BATCH_SIZE):
        batch = documents[i : i + CHROMA_BATCH_SIZE]
        vector_store.add_documents(batch)
        logger.info("Added batch %d-%d to contextual index", i, i + len(batch))

    logger.info("Indexed %d chunks in contextual collection", len(documents))
    return vector_store


def query_contextual_index(
    query: str,
    persist_dir: Path,
    collection_name: str,
    embedding_model: str = "text-embedding-3-small",
    top_k: int = 20,
) -> list[RetrievalHit]:
    """Query the contextual (OpenAI-embedded) ChromaDB index."""
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    results = vector_store.similarity_search_with_score(query, k=top_k)
    hits: list[RetrievalHit] = []
    for rank, (doc, score) in enumerate(results, start=1):
        chunk_id = doc.metadata.get("chunk_id", f"unknown_{rank}")
        hits.append(
            RetrievalHit(
                chunk_id=chunk_id,
                signal_name="contextual",
                rank=rank,
                score=float(score),
                text_preview=doc.metadata.get("original_text", doc.page_content[:200]),
            )
        )

    return hits
