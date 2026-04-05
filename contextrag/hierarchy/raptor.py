"""RAPTOR-style hierarchical retrieval: build summary tree, query across levels."""

from __future__ import annotations

import logging
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from contextrag.config import (
    CHROMA_BATCH_SIZE,
    DATABASE_DIR,
    EMBEDDING_MODEL,
    RAPTOR_CLUSTER_DIVISOR,
    RAPTOR_MAX_LEVELS,
    RAPTOR_MIN_NODES_TO_CLUSTER,
    RAPTOR_SUMMARY_MAX_TOKENS,
)
from contextrag.hierarchy.clustering import cluster_embeddings, compute_n_clusters
from contextrag.models import ContextualChunk, HierarchyNode, RetrievalHit
from contextrag.openai_utils import batch_for_embed, embed_texts, embed_query, invoke_llm_parallel

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """Summarize the following text passages into a single coherent paragraph that captures the key information. Focus on entities, numbers, and facts.

Passages:
{passages}

Summary:"""


def build_raptor_tree(
    ctx_chunks: list[ContextualChunk],
    persist_dir: Path | None = None,
    collection_name: str = "contextrag_raptor",
    max_levels: int = RAPTOR_MAX_LEVELS,
    min_nodes_to_cluster: int = RAPTOR_MIN_NODES_TO_CLUSTER,
) -> list[HierarchyNode]:
    """Build a RAPTOR summary tree from chunks.

    Level 0: original chunks (leaf nodes).
    Level 1+: K-means cluster → LLM summarize → embed → repeat.
    Stops when a level has <= min_nodes_to_cluster or max_levels reached.

    Checkpoints embeddings per level to database/contextrag/raptor_checkpoints/
    so that if the process crashes, it can resume without re-embedding.
    """
    checkpoint_dir = DATABASE_DIR / "raptor_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    all_nodes: list[HierarchyNode] = []

    # Level 0: leaf nodes
    current_texts: list[str] = []
    current_node_ids: list[str] = []
    current_source_chunks: list[list[str]] = []
    current_doc_names: list[list[str]] = []

    for ctx in ctx_chunks:
        node_id = f"level_0_{ctx.chunk.chunk_id}"
        node = HierarchyNode(
            node_id=node_id,
            level=0,
            summary_text=ctx.chunk.text,
            child_ids=[],
            source_chunk_ids=[ctx.chunk.chunk_id],
            doc_file_names=[ctx.chunk.doc_file_name],
        )
        all_nodes.append(node)
        current_texts.append(ctx.chunk.text)
        current_node_ids.append(node_id)
        current_source_chunks.append([ctx.chunk.chunk_id])
        current_doc_names.append([ctx.chunk.doc_file_name])

    logger.info("RAPTOR Level 0: %d leaf nodes", len(current_texts))

    # Build higher levels
    for level in range(1, max_levels + 1):
        if len(current_texts) <= min_nodes_to_cluster:
            logger.info("RAPTOR stopping: level %d has only %d nodes", level - 1, len(current_texts))
            break

        # Check for embedding checkpoint
        emb_ckpt = checkpoint_dir / f"level_{level}_embeddings.pkl"
        if emb_ckpt.exists():
            logger.info("Loading embedding checkpoint for level %d from %s", level, emb_ckpt)
            with open(emb_ckpt, "rb") as f:
                level_embeddings = pickle.load(f)
            logger.info("Loaded %d embeddings from checkpoint", len(level_embeddings))
        else:
            level_embeddings = None

        if level_embeddings is None:
            logger.info("RAPTOR Level %d: embedding %d texts...", level, len(current_texts))
            batches = batch_for_embed(current_texts)
            all_embeds: list[list[float]] = []
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(embed_texts, batch, EMBEDDING_MODEL, f"Level {level} embeddings"): idx
                    for idx, batch in enumerate(batches)
                }
                batch_results: dict[int, list[list[float]]] = {}
                for future in tqdm(as_completed(futures), total=len(batches),
                                   desc=f"Level {level} embeddings", unit="batch"):
                    idx = futures[future]
                    batch_results[idx] = future.result()
            for idx in range(len(batches)):
                all_embeds.extend(batch_results[idx])
            level_embeddings = np.array(all_embeds)

            # Save checkpoint
            with open(emb_ckpt, "wb") as f:
                pickle.dump(level_embeddings, f)
            logger.info("Saved embedding checkpoint to %s", emb_ckpt)

        # Cluster
        n_clusters = compute_n_clusters(len(current_texts), RAPTOR_CLUSTER_DIVISOR)
        clusters = cluster_embeddings(level_embeddings, n_clusters)

        # Build prompts and summarize all clusters in parallel
        prompts = [
            _SUMMARY_PROMPT.format(
                passages="\n---\n".join(current_texts[i][:2000] for i in members)[:8000]
            )
            for members in clusters
        ]
        summaries_list = invoke_llm_parallel(
            prompts,
            max_tokens=RAPTOR_SUMMARY_MAX_TOKENS,
            desc=f"Level {level} summarization",
        )
        # Fallback: replace empty responses with truncated member text
        for idx, (summary, members) in enumerate(zip(summaries_list, clusters)):
            if not summary:
                summaries_list[idx] = " ".join(current_texts[i][:500] for i in members)[:1000]

        next_texts: list[str] = []
        next_node_ids: list[str] = []
        next_source_chunks: list[list[str]] = []
        next_doc_names: list[list[str]] = []

        for cluster_idx, member_indices in enumerate(clusters):
            summary = summaries_list[cluster_idx]

            # Gather child info
            child_ids = [current_node_ids[i] for i in member_indices]
            source_ids: list[str] = []
            doc_names: set[str] = set()
            for i in member_indices:
                source_ids.extend(current_source_chunks[i])
                doc_names.update(current_doc_names[i])

            node_id = f"level_{level}_cluster_{cluster_idx}"
            node = HierarchyNode(
                node_id=node_id,
                level=level,
                summary_text=summary,
                child_ids=child_ids,
                source_chunk_ids=source_ids,
                doc_file_names=sorted(doc_names),
            )
            all_nodes.append(node)

            next_texts.append(summary)
            next_node_ids.append(node_id)
            next_source_chunks.append(source_ids)
            next_doc_names.append(sorted(doc_names))

        current_texts = next_texts
        current_node_ids = next_node_ids
        current_source_chunks = next_source_chunks
        current_doc_names = next_doc_names

        logger.info("RAPTOR Level %d: %d summary nodes", level, len(current_texts))

    logger.info("RAPTOR tree complete: %d total nodes", len(all_nodes))
    return all_nodes


def index_raptor_nodes(
    nodes: list[HierarchyNode],
    persist_dir: Path,
    collection_name: str,
    embedding_model: str = EMBEDDING_MODEL,
    force_rebuild: bool = False,
) -> None:
    """Store RAPTOR nodes (level > 0) in ChromaDB for retrieval."""
    import chromadb

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    if force_rebuild:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0 and not force_rebuild:
        logger.info("RAPTOR collection already has %d items, skipping", collection.count())
        return

    # Only index summary nodes (level > 0); level 0 is already in the conditioned index
    summary_nodes = [n for n in nodes if n.level > 0]
    if not summary_nodes:
        logger.info("No summary nodes to index")
        return

    # Embed summaries with OpenAI (parallel batches)
    texts = [n.summary_text for n in summary_nodes]
    all_embeddings: list[list[float]] = []
    batches = batch_for_embed(texts)
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(embed_texts, batch, embedding_model, "Indexing RAPTOR nodes"): idx
            for idx, batch in enumerate(batches)
        }
        batch_results: dict[int, list[list[float]]] = {}
        for future in tqdm(as_completed(futures), total=len(batches),
                           desc="Indexing RAPTOR nodes", unit="batch"):
            idx = futures[future]
            batch_results[idx] = future.result()
    for idx in range(len(batches)):
        all_embeddings.extend(batch_results[idx])

    for i in range(0, len(summary_nodes), CHROMA_BATCH_SIZE):
        batch_nodes = summary_nodes[i : i + CHROMA_BATCH_SIZE]
        batch_embeddings = all_embeddings[i : i + CHROMA_BATCH_SIZE]

        collection.upsert(
            ids=[n.node_id for n in batch_nodes],
            documents=[n.summary_text for n in batch_nodes],
            embeddings=batch_embeddings,
            metadatas=[
                {
                    "level": n.level,
                    "source_chunk_count": len(n.source_chunk_ids),
                    "doc_count": len(n.doc_file_names),
                }
                for n in batch_nodes
            ],
        )

    logger.info("Indexed %d RAPTOR summary nodes", len(summary_nodes))


def query_raptor(
    query: str,
    persist_dir: Path,
    collection_name: str,
    embedding_model: str = EMBEDDING_MODEL,
    top_k_per_level: int = 10,
    max_level: int = RAPTOR_MAX_LEVELS,
) -> list[RetrievalHit]:
    """Query RAPTOR index across all levels."""
    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        logger.warning("RAPTOR collection %s not found", collection_name)
        return []

    if collection.count() == 0:
        return []

    query_embedding = embed_query(query, model=embedding_model)

    hits: list[RetrievalHit] = []

    for level in range(1, max_level + 1):
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k_per_level,
            where={"level": level},
            include=["documents", "distances", "metadatas"],
        )

        if results["ids"] and results["ids"][0]:
            for rank, (nid, doc, dist) in enumerate(
                zip(results["ids"][0], results["documents"][0], results["distances"][0]),
                start=1,
            ):
                score = 1.0 - dist
                hits.append(
                    RetrievalHit(
                        chunk_id=nid,
                        signal_name=f"raptor_L{level}",
                        rank=rank,
                        score=score,
                        text_preview=doc[:200] if doc else "",
                    )
                )

    logger.info("RAPTOR query returned %d hits across levels", len(hits))
    return hits


class RaptorBackend:
    """K-Means-based RAPTOR implementation."""

    def build_tree(self, ctx_chunks: list[ContextualChunk], **kwargs: Any) -> list[HierarchyNode]:
        return build_raptor_tree(ctx_chunks, **kwargs)

    def index_nodes(
        self, nodes: list[HierarchyNode], persist_dir: Path, collection_name: str, **kwargs: Any,
    ) -> None:
        index_raptor_nodes(nodes, persist_dir, collection_name, **kwargs)

    def query(
        self, query: str, persist_dir: Path, collection_name: str, **kwargs: Any,
    ) -> list[RetrievalHit]:
        return query_raptor(query, persist_dir, collection_name, **kwargs)
