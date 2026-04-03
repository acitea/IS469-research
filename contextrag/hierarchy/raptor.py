"""RAPTOR-style hierarchical retrieval: build summary tree, query across levels."""

from __future__ import annotations

import logging
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
from tqdm import tqdm

from contextrag.config import (
    CHROMA_BATCH_SIZE,
    DATABASE_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL,
    RAPTOR_CLUSTER_DIVISOR,
    RAPTOR_MAX_LEVELS,
    RAPTOR_MIN_NODES_TO_CLUSTER,
    RAPTOR_SUMMARY_MAX_TOKENS,
)
from contextrag.hierarchy.clustering import cluster_embeddings, compute_n_clusters
from contextrag.models import ContextualChunk, HierarchyNode, RetrievalHit

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
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, max_tokens=RAPTOR_SUMMARY_MAX_TOKENS)
    embed_batch_size = 2048
    openai_client = OpenAI()
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
            # Embed current level texts (parallel batches)
            logger.info("RAPTOR Level %d: embedding %d texts...", level, len(current_texts))
            batches = [
                (i, current_texts[i : i + embed_batch_size])
                for i in range(0, len(current_texts), embed_batch_size)
            ]

            def _embed_batch(start: int, texts: list[str]) -> tuple[int, list[list[float]]]:
                response = openai_client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
                return start, [d.embedding for d in response.data]

            batch_results: dict[int, list[list[float]]] = {}
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(_embed_batch, start, texts): start for start, texts in batches}
                for future in tqdm(as_completed(futures), total=len(batches),
                                   desc=f"Level {level} embeddings", unit="batch"):
                    start, embeds = future.result()
                    batch_results[start] = embeds

            all_embeds: list[list[float]] = []
            for start, _ in batches:
                all_embeds.extend(batch_results[start])
            level_embeddings = np.array(all_embeds)

            # Save checkpoint
            with open(emb_ckpt, "wb") as f:
                pickle.dump(level_embeddings, f)
            logger.info("Saved embedding checkpoint to %s", emb_ckpt)

        # Cluster
        n_clusters = compute_n_clusters(len(current_texts), RAPTOR_CLUSTER_DIVISOR)
        clusters = cluster_embeddings(level_embeddings, n_clusters)

        # Summarize each cluster (parallel LLM calls via ThreadPoolExecutor)
        def _summarize_cluster(cluster_idx: int, member_indices: list[int]) -> tuple[int, str]:
            member_texts = [current_texts[i][:2000] for i in member_indices]
            passages = "\n---\n".join(member_texts)
            try:
                prompt = _SUMMARY_PROMPT.format(passages=passages[:8000])
                response = llm.invoke(prompt)
                return cluster_idx, response.content.strip()
            except Exception as e:
                logger.warning("RAPTOR summarization failed for cluster %d: %s", cluster_idx, e)
                return cluster_idx, " ".join(member_texts)[:1000]

        summaries: dict[int, str] = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(_summarize_cluster, idx, members): idx
                for idx, members in enumerate(clusters)
            }
            for future in tqdm(as_completed(futures), total=len(clusters),
                               desc=f"Level {level} summarization", unit="cluster"):
                idx, summary = future.result()
                summaries[idx] = summary

        next_texts: list[str] = []
        next_node_ids: list[str] = []
        next_source_chunks: list[list[str]] = []
        next_doc_names: list[list[str]] = []

        for cluster_idx, member_indices in enumerate(clusters):
            summary = summaries[cluster_idx]

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

    # Embed summaries with OpenAI
    embeddings_model = OpenAIEmbeddings(model=embedding_model)
    texts = [n.summary_text for n in summary_nodes]

    for i in tqdm(range(0, len(summary_nodes), CHROMA_BATCH_SIZE),
                   desc="Indexing RAPTOR nodes", unit="batch"):
        batch_nodes = summary_nodes[i : i + CHROMA_BATCH_SIZE]
        batch_texts = texts[i : i + CHROMA_BATCH_SIZE]
        batch_embeddings = embeddings_model.embed_documents(batch_texts)

        collection.upsert(
            ids=[n.node_id for n in batch_nodes],
            documents=batch_texts,
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

    embeddings_model = OpenAIEmbeddings(model=embedding_model)
    query_embedding = embeddings_model.embed_query(query)

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
