"""CLI subcommands: index, query, demo, status."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from contextrag.config import (
    DENSE_COLLECTION,
    CONTEXTUAL_COLLECTION,
    DATABASE_DIR,
    DEFAULT_TOP_K,
    RAPTOR_COLLECTION,
    TEXTS_DIR,
    load_env,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


def cmd_index(args: argparse.Namespace) -> None:
    """Build retrieval indexes."""
    setup_logging()
    load_env()

    from contextrag.corpus.chunker import chunk_sections
    from contextrag.corpus.loader import load_corpus
    from contextrag.corpus.section_parser import parse_sections
    from contextrag.index.persistence import save_chunks, save_raptor_tree
    from contextrag.models import ContextualChunk

    ALL_INDEXES = {"dense", "contextual", "coil", "raptor"}
    indexes = set(args.indexes.split(",")) if args.indexes else ALL_INDEXES
    invalid = indexes - ALL_INDEXES
    if invalid:
        raise ValueError(f"Unknown indexes: {invalid}. Choose from: {sorted(ALL_INDEXES)}")

    texts_dir = Path(args.texts_dir)
    force = args.force_rebuild

    # Step 1: Load and chunk corpus (always needed)
    logger.info("Loading corpus from %s", texts_dir)
    doc_pairs = load_corpus(texts_dir)
    doc_texts = {meta.file_name: text for meta, text in doc_pairs}

    all_chunks = []
    for meta, text in doc_pairs:
        sections = parse_sections(meta, text)
        chunks = chunk_sections(meta, sections)
        all_chunks.extend(chunks)
    logger.info("Total chunks: %d", len(all_chunks))

    # Step 2: Dense embeddings (using specified embedder mode)
    if "dense" in indexes:
        from contextrag.nn.embed import embed_chunks
        from contextrag.index.dense_store import create_conditioned_index

        logger.info("Generating dense embeddings using %s mode...", args.embedder_mode)
        ctx_chunks, embeddings = embed_chunks(
            all_chunks, doc_texts,
            embedder_mode=args.embedder_mode,
        )

        logger.info("Indexing dense embeddings in ChromaDB...")
        dense_dir = DATABASE_DIR / "dense"
        create_conditioned_index(
            ctx_chunks, embeddings, dense_dir, DENSE_COLLECTION, force_rebuild=force,
        )
    else:
        # Wrap chunks with preceding context (no embedding inference)
        from contextrag.nn.embed import prepare_ctx_chunks
        ctx_chunks = prepare_ctx_chunks(all_chunks, doc_texts)

    # Step 3: LLM-based contextual retrieval
    if "contextual" in indexes:
        logger.info("Generating LLM context descriptions...")
        from contextrag.context.contextual import generate_llm_contexts
        from contextrag.index.dense_store import create_contextual_index

        ctx_chunks = generate_llm_contexts(ctx_chunks, doc_texts)

        logger.info("Indexing contextual embeddings in ChromaDB...")
        contextual_dir = DATABASE_DIR / "contextual"
        create_contextual_index(
            ctx_chunks, contextual_dir, CONTEXTUAL_COLLECTION, force_rebuild=force,
        )

    # Step 4: COIL/BM25 lexical index
    if "coil" in indexes:
        from contextrag.index.bm25_store import create_bm25_index

        logger.info("Building COIL/BM25 lexical index...")
        bm25_path = DATABASE_DIR / "bm25_coil.pkl"
        create_bm25_index(ctx_chunks, bm25_path, force_rebuild=force)

    # Step 5: RAPTOR hierarchy
    if "raptor" in indexes:
        logger.info("Building RAPTOR hierarchy...")
        from contextrag.hierarchy.raptor import RaptorBackend

        backend = RaptorBackend()
        raptor_nodes = backend.build_tree(ctx_chunks)
        save_raptor_tree(raptor_nodes, DATABASE_DIR / "raptor_tree.json")

        raptor_dir = DATABASE_DIR / "raptor"
        backend.index_nodes(raptor_nodes, raptor_dir, RAPTOR_COLLECTION, force_rebuild=force)

    # Step 6: Save chunk data for query-time hydration
    save_chunks(ctx_chunks, DATABASE_DIR / "chunks.json")

    built = ", ".join(sorted(indexes))
    print(f"\nIndexing complete ({built}). Data stored in {DATABASE_DIR}")


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


def cmd_query(args: argparse.Namespace) -> None:
    """Run a query against all indexes and display fused results."""
    setup_logging()
    load_env()

    import numpy as np

    from contextrag.cli.display import print_results
    from contextrag.fusion.ranker import reciprocal_rank_fusion
    from contextrag.fusion.result import assemble_results
    from contextrag.index.persistence import load_chunks, load_raptor_tree
    from contextrag.models import ContextualChunk, Chunk, HierarchyNode

    query = args.query
    top_k = args.top_k
    signals = set(args.signals.split(",")) if args.signals else {"dense", "contextual", "coil", "raptor"}

    # Load chunks for hydration
    chunk_data = load_chunks(DATABASE_DIR / "chunks.json")
    chunks_by_id: dict[str, ContextualChunk] = {}
    for d in chunk_data:
        chunk = Chunk(
            chunk_id=d["chunk_id"],
            doc_file_name=d["doc_file_name"],
            section_title=d["section_title"],
            chunk_index=d["chunk_index"],
            text=d["text"],
            char_offset_start=d["char_offset_start"],
            char_offset_end=d["char_offset_end"],
        )
        ctx = ContextualChunk(
            chunk=chunk,
            preceding_context=d.get("preceding_context", ""),
            llm_context=d.get("llm_context", ""),
        )
        chunks_by_id[chunk.chunk_id] = ctx

    # Collect hits from each signal
    all_hits: list[list] = []
    hierarchy_nodes: dict[str, HierarchyNode] | None = None

    if "dense" in signals:
        logger.info("Querying dense index...")
        from contextrag.retrieval.dense import retrieve_dense
        try:
            hits = retrieve_dense(query, top_k=top_k)
            all_hits.append(hits)
            logger.info("  Dense: %d hits", len(hits))
        except Exception as e:
            logger.warning("Dense index not available: %s", e)

    if "contextual" in signals:
        logger.info("Querying contextual dense index...")
        from contextrag.retrieval.dense import retrieve_contextual
        try:
            hits = retrieve_contextual(query, top_k=top_k)
            all_hits.append(hits)
            logger.info("  Contextual: %d hits", len(hits))
        except Exception as e:
            logger.warning("Contextual index not available: %s", e)

    if "coil" in signals:
        logger.info("Querying COIL/BM25 lexical index...")
        from contextrag.retrieval.lexical import retrieve_lexical
        hits = retrieve_lexical(query, top_k=top_k)
        all_hits.append(hits)
        logger.info("  COIL: %d hits", len(hits))

    if "raptor" in signals:
        logger.info("Querying RAPTOR hierarchical index...")
        from contextrag.retrieval.hierarchical import retrieve_hierarchical
        hits = retrieve_hierarchical(query)
        all_hits.append(hits)
        logger.info("  RAPTOR: %d hits", len(hits))
        # Load hierarchy nodes only if RAPTOR was queried
        raptor_nodes_list = load_raptor_tree(DATABASE_DIR / "raptor_tree.json")
        hierarchy_nodes = {n.node_id: n for n in raptor_nodes_list}

    # Fuse
    fused = reciprocal_rank_fusion(all_hits)


    results = assemble_results(fused, chunks_by_id, hierarchy_nodes, top_n=top_k)
    print_results(results, verbose=args.verbose)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> None:
    """Show index and model status."""
    from contextrag.cli.display import print_index_status
    from contextrag.index.persistence import index_exists

    status: dict[str, str | int] = {}

    # Indexes
    status["Dense index"] = "yes" if index_exists(DATABASE_DIR / "dense", DENSE_COLLECTION) else "no"
    status["Contextual index"] = "yes" if index_exists(DATABASE_DIR / "contextual", CONTEXTUAL_COLLECTION) else "no"
    status["BM25/COIL index"] = "yes" if (DATABASE_DIR / "bm25_coil.pkl").exists() else "no"
    status["RAPTOR index"] = "yes" if index_exists(DATABASE_DIR / "raptor", RAPTOR_COLLECTION) else "no"
    status["RAPTOR tree"] = "yes" if (DATABASE_DIR / "raptor_tree.json").exists() else "no"
    status["Chunks data"] = "yes" if (DATABASE_DIR / "chunks.json").exists() else "no"
    status["LLM context cache"] = "yes" if (DATABASE_DIR / "llm_context_cache.json").exists() else "no"

    print_index_status(status)
