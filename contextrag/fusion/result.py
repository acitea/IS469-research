"""Assemble final fused results with full provenance."""

from __future__ import annotations

from typing import Any

from contextrag.models import (
    ContextualChunk,
    FusedResult,
    HierarchyNode,
    RetrievalHit,
)


def assemble_results(
    fused: list[tuple[str, float, list[RetrievalHit]]],
    chunks_by_id: dict[str, ContextualChunk],
    hierarchy_nodes: dict[str, HierarchyNode] | None = None,
    top_n: int = 10,
) -> list[FusedResult]:
    """Hydrate fused rankings into full FusedResult objects.

    Args:
        fused: Output of ``reciprocal_rank_fusion``.
        chunks_by_id: Mapping from chunk_id to ``ContextualChunk``.
        hierarchy_nodes: Mapping from node_id to ``HierarchyNode`` (for RAPTOR hits). 
                         If None/empty, RAPTOR hydration is skipped.
        top_n: Number of results to return.
    """
    hierarchy_nodes = hierarchy_nodes or {}
    results: list[FusedResult] = []

    for rank, (chunk_id, fused_score, contributions) in enumerate(fused[:top_n], start=1):
        # Try to hydrate from chunks
        ctx = chunks_by_id.get(chunk_id)
        if ctx:
            results.append(
                FusedResult(
                    rank=rank,
                    chunk_id=chunk_id,
                    text=ctx.chunk.text,
                    doc_file_name=ctx.chunk.doc_file_name,
                    section_title=ctx.chunk.section_title,
                    preceding_context=ctx.preceding_context[:500] if ctx.preceding_context else "",
                    llm_context=ctx.llm_context,
                    fused_score=fused_score,
                    signal_contributions=contributions,
                    hierarchy_level=0,
                    source_chunk_ids=[chunk_id],
                )
            )
            continue

        # Try RAPTOR hierarchy node (only if hierarchy nodes were provided)
        if hierarchy_nodes:
            node = hierarchy_nodes.get(chunk_id)
            if node:
                results.append(
                    FusedResult(
                        rank=rank,
                        chunk_id=chunk_id,
                        text=node.summary_text,
                        doc_file_name=", ".join(node.doc_file_names[:3]),
                        section_title=f"RAPTOR Level {node.level} Summary",
                        preceding_context="",
                        llm_context="",
                        fused_score=fused_score,
                        signal_contributions=contributions,
                        hierarchy_level=node.level,
                        source_chunk_ids=node.source_chunk_ids,
                    )
                )
                continue

        # Fallback: minimal result
        text_preview = ""
        for hit in contributions:
            if hit.text_preview:
                text_preview = hit.text_preview
                break

        results.append(
            FusedResult(
                rank=rank,
                chunk_id=chunk_id,
                text=text_preview,
                doc_file_name="unknown",
                section_title="unknown",
                preceding_context="",
                llm_context="",
                fused_score=fused_score,
                signal_contributions=contributions,
            )
        )

    return results
