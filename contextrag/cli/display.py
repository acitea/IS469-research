"""Formatted terminal output for retrieval results."""

from __future__ import annotations

from contextrag.models import FusedResult


def print_results(results: list[FusedResult], verbose: bool = False) -> None:
    """Print fused retrieval results to the terminal."""
    if not results:
        print("No results found.")
        return

    print(f"\n{'='*80}")
    print(f" RETRIEVAL RESULTS ({len(results)} results)")
    print(f"{'='*80}")

    for r in results:
        print(f"\n[{r.rank}] Score: {r.fused_score:.6f}  |  {r.doc_file_name}")
        print(f"    Section: {r.section_title}")
        print(f"    Chunk ID: {r.chunk_id}")

        if r.hierarchy_level > 0:
            print(f"    RAPTOR Level: {r.hierarchy_level} (covers {len(r.source_chunk_ids)} source chunks)")

        # Signal breakdown
        signals = {}
        for hit in r.signal_contributions:
            signals[hit.signal_name] = f"rank={hit.rank} score={hit.score:.4f}"
        if signals:
            signal_str = " | ".join(f"{k}: {v}" for k, v in signals.items())
            print(f"    Signals: {signal_str}")

        # Text preview
        text = r.text
        if not verbose:
            text = text[:300] + ("..." if len(text) > 300 else "")
        print(f"    Text: {text}")

        if verbose:
            if r.llm_context:
                print(f"    LLM Context: {r.llm_context}")
            if r.preceding_context:
                ctx_preview = r.preceding_context[:200] + "..." if len(r.preceding_context) > 200 else r.preceding_context
                print(f"    Preceding Context: {ctx_preview}")

    print(f"\n{'='*80}")


def print_index_status(status: dict[str, str | int]) -> None:
    """Print index status summary."""
    print(f"\n{'='*60}")
    print(" INDEX STATUS")
    print(f"{'='*60}")
    for key, value in status.items():
        print(f"  {key:30s}: {value}")
    print(f"{'='*60}")
