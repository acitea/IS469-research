"""Persistence utilities: index existence checks, serialization helpers."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

from contextrag.models import ContextualChunk, HierarchyNode

logger = logging.getLogger(__name__)


def index_exists(persist_dir: Path, collection_name: str) -> bool:
    """Check whether a ChromaDB collection exists and has data."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(persist_dir))
        col = client.get_collection(collection_name)
        return col.count() > 0
    except Exception:
        return False


def save_chunks(chunks: list[ContextualChunk], path: Path) -> None:
    """Serialize contextual chunks to JSON for later hydration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for ctx in chunks:
        c = ctx.chunk
        data.append({
            "chunk_id": c.chunk_id,
            "doc_file_name": c.doc_file_name,
            "section_title": c.section_title,
            "chunk_index": c.chunk_index,
            "text": c.text,
            "char_offset_start": c.char_offset_start,
            "char_offset_end": c.char_offset_end,
            "preceding_context": ctx.preceding_context[:1000],  # Truncate for storage
            "llm_context": ctx.llm_context,
        })
    path.write_text(json.dumps(data, indent=2))
    logger.info("Saved %d chunks to %s", len(data), path)


def load_chunks(path: Path) -> list[dict[str, Any]]:
    """Load serialized chunk data."""
    if not path.exists():
        return []
    return json.loads(path.read_text())


def save_raptor_tree(nodes: list[HierarchyNode], path: Path) -> None:
    """Save RAPTOR hierarchy tree to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for node in nodes:
        data.append({
            "node_id": node.node_id,
            "level": node.level,
            "summary_text": node.summary_text,
            "child_ids": node.child_ids,
            "source_chunk_ids": node.source_chunk_ids,
            "doc_file_names": node.doc_file_names,
        })
    path.write_text(json.dumps(data, indent=2))
    logger.info("Saved %d RAPTOR nodes to %s", len(data), path)


def load_raptor_tree(path: Path) -> list[HierarchyNode]:
    """Load RAPTOR hierarchy tree from JSON."""
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [
        HierarchyNode(
            node_id=d["node_id"],
            level=d["level"],
            summary_text=d["summary_text"],
            child_ids=d.get("child_ids", []),
            source_chunk_ids=d.get("source_chunk_ids", []),
            doc_file_names=d.get("doc_file_names", []),
        )
        for d in data
    ]


def save_bm25(bm25_obj: Any, chunk_ids: list[str], path: Path) -> None:
    """Serialize BM25 index and chunk ID mapping to pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25_obj, "chunk_ids": chunk_ids}, f)
    logger.info("Saved BM25 index (%d chunks) to %s", len(chunk_ids), path)


def load_bm25(path: Path) -> tuple[Any, list[str]]:
    """Load serialized BM25 index."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunk_ids"]


def save_manifest(manifest: dict[str, Any], path: Path) -> None:
    """Save index manifest for incremental rebuild detection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def load_manifest(path: Path) -> dict[str, Any]:
    """Load index manifest."""
    if not path.exists():
        return {}
    return json.loads(path.read_text())
