"""RAPTOR backend abstraction: choose between custom (K-Means) and official (UMAP+GMM)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from contextrag.models import ContextualChunk, HierarchyNode, RetrievalHit


class CustomRaptorBackend:
    """Wraps the existing K-Means-based RAPTOR implementation."""

    def build_tree(self, ctx_chunks: list[ContextualChunk], **kwargs: Any) -> list[HierarchyNode]:
        from contextrag.hierarchy.raptor import build_raptor_tree
        return build_raptor_tree(ctx_chunks, **kwargs)

    def index_nodes(
        self, nodes: list[HierarchyNode], persist_dir: Path, collection_name: str, **kwargs: Any,
    ) -> None:
        from contextrag.hierarchy.raptor import index_raptor_nodes
        index_raptor_nodes(nodes, persist_dir, collection_name, **kwargs)

    def query(
        self, query: str, persist_dir: Path, collection_name: str, **kwargs: Any,
    ) -> list[RetrievalHit]:
        from contextrag.hierarchy.raptor import query_raptor
        return query_raptor(query, persist_dir, collection_name, **kwargs)


class OfficialRaptorBackend:
    """Uses UMAP+GMM soft clustering and collapsed-tree retrieval (ICLR 2024 paper)."""

    def build_tree(self, ctx_chunks: list[ContextualChunk], **kwargs: Any) -> list[HierarchyNode]:
        from contextrag.hierarchy.official_raptor import build_raptor_tree_official
        return build_raptor_tree_official(ctx_chunks, **kwargs)

    def index_nodes(
        self, nodes: list[HierarchyNode], persist_dir: Path, collection_name: str, **kwargs: Any,
    ) -> None:
        # Still index in ChromaDB for compatibility / status checks
        from contextrag.hierarchy.raptor import index_raptor_nodes
        index_raptor_nodes(nodes, persist_dir, collection_name, **kwargs)

    def query(
        self, query: str, persist_dir: Path, collection_name: str, **kwargs: Any,
    ) -> list[RetrievalHit]:
        from contextrag.hierarchy.official_raptor import query_raptor_official
        from contextrag.index.persistence import load_raptor_tree
        from contextrag.config import DATABASE_DIR

        nodes = load_raptor_tree(DATABASE_DIR / "raptor_tree.json")
        return query_raptor_official(query, nodes, **kwargs)


def get_raptor_backend(name: str = "custom") -> CustomRaptorBackend | OfficialRaptorBackend:
    """Factory: return the requested RAPTOR backend."""
    if name == "custom":
        return CustomRaptorBackend()
    if name == "official":
        return OfficialRaptorBackend()
    raise ValueError(f"Unknown RAPTOR backend: {name!r}. Choose 'custom' or 'official'.")
