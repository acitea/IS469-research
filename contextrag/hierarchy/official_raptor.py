"""Official RAPTOR backend using the raptor package (ICLR 2024).

Uses the actual RAPTOR implementation from https://github.com/parthsarthi03/raptor:
  - UMAP + GMM soft clustering with BIC-optimal K
  - Recursive reclustering of oversized clusters
  - Collapsed-tree retrieval (flat search across ALL levels)

Installed as a local package from ./raptor.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from contextrag.config import (
    DATABASE_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL,
    RAPTOR_MAX_LEVELS,
    RAPTOR_OFFICIAL_TOP_K,
    RAPTOR_SUMMARY_MAX_TOKENS,
)
from contextrag.models import ContextualChunk, HierarchyNode, RetrievalHit
from raptor import (
    BaseEmbeddingModel,
    BaseSummarizationModel,
    RetrievalAugmentation,
    RetrievalAugmentationConfig,
    Tree,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom model adapters for the raptor raptor package
# ---------------------------------------------------------------------------


class _EmbeddingAdapter(BaseEmbeddingModel):
    """Wraps OpenAI text-embedding-3-small for the raptor raptor package."""

    def __init__(self, model: str = EMBEDDING_MODEL) -> None:
        self._client = OpenAI()
        self._model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text: str) -> list[float]:
        text = text.replace("\n", " ")
        return (
            self._client.embeddings.create(input=[text], model=self._model)
            .data[0]
            .embedding
        )


class _SummarizationAdapter(BaseSummarizationModel):
    """Wraps gpt-4o-mini for the raptor raptor package."""

    def __init__(self, model: str = LLM_MODEL, max_tokens: int = RAPTOR_SUMMARY_MAX_TOKENS) -> None:
        self._client = OpenAI()
        self._model = model
        self._max_tokens = max_tokens

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context: str, max_tokens: int = 150) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        "Write a summary of the following, including as many key details as possible. "
                        f"Focus on entities, numbers, and facts:\n\n{context}"
                    ),
                },
            ],
            max_tokens=max_tokens or self._max_tokens,
            temperature=0,
        )
        return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Shared config helper
# ---------------------------------------------------------------------------

_OFFICIAL_TREE_PICKLE = "official_raptor_tree.pkl"


def _build_ra_config(max_levels: int = RAPTOR_MAX_LEVELS) -> RetrievalAugmentationConfig:
    """Build a RetrievalAugmentationConfig with our custom model adapters."""
    return RetrievalAugmentationConfig(
        embedding_model=_EmbeddingAdapter(),
        summarization_model=_SummarizationAdapter(),
        tb_num_layers=max_levels,
        tb_max_tokens=100,
        tb_summarization_length=RAPTOR_SUMMARY_MAX_TOKENS,
        tr_top_k=RAPTOR_OFFICIAL_TOP_K,
    )


def _tree_path() -> Path:
    return DATABASE_DIR / _OFFICIAL_TREE_PICKLE


# ---------------------------------------------------------------------------
# Tree building
# ---------------------------------------------------------------------------


def build_raptor_tree_official(
    ctx_chunks: list[ContextualChunk],
    max_levels: int = RAPTOR_MAX_LEVELS,
    **kwargs: Any,
) -> list[HierarchyNode]:
    """Build a RAPTOR tree using the official raptor package (UMAP+GMM).

    Returns list[HierarchyNode] for compatibility with the rest of the pipeline.
    Persists the native Tree pickle for query-time retrieval via RA.retrieve().
    """
    ra = RetrievalAugmentation(config=_build_ra_config(max_levels))

    combined_text = "\n\n".join(ctx.chunk.text for ctx in ctx_chunks)
    logger.info("RAPTOR-Official: building tree from %d chunks...", len(ctx_chunks))
    ra.add_documents(combined_text)

    # Save using the official API (README: RA.save(path))
    path = _tree_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    ra.save(str(path))
    logger.info("Saved official RAPTOR tree to %s", path)

    # Convert to HierarchyNode for compatibility with our pipeline
    nodes = _convert_tree_to_hierarchy_nodes(ra.tree)
    logger.info("RAPTOR-Official tree complete: %d total nodes", len(nodes))
    return nodes


def _convert_tree_to_hierarchy_nodes(tree: Tree) -> list[HierarchyNode]:
    """Convert the official raptor Tree into our HierarchyNode format."""
    from raptor.utils import reverse_mapping

    node_to_layer = reverse_mapping(tree.layer_to_nodes)
    all_nodes: list[HierarchyNode] = []
    index_to_node_id: dict[int, str] = {}

    for node_idx, node in sorted(tree.all_nodes.items()):
        layer = node_to_layer.get(node_idx, 0)
        node_id = f"level_{layer}_official_{node_idx}"
        index_to_node_id[node_idx] = node_id

        child_ids = [index_to_node_id.get(c, f"node_{c}") for c in node.children]
        source_chunk_ids = (
            [node_id] if layer == 0
            else _get_all_leaf_ids(node, tree, node_to_layer, index_to_node_id)
        )

        all_nodes.append(HierarchyNode(
            node_id=node_id,
            level=layer,
            summary_text=node.text,
            child_ids=child_ids,
            source_chunk_ids=source_chunk_ids,
            doc_file_names=["official_raptor"],
        ))

    return all_nodes


def _get_all_leaf_ids(node, tree: Tree, node_to_layer: dict, index_to_node_id: dict) -> list[str]:
    """Recursively gather all leaf node IDs under a given node."""
    if not node.children:
        return [index_to_node_id.get(node.index, f"official_leaf_{node.index}")]
    leaf_ids: list[str] = []
    for child_idx in node.children:
        child_node = tree.all_nodes.get(child_idx)
        if child_node is None:
            continue
        if node_to_layer.get(child_idx, 0) == 0:
            leaf_ids.append(index_to_node_id.get(child_idx, f"official_leaf_{child_idx}"))
        else:
            leaf_ids.extend(_get_all_leaf_ids(child_node, tree, node_to_layer, index_to_node_id))
    return list(dict.fromkeys(leaf_ids))


# ---------------------------------------------------------------------------
# Collapsed-tree retrieval (follows the README pattern)
# ---------------------------------------------------------------------------


def query_raptor_official(
    query: str,
    nodes: list[HierarchyNode] | None = None,
    top_k: int = RAPTOR_OFFICIAL_TOP_K,
    **kwargs: Any,
) -> list[RetrievalHit]:
    """Collapsed-tree retrieval using the official raptor package.

    Follows the README pattern:
        RA = RetrievalAugmentation(config=config, tree=SAVE_PATH)
        context, layer_info = RA.retrieve(query, collapse_tree=True)
    """
    path = _tree_path()
    if not path.exists():
        logger.warning("No official RAPTOR tree found. Run indexing with --raptor-backend official first.")
        return []

    # Load tree + create retriever in one shot (README: RA = RetrievalAugmentation(tree=path))
    ra = RetrievalAugmentation(config=_build_ra_config(), tree=str(path))

    logger.info("RAPTOR-Official collapsed-tree retrieval (top_k=%d)...", top_k)
    context, layer_info = ra.retrieve(
        query, top_k=top_k, collapse_tree=True, return_layer_information=True,
    )

    # Convert layer_info to RetrievalHit
    hits: list[RetrievalHit] = []
    for rank, info in enumerate(layer_info, start=1):
        node = ra.tree.all_nodes[info["node_index"]]
        layer = info["layer_number"]
        hits.append(RetrievalHit(
            chunk_id=f"level_{layer}_official_{info['node_index']}",
            signal_name=f"raptor_official_L{layer}",
            rank=rank,
            score=1.0 - (rank / (top_k + 1)),
            text_preview=node.text[:200],
        ))

    logger.info("RAPTOR-Official: %d hits", len(hits))
    return hits
