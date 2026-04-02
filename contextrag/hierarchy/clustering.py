"""K-means clustering wrapper for RAPTOR."""

from __future__ import annotations

import logging
import math

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int,
    seed: int = 42,
) -> list[list[int]]:
    """Cluster embedding vectors into groups using K-means.

    Returns a list of index groups, one per cluster.
    """
    n_clusters = min(n_clusters, len(embeddings))
    if n_clusters <= 1:
        return [list(range(len(embeddings)))]

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = km.fit_predict(embeddings)

    clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(idx)

    result = [clusters[k] for k in sorted(clusters.keys())]
    logger.info("Clustered %d items into %d groups", len(embeddings), len(result))
    return result


def compute_n_clusters(n_items: int, divisor: int = 10) -> int:
    """Compute the number of clusters for a given item count."""
    return max(2, math.ceil(n_items / divisor))
