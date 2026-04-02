"""Central configuration for the contextrag package."""

from __future__ import annotations

import logging
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXTS_DIR = PROJECT_ROOT / "texts"
DATABASE_DIR = PROJECT_ROOT / "database" / "contextrag"
MODEL_DIR = DATABASE_DIR / "model"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embeddings for contextual retrieval
BACKBONE_MODEL = "BAAI/bge-base-en-v1.5"   # Frozen encoder backbone for conditioned embeddings
BACKBONE_DIM = 768                          # Hidden dimension of backbone
BACKBONE_MAX_TOKENS = 512                   # Max sequence length per encoding pass

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE = 1000          # characters per chunk
CHUNK_OVERLAP = 200        # character overlap between chunks

# ---------------------------------------------------------------------------
# Context-Conditioned Encoder
# ---------------------------------------------------------------------------

DEFAULT_CONTEXT_WINDOW = 512   # tokens of preceding text (0=none, -1=full doc)
CROSS_ATTENTION_HEADS = 8
TRAIN_EPOCHS = 20
TRAIN_BATCH_SIZE = 32
TRAIN_LR = 1e-4
TRAIN_VAL_SPLIT = 0.1
TRAIN_NUM_ARTICLES = 5000      # Wikipedia articles to sample for training
TRAIN_MIN_ARTICLE_CHARS = 2000 # Minimum article length to include

# ---------------------------------------------------------------------------
# ChromaDB Collections
# ---------------------------------------------------------------------------

CONDITIONED_COLLECTION = "contextrag_conditioned"
CONTEXTUAL_COLLECTION = "contextrag_contextual"
RAPTOR_COLLECTION = "contextrag_raptor"
CHROMA_BATCH_SIZE = 5000

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = 10
RRF_K = 60                # Reciprocal Rank Fusion constant
DENSE_CANDIDATES = 20     # Top-k per dense signal
LEXICAL_CANDIDATES = 20   # Top-k for COIL/BM25
RAPTOR_CANDIDATES_PER_LEVEL = 10

# ---------------------------------------------------------------------------
# COIL / BM25
# ---------------------------------------------------------------------------

ANCHOR_BOOST = 3           # Token duplication factor for anchor terms in BM25
COIL_ALPHA = 0.7           # Weight of BM25 vs anchor-overlap in COIL score

# ---------------------------------------------------------------------------
# RAPTOR
# ---------------------------------------------------------------------------

RAPTOR_MAX_LEVELS = 3
RAPTOR_MIN_NODES_TO_CLUSTER = 6   # Stop clustering when level has <= this many nodes
RAPTOR_CLUSTER_DIVISOR = 10       # k = max(2, ceil(n / divisor))
RAPTOR_SUMMARY_MAX_TOKENS = 200

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)


def load_env() -> None:
    """Load .env file from project root into os.environ."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
