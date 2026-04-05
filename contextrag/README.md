# contextrag

Context-conditioned retrieval POC integrating four distinct retrieval signals into a unified ranking pipeline via Reciprocal Rank Fusion.

## Core Idea

Standard chunk embeddings ignore where a chunk sits in its document. Two identical paragraphs appearing in different sections (e.g. "Risk Factors" vs "Financial Highlights") receive the same embedding, even though their meaning differs based on surrounding context.

**contextrag** solves this with a custom neural network that produces **context-conditioned chunk embeddings** — a trainable cross-attention layer on a frozen BGE-base-en-v1.5 backbone, where each chunk's representation is modulated by its preceding document context. The model is trained via self-supervised contrastive learning on Wikipedia (domain-agnostic), so the `texts/` corpus remains purely for evaluation.

This conditioned signal is fused with three complementary retrieval signals to cover different aspects of relevance.

## Four Retrieval Signals

| Signal | Method | What it captures |
|--------|--------|-----------------|
| **Conditioned** | Custom cross-attention NN on frozen BGE-base-en-v1.5 | Semantic meaning influenced by document context |
| **Contextual** | GPT-4o-mini generates per-chunk context descriptions, embedded with `text-embedding-3-small` | Explicit articulation of chunk's role in the document |
| **COIL** | Anchor-term-boosted weighted BM25 + Jaccard overlap | Exact entity/date matching (dollar amounts, tickers, years) |
| **RAPTOR** | Multi-level clustering + LLM summarization tree (custom K-Means) | Cross-section and cross-document evidence |

All signals are combined via **Reciprocal Rank Fusion** (RRF, k=60) into a single ranked result with full per-signal provenance.

## Architecture

### Context-Conditioned Encoder

```
  Preceding Context (variable length)     Chunk Text
       |                                      |
  [Frozen BGE-base-en-v1.5]         [Frozen BGE-base-en-v1.5]
   (768-dim, shared weights)          (shared weights)
       |                                      |
   context_hidden (N x 768)           chunk_hidden (M x 768)
       |                                      |
       +---------> Cross-Attention <----------+
                   (chunk attends to context)
                   (trainable, 8 heads)
                          |
                   conditioned_repr (M x 768)
                          |
                     Mean Pooling
                          |
                   Projection Head (trainable)
                   768 -> 768 linear + LayerNorm
                          |
                   chunk_embedding (768-dim)
```

- **Frozen backbone:** BAAI/bge-base-en-v1.5 (137M params) — retrieval-optimized, not fine-tuned
- **Trainable params:** ~2.96M (cross-attention + LayerNorm + projection)
- **Long context handling:** Preceding text split into 512-token windows with 50% overlap, encoded separately, hidden states concatenated before cross-attention
- **Context window:** Configurable via `--context-window` (0 = no context baseline, N = last N tokens, -1 = full preceding document)

### Training

- **Data:** Wikipedia (wikimedia/wikipedia 20231101.en), ~5,000 articles sampled via reservoir sampling
- **Objective:** InfoNCE (NT-Xent) contrastive loss with in-batch negatives
- **Positive pairs:** Neighboring chunks (within 3 positions) from the same document
- **Key negative signal:** Same chunk text under different document contexts forces the model to learn that context changes meaning
- **Hardware:** Trains on 24GB GPU; infers on 12GB; CPU is supported but slower

## Quickstart

### Prerequisites

- Python 3.12+
- [UV](https://docs.astral.sh/uv/) package manager
- `OPENAI_API_KEY` in `.env` at project root (required for contextual retrieval, RAPTOR, and LLM context generation)

### Install

### 1. Train the encoder

Train the context-conditioned encoder on Wikipedia articles:

```bash
uv run python -m contextrag train
```

Options:
```
--context-window 512    # preceding context in tokens (0=baseline, N=fixed, -1=full doc)
--epochs 20             # training epochs
--batch-size 32         # batch size
--lr 1e-4               # learning rate
--num-articles 5000     # Wikipedia articles to sample
--device cuda           # device (cuda, mps, cpu; auto-detected if omitted)
```

Training saves to `database/contextrag/model/`:
- `best_model.pt` — best checkpoint by validation loss
- `train_config.json` — hyperparameters and best validation loss
- `training_log.json` — per-epoch loss curve

### 2. Build indexes

Index the evaluation corpus (`texts/`) using all four signals:

```bash
uv run python -m contextrag index
```

Build specific indexes only:

```bash
uv run python -m contextrag index --indexes conditioned,coil
uv run python -m contextrag index --indexes raptor
uv run python -m contextrag index --indexes contextual --force-rebuild
```

Options:
```
--texts-dir DIR                  # corpus directory (default: texts/)
--force-rebuild                  # rebuild indexes even if they exist
--context-window 512             # must match the value used during training
--device cuda                    # device for conditioned embedding inference
--indexes conditioned,contextual,coil,raptor  # select which indexes to build (default: all)
```

The indexing pipeline runs up to 7 steps (depending on `--indexes`):
1. Load and chunk the corpus (fixed 1000-char chunks, 200-char overlap, section-aware) — always runs
2. Generate context-conditioned embeddings (custom model) — `conditioned`
3. Store conditioned embeddings in ChromaDB — `conditioned`
4. Generate LLM context descriptions via GPT-4o-mini (cached) — `contextual`
5. Store contextual embeddings in ChromaDB (OpenAI text-embedding-3-small) — `contextual`
6. Build COIL/BM25 lexical index with anchor-term boosting — `coil`
7. Build RAPTOR hierarchy (cluster, summarize, repeat up to 3 levels) — `raptor`

All data is stored in `database/contextrag/`.

### 3. Query

```bash
uv run python -m contextrag query "What was Foot Locker's net income in 2022?" --verbose
```

Options:
```
--top-k 10                                          # number of results
--signals conditioned,contextual,coil,raptor         # select specific signals
--verbose                                            # show full text + LLM context
```

### 4. Run demo queries

Four preset queries that showcase different retrieval strengths:

```bash
uv run python -m contextrag demo
uv run python -m contextrag demo --query-type anchor          # exact entity/date matching
uv run python -m contextrag demo --query-type paraphrase      # semantic understanding
uv run python -m contextrag demo --query-type disambiguation  # context sensitivity
uv run python -m contextrag demo --query-type cross-section   # cross-document evidence
```

### 5. Check status

```bash
uv run python -m contextrag status
```

Shows which indexes and models are built.

## Package Structure

```
contextrag/
├── __init__.py
├── __main__.py              # CLI entry point (argparse subcommands)
├── config.py                # Paths, model names, hyperparameters
├── models.py                # All dataclasses (Chunk, ContextualChunk, FusedResult, etc.)
├── corpus/                  # Corpus loading and chunking
│   ├── loader.py            #   Read texts/*.md -> (DocumentMeta, full_text)
│   ├── section_parser.py    #   Split on ## headings -> DocumentSection list
│   └── chunker.py           #   Fixed-size char chunking within sections -> Chunk list
├── nn/                      # Custom neural network
│   ├── encoder.py           #   ContextConditionedEncoder (cross-attention on frozen BGE)
│   ├── dataset.py           #   Wikipedia loading + contrastive pair generation
│   ├── trainer.py           #   Training loop (InfoNCE loss, checkpointing)
│   └── embed.py             #   Batch inference -> conditioned embeddings
├── context/
│   └── contextual.py        # Anthropic-style LLM context descriptions (GPT-4o-mini, cached)
├── lexical/
│   └── coil.py              # Anchor term extraction + weighted BM25 + Jaccard scoring
├── hierarchy/
│   ├── clustering.py        # K-means wrapper (scikit-learn)
│   └── raptor.py            # K-Means RAPTOR tree: build, index, and query across levels
├── index/
│   ├── dense_store.py       # ChromaDB create/query for conditioned + contextual indexes
│   ├── bm25_store.py        # BM25 build/serialize/query
│   └── persistence.py       # Chunk/RAPTOR/manifest JSON serialization, index existence checks
├── retrieval/
│   ├── dense.py             # Query conditioned + contextual ChromaDB -> RetrievalHit
│   ├── lexical.py           # Query COIL/BM25 -> RetrievalHit
│   └── hierarchical.py      # Query RAPTOR across levels -> RetrievalHit
├── fusion/
│   ├── ranker.py            # Reciprocal Rank Fusion: score(d) = sum(1/(k + rank))
│   └── result.py            # Hydrate FusedResult with full provenance
└── cli/
    ├── commands.py           # Subcommand implementations (train, index, query, demo, status)
    └── display.py            # Formatted terminal output
```

## Data Storage

```
database/contextrag/
├── model/
│   ├── best_model.pt            # Trained cross-attention + projection weights
│   ├── train_config.json        # Training hyperparameters and best validation loss
│   └── training_log.json        # Per-epoch loss curve
├── conditioned/                 # ChromaDB: context-conditioned dense index (custom model)
├── contextual/                  # ChromaDB: Anthropic-style contextual index (OpenAI embeddings)
├── raptor/                      # ChromaDB: RAPTOR hierarchy embeddings
├── bm25_coil.pkl                # Serialized weighted BM25 + anchor data
├── raptor_tree.json             # RAPTOR hierarchy node tree
├── llm_context_cache.json       # Cached LLM context descriptions (keyed by chunk_id)
└── chunks.json                  # Serialized chunk data for query-time hydration
```

## Environment Variables

| Variable | Required | Used by |
|----------|----------|---------|
| `OPENAI_API_KEY` | Yes (for contextual + RAPTOR signals) | LLM context generation, RAPTOR summarization, contextual embeddings |

Place in `.env` at the project root. Not needed if you only build the conditioned + COIL indexes (`--indexes conditioned,coil`).

## Key Configuration (config.py)

| Constant | Default | Description |
|----------|---------|-------------|
| `BACKBONE_MODEL` | `BAAI/bge-base-en-v1.5` | Frozen encoder backbone (768-dim) |
| `DEFAULT_CONTEXT_WINDOW` | `512` | Tokens of preceding context |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Character overlap between chunks |
| `RRF_K` | `60` | RRF fusion constant |
| `ANCHOR_BOOST` | `3` | BM25 token duplication for anchor terms |
| `COIL_ALPHA` | `0.7` | BM25 vs anchor-overlap weight in COIL scoring |
| `RAPTOR_MAX_LEVELS` | `3` | Maximum RAPTOR tree depth |
| `TRAIN_NUM_ARTICLES` | `5000` | Wikipedia articles for training |

## Comparing Context-Conditioned vs Baseline

To demonstrate the value of context conditioning, train and index twice:

```bash
# With context (default)
uv run python -m contextrag train --context-window 512
uv run python -m contextrag index --indexes conditioned

# Without context (baseline)
uv run python -m contextrag train --context-window 0
uv run python -m contextrag index --indexes conditioned --force-rebuild --context-window 0
```

Then run the disambiguation demo query — the context-conditioned model should rank chunks differently based on their position in the document, while the baseline treats identical text the same regardless of where it appears.
