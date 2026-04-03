"""CLI entry point: ``uv run python -m contextrag <command>``."""

from __future__ import annotations

import argparse
import sys

from contextrag.config import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_TOP_K,
    TEXTS_DIR,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_LR,
    TRAIN_NUM_ARTICLES,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contextrag",
        description="Context-conditioned retrieval POC with multi-signal fusion.",
    )
    sub = parser.add_subparsers(dest="command")

    # -- train ---------------------------------------------------------------
    p_train = sub.add_parser("train", help="Train the context-conditioned encoder on Wikipedia.")
    p_train.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW,
                         help="Preceding context in tokens (0=none, N=fixed, -1=full doc). Default: %(default)s")
    p_train.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    p_train.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    p_train.add_argument("--lr", type=float, default=TRAIN_LR)
    p_train.add_argument("--num-articles", type=int, default=TRAIN_NUM_ARTICLES,
                         help="Wikipedia articles to sample. Default: %(default)s")
    p_train.add_argument("--device", type=str, default=None,
                         help="Device name (cuda, mps, cpu). Auto-detected if omitted.")

    # -- index ---------------------------------------------------------------
    p_index = sub.add_parser("index", help="Build all retrieval indexes.")
    p_index.add_argument("--texts-dir", type=str, default=str(TEXTS_DIR),
                         help="Directory containing .md corpus files. Default: %(default)s")
    p_index.add_argument("--force-rebuild", action="store_true",
                         help="Rebuild indexes even if they already exist.")
    p_index.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW,
                         help="Preceding context in tokens for conditioned embeddings. Default: %(default)s")
    p_index.add_argument("--device", type=str, default=None,
                         help="Device name (cuda, mps, cpu). Auto-detected if omitted.")
    p_index.add_argument("--skip-llm-context", action="store_true",
                         help="Skip Anthropic-style LLM context generation.")
    p_index.add_argument("--skip-raptor", action="store_true",
                         help="Skip RAPTOR hierarchy building.")
    p_index.add_argument("--raptor-backend", type=str, choices=["custom", "official"], default="custom",
                         help="RAPTOR impl: 'custom' (K-Means) or 'official' (UMAP+GMM). Default: %(default)s")

    # -- query ---------------------------------------------------------------
    p_query = sub.add_parser("query", help="Run a query against all indexes.")
    p_query.add_argument("query", type=str, help="The query string.")
    p_query.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p_query.add_argument("--signals", type=str, default=None,
                         help="Comma-separated signals: conditioned,contextual,coil,raptor. Default: all.")
    p_query.add_argument("--verbose", action="store_true")
    p_query.add_argument("--raptor-backend", type=str, choices=["custom", "official"], default="custom",
                         help="RAPTOR backend to query. Must match indexing backend. Default: %(default)s")

    # -- demo ----------------------------------------------------------------
    p_demo = sub.add_parser("demo", help="Run preset demo queries.")
    p_demo.add_argument("--query-type", type=str, default=None,
                        choices=["anchor", "paraphrase", "disambiguation", "cross-section"],
                        help="Run a specific demo query type. Default: all.")

    # -- status --------------------------------------------------------------
    sub.add_parser("status", help="Show index and model status.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    from contextrag.cli.commands import cmd_demo, cmd_index, cmd_query, cmd_status, cmd_train

    commands = {
        "train": cmd_train,
        "index": cmd_index,
        "query": cmd_query,
        "demo": cmd_demo,
        "status": cmd_status,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
