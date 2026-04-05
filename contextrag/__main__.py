"""CLI entry point: ``uv run python -m contextrag <command>``."""

from __future__ import annotations

import argparse
import sys

from contextrag.config import (
    DEFAULT_TOP_K,
    TEXTS_DIR,
    EMBEDDER_MODE,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contextrag",
        description="Context-conditioned retrieval POC with multi-signal fusion.",
    )
    sub = parser.add_subparsers(dest="command")

    # -- index ---------------------------------------------------------------
    p_index = sub.add_parser("index", help="Build retrieval indexes.")
    p_index.add_argument("--texts-dir", type=str, default=str(TEXTS_DIR),
                         help="Directory containing .md corpus files. Default: %(default)s")
    p_index.add_argument("--force-rebuild", action="store_true",
                         help="Rebuild indexes even if they already exist.")
    p_index.add_argument("--embedder-mode", type=str, default=EMBEDDER_MODE,
                         choices=["default", "contextual", "voyage"],
                         help="Embedder mode: default (OpenAI standalone), contextual (OpenAI + pooling), or voyage (Voyage API). Default: %(default)s")
    p_index.add_argument("--indexes", type=str, default=None,
                         help="Comma-separated indexes to build: dense,contextual,coil,raptor. Default: all.")

    # -- query ---------------------------------------------------------------
    p_query = sub.add_parser("query", help="Run a query against all indexes.")
    p_query.add_argument("query", type=str, help="The query string.")
    p_query.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p_query.add_argument("--signals", type=str, default=None,
                         help="Comma-separated signals: dense,contextual,coil,raptor. Default: all.")
    p_query.add_argument("--verbose", action="store_true")

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

    from contextrag.cli.commands import cmd_demo, cmd_index, cmd_query, cmd_status

    commands = {
        "index": cmd_index,
        "query": cmd_query,
        "demo": cmd_demo,
        "status": cmd_status,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
