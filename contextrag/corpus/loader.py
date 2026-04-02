"""Load markdown documents from the corpus directory."""

from __future__ import annotations

import logging
from pathlib import Path

from contextrag.models import DocumentMeta

logger = logging.getLogger(__name__)


def load_corpus(texts_dir: Path) -> list[tuple[DocumentMeta, str]]:
    """Read all .md files from *texts_dir* and return (meta, full_text) pairs.

    Raises ``FileNotFoundError`` if the directory does not exist and
    ``ValueError`` if it contains no markdown files with content.
    """
    if not texts_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {texts_dir}")

    md_files = sorted(texts_dir.glob("*.md"))
    if not md_files:
        raise ValueError(f"No .md files found in {texts_dir}")

    results: list[tuple[DocumentMeta, str]] = []
    for path in md_files:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            logger.warning("Skipping empty file: %s", path.name)
            continue
        meta = DocumentMeta(
            file_name=path.name,
            file_path=str(path),
            char_count=len(text),
        )
        results.append((meta, text))

    if not results:
        raise ValueError(f"All .md files in {texts_dir} are empty")

    logger.info("Loaded %d documents from %s", len(results), texts_dir)
    return results
