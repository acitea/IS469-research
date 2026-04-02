"""Fixed-size character chunking within section boundaries."""

from __future__ import annotations

import logging

from contextrag.config import CHUNK_OVERLAP, CHUNK_SIZE
from contextrag.models import Chunk, DocumentMeta, DocumentSection

logger = logging.getLogger(__name__)


def chunk_sections(
    doc_meta: DocumentMeta,
    sections: list[DocumentSection],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk a document's sections into fixed-size character windows.

    Each chunk records its position in the overall document (via
    ``chunk_index``) and its character offsets relative to the original
    full document text.
    """
    chunks: list[Chunk] = []
    global_index = 0

    for section in sections:
        text = section.text
        if not text.strip():
            continue

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                abs_start = section.char_offset_start + start
                abs_end = section.char_offset_start + end
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_meta.file_name}::chunk_{global_index}",
                        doc_file_name=doc_meta.file_name,
                        section_title=section.section_title,
                        chunk_index=global_index,
                        text=chunk_text,
                        char_offset_start=abs_start,
                        char_offset_end=abs_end,
                    )
                )
                global_index += 1

            # Advance by (chunk_size - overlap), but at least 1 to avoid infinite loop
            step = max(chunk_size - chunk_overlap, 1)
            start += step

    logger.info(
        "Chunked %s into %d chunks (%d sections)",
        doc_meta.file_name,
        len(chunks),
        len(sections),
    )
    return chunks


def chunk_text_simple(
    text: str,
    source_label: str = "unknown",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk a raw text string without section parsing.

    Used for training data (Wikipedia articles) where we don't need
    section-level parsing but still want consistent chunk objects.
    """
    chunks: list[Chunk] = []
    idx = 0
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=f"{source_label}::chunk_{idx}",
                    doc_file_name=source_label,
                    section_title="",
                    chunk_index=idx,
                    text=chunk_text,
                    char_offset_start=start,
                    char_offset_end=end,
                )
            )
            idx += 1
        step = max(chunk_size - chunk_overlap, 1)
        start += step
    return chunks
