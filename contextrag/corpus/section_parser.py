"""Parse document structure into sections."""

from __future__ import annotations

import re
from contextrag.models import DocumentMeta, DocumentSection

# Matches markdown H2 headings like "## page_001.png" or "## Item 1A. Risk Factors"
_SECTION_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)


def parse_sections(doc_meta: DocumentMeta, full_text: str) -> list[DocumentSection]:
    """Split a document into structural sections based on ``##`` headings.

    If no ``##`` headings are found the entire document becomes one section.
    """
    matches = list(_SECTION_RE.finditer(full_text))

    if not matches:
        return [
            DocumentSection(
                doc_file_name=doc_meta.file_name,
                section_index=0,
                section_title="full_document",
                text=full_text,
                char_offset_start=0,
                char_offset_end=len(full_text),
            )
        ]

    sections: list[DocumentSection] = []

    # Text before the first heading (preamble)
    preamble = full_text[: matches[0].start()].strip()
    if preamble:
        sections.append(
            DocumentSection(
                doc_file_name=doc_meta.file_name,
                section_index=0,
                section_title="preamble",
                text=preamble,
                char_offset_start=0,
                char_offset_end=matches[0].start(),
            )
        )

    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        text = full_text[start:end].strip()
        if not text:
            continue
        sections.append(
            DocumentSection(
                doc_file_name=doc_meta.file_name,
                section_index=len(sections),
                section_title=title,
                text=text,
                char_offset_start=start,
                char_offset_end=end,
            )
        )

    return sections
