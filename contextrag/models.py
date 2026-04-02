"""Data models for the contextrag package."""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Source Identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocumentMeta:
    """Metadata about a source document."""

    file_name: str
    file_path: str
    char_count: int


# ---------------------------------------------------------------------------
# Structural Sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocumentSection:
    """A structural section within a document (page or heading-delimited)."""

    doc_file_name: str
    section_index: int
    section_title: str
    text: str
    char_offset_start: int
    char_offset_end: int


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """Atomic unit of retrieval — a fixed-size text span within a section."""

    chunk_id: str              # globally unique: "{file_name}::chunk_{index}"
    doc_file_name: str
    section_title: str
    chunk_index: int           # position within document (0-indexed)
    text: str
    char_offset_start: int     # offset in original document
    char_offset_end: int


@dataclass
class ContextualChunk:
    """A chunk enriched with context for embedding.

    Holds all context variants so different retrieval signals can use
    whichever combination they need.
    """

    chunk: Chunk

    # Context-conditioned (component 1): raw preceding document text
    preceding_context: str = ""

    # Anthropic-style (component 2): LLM-generated context description
    llm_context: str = ""

    # Final embedding texts built from the above
    conditioned_embedding_text: str = ""   # preceding_context + chunk.text
    contextual_embedding_text: str = ""    # llm_context + chunk.text


# ---------------------------------------------------------------------------
# RAPTOR Hierarchy
# ---------------------------------------------------------------------------


@dataclass
class HierarchyNode:
    """A node in the RAPTOR summary tree."""

    node_id: str                           # "level_{L}_cluster_{C}"
    level: int                             # 0 = leaf (original chunk), 1+ = summary
    summary_text: str                      # original text for level 0, summary for 1+
    child_ids: list[str] = field(default_factory=list)
    source_chunk_ids: list[str] = field(default_factory=list)  # all leaf chunk_ids covered
    doc_file_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Retrieval Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalHit:
    """A single retrieval result from one signal."""

    chunk_id: str
    signal_name: str           # "conditioned", "contextual", "coil", "raptor_L0", etc.
    rank: int                  # 1-indexed rank within this signal
    score: float
    text_preview: str = ""     # truncated text for display


@dataclass
class FusedResult:
    """Final ranked result combining all signals."""

    rank: int
    chunk_id: str
    text: str
    doc_file_name: str
    section_title: str
    preceding_context: str
    llm_context: str
    fused_score: float
    signal_contributions: list[RetrievalHit] = field(default_factory=list)
    hierarchy_level: int = 0
    source_chunk_ids: list[str] = field(default_factory=list)
