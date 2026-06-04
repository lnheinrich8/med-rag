"""Core data structures passed between pipeline stages.

`Document` is one loaded PDF (before chunking); `Chunk` is the unit of retrieval
that gets embedded and stored. These are plain dataclasses — the mapping to/from
DB rows lives in ``store/``, not here, so the pipeline stays decoupled from SQL.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    """One source PDF, extracted to per-page text (index 0 == page 1)."""

    source_path: Path
    source_type: str  # guideline | statpearl | review
    pages: list[str]
    title: str | None = None
    section_number: str | None = None

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        """All pages concatenated, one blank line between page boundaries.

        Chunkers operate on this string; ``\\n\\n`` (2 chars) is the page
        separator, which the chunker relies on to map char offsets back to pages.
        """
        return "\n\n".join(self.pages)

    @property
    def content_hash(self) -> str:
        """Stable sha256 of the text — drives idempotent re-ingest in store/."""
        return hashlib.sha256(self.full_text.encode("utf-8")).hexdigest()


@dataclass
class Chunk:
    """A retrieval unit: a slice of one document, optionally embedded."""

    content: str
    chunk_index: int  # order within the parent document
    token_count: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    embedding: list[float] | None = None  # None until the embedder fills it

    def preview(self, width: int = 100) -> str:
        """A single-line, whitespace-collapsed snippet for CLI display."""
        flat = " ".join(self.content.split())
        return flat if len(flat) <= width else flat[:width] + "..."
