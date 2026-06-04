"""Load corpus PDFs into `Document` objects.

Uses pymupdf4llm for markdown-aware extraction (keeps headings and tables
readable, which the section-aware chunker depends on). We request per-page
chunks so we can keep one string per page and track page_start/page_end for
citations. ``source_type`` is inferred from which data/raw/<subdir> the file is
in; ``section_number`` is parsed from ADA Standards filenames (e.g. dc26s009).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from ..models import Document

# data/raw subdirectory name -> source_type label stored in the documents table.
_DIR_TO_TYPE = {
    "guidelines": "guideline",
    "statpearls": "statpearl",
    "reviews": "review",
}


def _infer_source_type(path: Path) -> str:
    """Map the corpus subfolder this file lives under to a source_type label."""
    for part in path.parts:
        if part in _DIR_TO_TYPE:
            return _DIR_TO_TYPE[part]
    return "unknown"


def _section_number_from_name(path: Path) -> str | None:
    """ADA files look like 'dc26s009.pdf' -> section '9'. Others -> None."""
    m = re.search(r"s0*(\d+)", path.stem)
    return m.group(1) if m else None


def load_pdf(path: Path) -> Document:
    """Extract a single PDF into a Document (one markdown string per page)."""
    import pymupdf4llm

    # pymupdf4llm >=1.27 defaults to a layout engine that invokes Tesseract OCR;
    # we want the classic text-based markdown extractor (no OCR dependency).
    pymupdf4llm.use_layout(False)

    # page_chunks=True -> list of dicts, each with "text" and "metadata".
    page_dicts = pymupdf4llm.to_markdown(str(path), page_chunks=True)

    pages = [pd["text"] for pd in page_dicts]

    meta = page_dicts[0].get("metadata", {}) if page_dicts else {}
    title = (meta.get("title") or "").strip() or None

    return Document(
        source_path=path.resolve(),
        source_type=_infer_source_type(path),
        pages=pages,
        title=title,
        section_number=_section_number_from_name(path),
    )


def load_corpus(root: Path) -> Iterator[Document]:
    """Yield a Document for every *.pdf under `root`, recursively and sorted."""
    for pdf in sorted(root.rglob("*.pdf")):
        yield load_pdf(pdf)
