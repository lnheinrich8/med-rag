"""Chunking strategies (the first real ablation axis).

Three strategies, all config-driven via ``ChunkConfig``:

* ``recursive`` (default) — split on a separator hierarchy (paragraph -> line ->
  sentence -> word -> char), greedily merging pieces up to a target size with
  overlap. The general-purpose workhorse.
* ``section`` — split on markdown headings first, then recursively within each
  section; records the heading on every chunk (useful for guideline structure).
* ``fixed`` — a dumb sliding window. Deliberately weak; a baseline to beat.

``semantic`` is deferred to Step 6. Sizes in ``ChunkConfig`` are in *tokens*; we
approximate tokens with a chars-per-token constant so ingest needs no tokenizer.
"""

from __future__ import annotations

import bisect
import re

from ..config import ChunkConfig
from ..models import Chunk, Document

# Rough bytes->tokens ratio for English; good enough for sizing (not for limits).
_CHARS_PER_TOKEN = 4
# Tried in order: paragraph, line, sentence, word, then character ("" splits all).
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
_HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)


def estimate_tokens(text: str) -> int:
    """Approximate token count from character length."""
    return max(1, round(len(text) / _CHARS_PER_TOKEN))


def _page_boundaries(pages: list[str], sep_len: int = 2) -> list[int]:
    """Char offset where each page starts in Document.full_text (sep = "\\n\\n")."""
    bounds: list[int] = []
    pos = 0
    for p in pages:
        bounds.append(pos)
        pos += len(p) + sep_len
    return bounds


def _page_number(boundaries: list[int], offset: int) -> int:
    """1-based page number containing char `offset`."""
    return bisect.bisect_right(boundaries, offset)


def _merge_splits(splits: list[str], sep: str, max_chars: int, overlap: int) -> list[str]:
    """Greedily merge small splits into ~max_chars chunks, keeping `overlap` chars."""
    chunks: list[str] = []
    current: list[str] = []
    total = 0
    for s in splits:
        addition = len(s) + (len(sep) if current else 0)
        if total + addition > max_chars and current:
            chunks.append(sep.join(current))
            # Trim from the front until the carried-over tail is under `overlap`.
            while total > overlap and current:
                total -= len(current[0]) + (len(sep) if len(current) > 1 else 0)
                current.pop(0)
        current.append(s)
        total += len(s) + (len(sep) if len(current) > 1 else 0)
    if current:
        chunks.append(sep.join(current))
    
    return chunks


def _recursive_split(text: str, separators: list[str], max_chars: int, overlap: int) -> list[str]:
    """Recursively split on the first applicable separator, then merge."""
    sep, remaining = separators[-1], []
    for i, candidate in enumerate(separators):
        if candidate == "":
            sep = candidate
            break
        if candidate in text:
            sep, remaining = candidate, separators[i + 1:]
            break

    splits = list(text) if sep == "" else text.split(sep)

    final: list[str] = []
    good: list[str] = []
    for s in splits:
        if len(s) < max_chars:
            good.append(s)
            continue
        if good:
            final.extend(_merge_splits(good, sep, max_chars, overlap))
            good = []
        if remaining:
            final.extend(_recursive_split(s, remaining, max_chars, overlap))
        else:
            final.append(s)
    if good:
        final.extend(_merge_splits(good, sep, max_chars, overlap))
    return final


def _fixed_split(text: str, max_chars: int, overlap: int) -> list[str]:
    """A naive sliding window of `max_chars`, stepping by (max_chars - overlap)."""
    step = max(1, max_chars - overlap)
    return [text[i:i + max_chars] for i in range(0, len(text), step)]


def _split_sections(text: str) -> list[tuple[str | None, str]]:
    """Split on markdown headings -> list of (heading, body). Preamble heading is None."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [(None, text)]
    sections: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        sections.append((None, text[: matches[0].start()]))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((m.group(1).strip(), text[start:end]))
    return sections


def _assemble(
    text: str,
    pieces: list[str],
    sections: list[str | None],
    boundaries: list[int],
) -> list[Chunk]:
    """Turn raw text pieces into Chunks, resolving each piece's page span."""
    chunks: list[Chunk] = []
    cursor = 0
    for piece, section in zip(pieces, sections):
        body = piece.strip()
        if not body:
            continue
        start = text.find(piece, cursor)
        if start == -1:
            start = cursor
        end = start + len(piece)
        cursor = start + 1  # allow overlapping pieces to still be located in order
        chunks.append(
            Chunk(
                content=body,
                chunk_index=len(chunks),
                token_count=estimate_tokens(body),
                page_start=_page_number(boundaries, start),
                page_end=_page_number(boundaries, max(start, end - 1)),
                section=section,
            )
        )
    return chunks


def chunk_document(doc: Document, cfg: ChunkConfig) -> list[Chunk]:
    """Chunk a Document according to the configured strategy."""
    text = doc.full_text
    boundaries = _page_boundaries(doc.pages)
    max_chars = cfg.chunk_size * _CHARS_PER_TOKEN
    overlap = cfg.chunk_overlap * _CHARS_PER_TOKEN

    if cfg.strategy == "fixed":
        pieces = _fixed_split(text, max_chars, overlap)
        sections: list[str | None] = [None] * len(pieces)
    elif cfg.strategy == "section":
        pieces, sections = [], []
        for heading, body in _split_sections(text):
            sub = _recursive_split(body, _SEPARATORS, max_chars, overlap)
            pieces.extend(sub)
            sections.extend([heading] * len(sub))
    else:  # "recursive" (default); "semantic" is deferred to Step 6
        pieces = _recursive_split(text, _SEPARATORS, max_chars, overlap)
        sections = [None] * len(pieces)

    return _assemble(text, pieces, sections, boundaries)
