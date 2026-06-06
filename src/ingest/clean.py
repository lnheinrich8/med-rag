"""Strip recurring PDF boilerplate from page text before chunking.

The corpus PDFs carry per-page furniture that survives extraction: StatPearls/NCBI
running headers ("6/2/26, 2:06 PM Metformin - StatPearls - NCBI Bookshelf"), the
NCBI Bookshelf URL + "page x/y" footer, and ADA Standards journal footers
("Diabetes Care Volume 49, Supplement 1, January 2026"). Repeated on every page,
this furniture dilutes chunk embeddings and term frequencies — a *measured* cause
of retrieval misses (the metformin mechanism-of-action chunk that's half NCBI
header never cracked the top-20). We strip only high-confidence patterns: the aim
is less noise, not perfect cleaning, and never touching clinical text.
"""

from __future__ import annotations

import re

_PATTERNS = [
    # StatPearls/NCBI running header: timestamp + page title + "… NCBI Bookshelf".
    re.compile(r"\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}\s*[AP]M[^\n]*?NCBI Bookshelf", re.I),
    # NCBI Bookshelf URL footer, optionally trailed by a "page x/y" marker.
    re.compile(r"https?://www\.ncbi\.nlm\.nih\.gov/books/NBK\d+/?(?:\s+\d+/\d+)?", re.I),
    # ADA Standards of Care journal running footer.
    re.compile(r"Diabetes Care\s+Volume\s+\d+,\s*Supplement\s+\d+,\s*[A-Za-z]+\s+\d{4}", re.I),
    re.compile(r"\bdiabetesjournals\.org/care\b", re.I),
]

# Tidy the whitespace left where boilerplate was removed (keeps chunk sizing sane).
_MULTISPACE = re.compile(r"[ \t]{2,}")
_BLANKLINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Remove recurring header/footer boilerplate from one page of extracted text."""
    for pattern in _PATTERNS:
        text = pattern.sub(" ", text)
    text = _MULTISPACE.sub(" ", text)
    text = _BLANKLINES.sub("\n\n", text)
    return text
