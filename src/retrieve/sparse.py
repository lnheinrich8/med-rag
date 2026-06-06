"""Sparse retrieval: Postgres full-text search over chunks.tsv.

The dense leg matches on *meaning*; this leg matches on *terms*. For exact
medical tokens — "metformin", "sulfonylurea", an eGFR threshold — lexical overlap
is exactly what dense embeddings blur, which is why several of our hard misses
never surfaced. ``tsv`` is a generated ``to_tsvector('english', content)`` column
with a GIN index, so this is a cheap query on data already in the DB.

We parse the question with ``websearch_to_tsquery`` (tolerant of arbitrary user
text — no escaping needed) and rank with ``ts_rank_cd`` (cover density, which
rewards matched terms appearing close together). Returns the same ``SearchHit``
shape as dense so the fusion layer can treat both legs uniformly.
"""

from __future__ import annotations

from ..config import RetrievalConfig
from ..db import connect
from .dense import SearchHit

_SQL = """
    SELECT c.id, c.document_id,
           ts_rank_cd(c.tsv, query) AS score,
           c.content, c.section, c.page_start, c.page_end,
           d.source_path, d.title
    FROM chunks c
    JOIN documents d ON d.id = c.document_id,
         websearch_to_tsquery('english', %s) AS query
    WHERE c.tsv @@ query
    ORDER BY score DESC
    LIMIT %s
"""


def sparse_search(query: str, retrieval_cfg: RetrievalConfig) -> list[SearchHit]:
    """Return the top_k chunks by full-text relevance (ts_rank_cd)."""
    with connect() as conn:
        rows = conn.execute(_SQL, (query, retrieval_cfg.top_k)).fetchall()
    return [SearchHit(*row) for row in rows]
