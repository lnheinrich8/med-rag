"""Dense retrieval: cosine ANN search over chunks.embedding.

The query is embedded with the same model used at ingest, then we ask Postgres
for the nearest chunks via the HNSW index (`<=>` is cosine *distance*; we report
similarity = 1 - distance so bigger is better). Sparse and hybrid retrieval
arrive in Step 5; this is the dense leg they'll build on.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import EmbedConfig, RetrievalConfig
from ..db import connect
from ..embed.embedder import Embedder


@dataclass
class SearchHit:
    chunk_id: int
    document_id: int
    score: float  # cosine similarity in [-1, 1]; 1.0 == identical direction
    content: str
    section: str | None
    page_start: int | None
    page_end: int | None
    source_path: str
    title: str | None


_SQL = """
    SELECT c.id, c.document_id,
           1 - (c.embedding <=> %s::vector) AS score,
           c.content, c.section, c.page_start, c.page_end,
           d.source_path, d.title
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE c.embedding IS NOT NULL
    ORDER BY c.embedding <=> %s::vector
    LIMIT %s
"""


def dense_search(
    query: str,
    embed_cfg: EmbedConfig,
    retrieval_cfg: RetrievalConfig,
) -> list[SearchHit]:
    """Embed `query` and return the top_k nearest chunks by cosine similarity."""
    qvec = Embedder(embed_cfg).embed_query(query)
    with connect() as conn:
        rows = conn.execute(_SQL, (qvec, qvec, retrieval_cfg.top_k)).fetchall()
    return [SearchHit(*row) for row in rows]
