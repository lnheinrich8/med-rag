"""Hybrid retrieval: fuse the dense and sparse rankings with RRF.

Reciprocal Rank Fusion combines rankings using only each item's *rank position*,
not its raw score — which is the whole point: a cosine similarity (0–1) and a
``ts_rank_cd`` (unbounded) aren't comparable, but "this chunk is rank 2 in dense
and rank 5 in sparse" is. Each list contributes ``1 / (rrf_k + rank)`` to a
chunk's fused score; a chunk that ranks decently in *both* legs beats one that
ranks highly in only one. ``rrf_k`` (default 60) damps the influence of the very
top ranks so a single leg can't dominate.

This is where the dense-only hard misses get a second chance: if the sparse leg
surfaces a term-matched chunk dense never did, fusion can lift it into the top_k.
"""

from __future__ import annotations

from ..config import EmbedConfig, RetrievalConfig
from .dense import SearchHit, dense_search
from .sparse import sparse_search


def rrf_fuse(
    rankings: list[list[SearchHit]], rrf_k: int, top_k: int
) -> list[SearchHit]:
    """Fuse several ranked hit-lists into one by Reciprocal Rank Fusion.

    Pure function (no DB) so it's unit-testable. Dedupes by ``chunk_id``, writes
    the fused score back onto each surviving hit, and returns the top_k.
    """
    scores: dict[int, float] = {}
    hit_by_id: dict[int, SearchHit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, 1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            hit_by_id.setdefault(hit.chunk_id, hit)

    fused = sorted(hit_by_id.values(), key=lambda h: scores[h.chunk_id], reverse=True)
    for hit in fused:
        hit.score = scores[hit.chunk_id]
    return fused[:top_k]


def hybrid_search(
    query: str, embed_cfg: EmbedConfig, retrieval_cfg: RetrievalConfig
) -> list[SearchHit]:
    """Run dense + sparse retrieval and fuse them with RRF into one top_k ranking."""
    dense_hits = dense_search(query, embed_cfg, retrieval_cfg)
    sparse_hits = sparse_search(query, retrieval_cfg)
    return rrf_fuse([dense_hits, sparse_hits], retrieval_cfg.rrf_k, retrieval_cfg.top_k)
