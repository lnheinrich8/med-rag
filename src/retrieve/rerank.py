"""Cross-encoder reranking of dense candidates (BAAI/bge-reranker-base).

Dense retrieval scores a query and a chunk *independently* (a bi-encoder): fast,
but it can't see how the two interact, so near-duplicates and boilerplate crowd
the top. A cross-encoder re-reads each (query, chunk) pair *together* with full
attention and emits a single relevance score, discriminating much better among
candidates — at the cost of one model pass per candidate. So we use it to
*reorder* the dense top_k, never to search from scratch.

The model is loaded once per (model, device) via a module-level cache: the eval
loop constructs a fresh ``Reranker`` per question, and without caching that would
reload ~1 GB onto the GPU every question — the same trap the embedder hit.
"""

from __future__ import annotations

from functools import cached_property, lru_cache

from ..config import RerankConfig, settings
from ..embed.embedder import _resolve_device
from .dense import SearchHit


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str, device: str):
    """Load (and cache) one CrossEncoder per (model, device). See module docstring."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device=device)


class Reranker:
    """Cross-encoder that reorders dense candidates by joint relevance score."""

    def __init__(self, cfg: RerankConfig) -> None:
        self.cfg = cfg

    @cached_property
    def _model(self):
        return _load_cross_encoder(self.cfg.model, _resolve_device(settings.device))

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        """Return all `hits` reordered by cross-encoder score (descending).

        Reorders the *full* candidate pool, not just the top few, so downstream
        retrieval metrics (recall@10/@20) still measure the reranked depth. Each
        hit's ``score`` is overwritten with its cross-encoder score for
        transparency — note this is an unbounded logit, not a cosine similarity.
        """
        if not hits:
            return hits
        scores = self._model.predict(
            [(query, h.content) for h in hits],
            show_progress_bar=False,
        )
        order = sorted(zip(hits, scores), key=lambda pair: float(pair[1]), reverse=True)
        reranked: list[SearchHit] = []
        for hit, score in order:
            hit.score = float(score)
            reranked.append(hit)
        return reranked
