"""Retrieval entrypoint: dense candidates + optional reranking.

``retrieve()`` is the single seam the rest of the app calls — answer generation
(`rag query`) and the eval runner both go through it, so swapping the retrieval
strategy is a one-file change. Today it's dense top_k → optional cross-encoder
rerank; Step 5.2's hybrid (sparse + RRF fusion) slots in right here without
touching any caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .dense import dense_search

if TYPE_CHECKING:
    from ..config import ExperimentConfig
    from .dense import SearchHit


def retrieve(query: str, cfg: "ExperimentConfig") -> list["SearchHit"]:
    """Return ranked candidates for `query`: dense top_k, reordered if rerank is on."""
    hits = dense_search(query, cfg.embed, cfg.retrieval)
    if cfg.rerank.enabled:
        from .rerank import Reranker

        hits = Reranker(cfg.rerank).rerank(query, hits)
    return hits


def select_contexts(hits: list["SearchHit"], cfg: "ExperimentConfig") -> list["SearchHit"]:
    """The slice of ranked hits that actually goes into the LLM prompt.

    With reranking on, ``rerank.top_n`` governs how many survive into the prompt
    (its stated purpose: "kept after reranking, fed to the LLM"); otherwise
    ``generation.context_chunks`` does. The full ranked list is still returned by
    ``retrieve()`` for metric scoring — only the prompt is truncated here.
    """
    n = cfg.rerank.top_n if cfg.rerank.enabled else cfg.generation.context_chunks
    return hits[:n]
