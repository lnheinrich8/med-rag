"""Retrieval quality metrics (pure functions, no DB / no model).

Everything here operates on a ranked list of retrieved hits (each exposing
``source_path``/``page_start``/``page_end``) and the gold ``RelevantPassage``
labels. Two granularities matter and are easy to conflate:

* **precision / nDCG / MRR** are *hit-level*: each retrieved chunk is relevant or
  not (does it overlap any gold passage?).
* **recall** is *passage-level*: of the distinct gold passages, how many did we
  retrieve? Several chunks can map to one passage, so recall counts covered
  passages, not relevant hits — otherwise duplicate chunks would inflate it.

All functions take ``k`` and look only at the top-k of the ranking.
"""

from __future__ import annotations

import math

from .dataset import RelevantPassage


def relevance_flags(hits: list, gold: list[RelevantPassage]) -> list[bool]:
    """Per-hit booleans: is hit *i* relevant to any gold passage?"""
    return [any(p.matches(h) for p in gold) for h in hits]


def covered_passages(hits: list, gold: list[RelevantPassage], k: int) -> int:
    """How many distinct gold passages are hit by the top-k retrieved chunks."""
    covered = {i for i, p in enumerate(gold) for h in hits[:k] if p.matches(h)}
    return len(covered)


def recall_at_k(hits: list, gold: list[RelevantPassage], k: int) -> float:
    """Fraction of gold passages retrieved within the top-k (passage-level)."""
    if not gold:
        return 0.0
    return covered_passages(hits, gold, k) / len(gold)


def precision_at_k(flags: list[bool], k: int) -> float:
    """Fraction of the top-k retrieved chunks that are relevant (hit-level)."""
    if k <= 0:
        return 0.0
    top = flags[:k]
    return sum(top) / k


def reciprocal_rank(flags: list[bool]) -> float:
    """1 / rank of the first relevant hit (1-based); 0 if none retrieved."""
    for i, rel in enumerate(flags, 1):
        if rel:
            return 1.0 / i
    return 0.0


def _dcg(flags: list[bool]) -> float:
    """Discounted cumulative gain for a binary-relevance ranking."""
    return sum((1.0 if rel else 0.0) / math.log2(i + 2) for i, rel in enumerate(flags))


def ndcg_at_k(flags: list[bool], k: int) -> float:
    """nDCG@k: DCG of the ranking vs. the ideal ranking, over the retrieved set."""
    dcg = _dcg(flags[:k])
    ideal = _dcg(sorted(flags, reverse=True)[:k])
    return dcg / ideal if ideal else 0.0


def evaluate_retrieval(
    hits: list,
    gold: list[RelevantPassage],
    ks: tuple[int, ...] = (1, 3, 5, 10, 20),
) -> dict[str, float]:
    """All retrieval metrics for one question's ranking, keyed like ``recall@5``."""
    flags = relevance_flags(hits, gold)
    out: dict[str, float] = {"mrr": reciprocal_rank(flags)}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(hits, gold, k)
        out[f"precision@{k}"] = precision_at_k(flags, k)
        out[f"ndcg@{k}"] = ndcg_at_k(flags, k)
    return out


def percentile(values: list[float], q: float) -> float:
    """The q-th percentile (q in [0, 100]) via linear interpolation; 0.0 if empty."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100) * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)
