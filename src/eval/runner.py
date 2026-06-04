"""Run a pipeline configuration over the gold set and collect per-question results.

For each gold question we retrieve top_k once (scored for retrieval metrics), and
optionally reuse the top context_chunks of that same ranking to generate an
answer and judge it. Retrieving once and slicing keeps the eval honest: the
contexts the generator sees are exactly the head of the ranking the metrics
scored, not a separate search.

This module stays display-free — it returns an ``EvalRun`` of raw results and
takes an optional ``on_result`` callback so the CLI can drive a progress bar.
Aggregation/rendering lives in ``report.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from ..config import ExperimentConfig
from ..generate.answer import generate_answer, is_abstention
from ..retrieve.search import retrieve, select_contexts
from .dataset import GoldQuestion
from .judge import judge_correctness, judge_faithfulness
from .metrics import evaluate_retrieval

DEFAULT_KS = (1, 3, 5, 10, 20)


@dataclass
class QuestionResult:
    id: str
    question: str
    metrics: dict[str, float]
    retrieval_s: float
    n_relevant: int
    n_hits: int
    # generation (only when generate=True)
    generated: bool = False
    answer_text: str | None = None
    abstained: bool = False
    generation_s: float = 0.0
    completion_tokens: int = 0
    n_citations: int = 0
    # judging (only when judge=True and the answer wasn't an abstention)
    faithfulness: float | None = None
    correctness: float | None = None


@dataclass
class EvalRun:
    config_name: str
    ks: tuple[int, ...]
    generate: bool
    judge: bool
    results: list[QuestionResult] = field(default_factory=list)


def _resolve_ks(top_k: int, ks: tuple[int, ...]) -> tuple[int, ...]:
    """Keep only cutoffs the retrieval depth can actually support."""
    usable = tuple(k for k in ks if k <= top_k)
    return usable or (top_k,)


def run_question(
    gold: GoldQuestion,
    cfg: ExperimentConfig,
    ks: tuple[int, ...],
    generate: bool,
    judge: bool,
) -> QuestionResult:
    """Evaluate one gold question: retrieval metrics (+ optional gen/judge)."""
    t0 = time.perf_counter()
    hits = retrieve(gold.question, cfg)
    retrieval_s = time.perf_counter() - t0

    result = QuestionResult(
        id=gold.id,
        question=gold.question,
        metrics=evaluate_retrieval(hits, gold.relevant, ks),
        retrieval_s=retrieval_s,
        n_relevant=len(gold.relevant),
        n_hits=len(hits),
    )
    if not generate:
        return result

    contexts = select_contexts(hits, cfg)
    ans = generate_answer(gold.question, contexts, cfg, retrieval_s)
    result.generated = True
    result.answer_text = ans.text
    result.abstained = is_abstention(ans.text)
    result.generation_s = ans.generation_s
    result.completion_tokens = ans.completion_tokens
    result.n_citations = len(ans.citations)

    # An abstention is trivially "faithful" and trivially not "correct"; judging
    # it would skew both means, so we skip it and report the abstention rate.
    if judge and not result.abstained:
        result.faithfulness = judge_faithfulness(ans.text, contexts, cfg.generation).score
        result.correctness = judge_correctness(
            gold.question, ans.text, gold.reference_answer, cfg.generation
        ).score
    return result


def run_eval(
    gold: list[GoldQuestion],
    cfg: ExperimentConfig,
    ks: tuple[int, ...] = DEFAULT_KS,
    generate: bool = True,
    judge: bool = True,
    on_result: Callable[[QuestionResult], None] | None = None,
) -> EvalRun:
    """Run the whole gold set; calls `on_result` after each item (for progress)."""
    ks = _resolve_ks(cfg.retrieval.top_k, ks)
    run = EvalRun(config_name=cfg.name, ks=ks, generate=generate, judge=judge)
    for gq in gold:
        result = run_question(gq, cfg, ks, generate, judge)
        run.results.append(result)
        if on_result is not None:
            on_result(result)
    return run
