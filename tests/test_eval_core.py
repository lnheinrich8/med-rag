"""Unit tests for the eval core — metrics + gold-set (de)serialization.

These are deliberately model-free and DB-free: the retrieval metrics and the
relevance-matching rules are the part of the eval harness most prone to silent
off-by-one / granularity bugs, so they get checked in isolation. Run with::

    pytest tests/test_eval_core.py
"""

from __future__ import annotations

from dataclasses import dataclass

from src.eval.dataset import GoldQuestion, RelevantPassage, load_gold, save_gold
from src.eval.metrics import (
    evaluate_retrieval,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    relevance_flags,
)


@dataclass
class FakeHit:
    """Stand-in for retrieve.dense.SearchHit — only the fields the code reads."""

    source_path: str
    page_start: int | None = None
    page_end: int | None = None
    chunk_id: int = 0  # used by RRF fusion
    score: float = 0.0  # RRF/rerank write the fused/cross-encoder score here


# --------------------------------------------------------------------------- #
# RelevantPassage.matches
# --------------------------------------------------------------------------- #


def test_match_basename_and_page_overlap():
    p = RelevantPassage("doc.pdf", page_start=5, page_end=8)
    # same file (absolute path), overlapping page range
    assert p.matches(FakeHit("/abs/path/doc.pdf", 7, 9))
    # touching the boundary still overlaps
    assert p.matches(FakeHit("/x/doc.pdf", 8, 12))
    assert p.matches(FakeHit("/x/doc.pdf", 1, 5))


def test_match_rejects_wrong_file_and_disjoint_pages():
    p = RelevantPassage("doc.pdf", page_start=5, page_end=8)
    assert not p.matches(FakeHit("/x/other.pdf", 6, 6))  # wrong file
    assert not p.matches(FakeHit("/x/doc.pdf", 1, 4))  # below range
    assert not p.matches(FakeHit("/x/doc.pdf", 9, 11))  # above range


def test_match_whole_doc_when_pages_absent():
    p = RelevantPassage("doc.pdf")  # no page range -> anywhere in the file
    assert p.matches(FakeHit("/x/doc.pdf", 1, 2))
    assert p.matches(FakeHit("/x/doc.pdf", 99, 100))
    assert not p.matches(FakeHit("/x/nope.pdf", 1, 2))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _ranking():
    # gold: pages 5-8 of doc.pdf, plus page 2 of other.pdf  (two passages)
    gold = [
        RelevantPassage("doc.pdf", 5, 8),
        RelevantPassage("other.pdf", 2, 2),
    ]
    # ranked hits: rel, irrel, rel(same passage as #1), irrel, rel(other.pdf)
    hits = [
        FakeHit("/x/doc.pdf", 6, 7),  # 1 relevant (passage A)
        FakeHit("/x/doc.pdf", 1, 2),  # 2 irrelevant
        FakeHit("/x/doc.pdf", 7, 8),  # 3 relevant (passage A again)
        FakeHit("/x/foo.pdf", 1, 1),  # 4 irrelevant
        FakeHit("/x/other.pdf", 2, 2),  # 5 relevant (passage B)
    ]
    return hits, gold


def test_relevance_flags():
    hits, gold = _ranking()
    assert relevance_flags(hits, gold) == [True, False, True, False, True]


def test_recall_is_passage_level_not_hit_level():
    hits, gold = _ranking()
    # top-3 hits include two relevant chunks but they're the SAME passage A,
    # so recall must be 1/2, not 2/2.
    assert recall_at_k(hits, gold, 3) == 0.5
    # top-5 covers both passages
    assert recall_at_k(hits, gold, 5) == 1.0
    assert recall_at_k(hits, gold, 1) == 0.5


def test_precision_at_k():
    hits, gold = _ranking()
    flags = relevance_flags(hits, gold)
    assert precision_at_k(flags, 1) == 1.0
    assert precision_at_k(flags, 2) == 0.5
    assert precision_at_k(flags, 5) == 0.6  # 3 of 5 relevant


def test_reciprocal_rank():
    assert reciprocal_rank([False, False, True]) == 1 / 3
    assert reciprocal_rank([True, False]) == 1.0
    assert reciprocal_rank([False, False]) == 0.0


def test_ndcg_perfect_and_zero():
    assert ndcg_at_k([True, True, False], 3) == 1.0  # all relevant up top
    assert ndcg_at_k([False, False], 2) == 0.0  # nothing relevant
    # an out-of-order ranking scores strictly between 0 and 1
    score = ndcg_at_k([False, True, True], 3)
    assert 0.0 < score < 1.0


def test_evaluate_retrieval_keys_and_values():
    hits, gold = _ranking()
    m = evaluate_retrieval(hits, gold, ks=(1, 3, 5))
    assert m["recall@5"] == 1.0
    assert m["precision@1"] == 1.0
    assert m["mrr"] == 1.0
    assert set(m) == {
        "mrr",
        "recall@1", "precision@1", "ndcg@1",
        "recall@3", "precision@3", "ndcg@3",
        "recall@5", "precision@5", "ndcg@5",
    }


def test_percentile():
    assert percentile([], 50) == 0.0
    assert percentile([42.0], 95) == 42.0
    assert percentile([1, 2, 3, 4], 50) == 2.5  # interpolated median
    assert percentile([1, 2, 3, 4], 0) == 1
    assert percentile([1, 2, 3, 4], 100) == 4


# --------------------------------------------------------------------------- #
# Gold set JSONL round-trip
# --------------------------------------------------------------------------- #


def test_rrf_fuse_rewards_agreement_and_dedupes():
    from src.retrieve.hybrid import rrf_fuse

    def h(cid):
        return FakeHit(f"/x/doc{cid}.pdf", chunk_id=cid)

    # chunk 2 is rank-1 in sparse and also present in dense; chunk 1 is rank-1
    # in dense only. A chunk seen in BOTH legs should beat one seen in just one.
    dense = [h(1), h(2)]
    sparse = [h(2)]
    fused = rrf_fuse([dense, sparse], rrf_k=60, top_k=10)

    assert sorted(hh.chunk_id for hh in fused) == [1, 2]  # deduped
    assert fused[0].chunk_id == 2  # appears in both legs -> wins
    # fused score = dense rank 2 + sparse rank 1
    assert abs(fused[0].score - (1 / (60 + 2) + 1 / (60 + 1))) < 1e-12


def test_rrf_fuse_truncates_to_top_k():
    from src.retrieve.hybrid import rrf_fuse

    ranking = [FakeHit(f"/x/{i}.pdf", chunk_id=i) for i in range(10)]
    assert len(rrf_fuse([ranking], rrf_k=60, top_k=5)) == 5


def test_select_contexts_honors_rerank_topn():
    from src.config import ExperimentConfig
    from src.retrieve.search import select_contexts

    hits = [FakeHit(f"/x/doc{i}.pdf", i, i) for i in range(20)]

    cfg = ExperimentConfig()  # rerank disabled -> uses generation.context_chunks (5)
    assert len(select_contexts(hits, cfg)) == 5

    cfg.rerank.enabled = True
    cfg.rerank.top_n = 3  # rerank on -> top_n governs how many reach the prompt
    assert len(select_contexts(hits, cfg)) == 3


def test_summarize_counts_errors_and_excludes_them():
    from src.eval.report import summarize
    from src.eval.runner import EvalRun, QuestionResult

    ks = (1, 5)
    def m():
        return ({f"recall@{k}": 0.0 for k in ks} | {f"precision@{k}": 0.0 for k in ks}
                | {f"ndcg@{k}": 0.0 for k in ks} | {"mrr": 0.0})

    ok = QuestionResult(id="q1", question="?", metrics=m(), retrieval_s=0.01, n_relevant=1,
                        n_hits=5, generated=True, abstained=False, faithfulness=4, correctness=4)
    err = QuestionResult(id="q2", question="?", metrics=m(), retrieval_s=0.01, n_relevant=1,
                         n_hits=5, error="RuntimeError: boom")
    run = EvalRun(config_name="x", ks=ks, generate=True, judge=True, results=[ok, err])

    s = summarize(run)
    assert s["n_errors"] == 1
    assert s["generation"]["n_generated"] == 1  # the errored question is excluded
    assert s["judge"]["n_judged"] == 1


def test_comparison_markdown_and_dig():
    from src.eval.report import _comparison_markdown, _dig

    s_base = {"config": "baseline", "retrieval": {"recall@5": 0.794, "mrr": 0.616}}
    s_tuned = {"config": "tuned", "retrieval": {"recall@5": 0.971, "mrr": 0.764}}

    assert _dig(s_base, "retrieval.recall@5") == 0.794
    assert _dig(s_base, "retrieval.nope") is None
    assert _dig(s_base, "judge.mean_correctness") is None  # missing branch

    md = _comparison_markdown([("baseline", s_base), ("tuned", s_tuned)])
    assert "| metric | baseline | tuned |" in md
    assert "recall@5" in md and "0.794" in md and "0.971" in md
    # rows where every config lacks the metric are omitted
    assert "faithfulness" not in md


def test_gold_roundtrip(tmp_path):
    questions = [
        GoldQuestion(
            id="q1",
            question="What is first-line pharmacotherapy for type 2 diabetes?",
            reference_answer="Metformin, alongside lifestyle modification.",
            relevant=[RelevantPassage("ada_s09.pdf", 3, 4)],
            category="pharmacology",
            verified=True,
        ),
        GoldQuestion(
            id="q2",
            question="Define the HbA1c threshold for diabetes diagnosis.",
            reference_answer="An HbA1c of 6.5% or higher.",
            relevant=[RelevantPassage("statpearls_dx.pdf")],
        ),
    ]
    path = tmp_path / "gold.jsonl"
    save_gold(path, questions)

    loaded = load_gold(path)
    assert [q.to_dict() for q in loaded] == [q.to_dict() for q in questions]

    verified = load_gold(path, verified_only=True)
    assert [q.id for q in verified] == ["q1"]
