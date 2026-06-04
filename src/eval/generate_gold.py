"""Draft a gold Q&A set by sampling chunks and having the local model write Qs.

This is a *bootstrap*, not the final gold set. We sample content-rich passages
from the corpus, ask the local Qwen to write one self-contained factual question
answerable from each passage plus a concise reference answer, and tag the source
passage as the relevant label. The output is written with ``verified=false`` to a
``.draft.jsonl`` — a human then reviews/edits/prunes it (dropping bad questions,
fixing answers, marking ``verified=true``) to produce the committed gold set.

Sampling deliberately skips reference lists and boilerplate: those chunks are
both unanswerable and the exact noise the dense baseline struggles with, so
generating questions from them would be meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import GenerationConfig
from ..db import connect
from ..generate.llm import LLM
from .dataset import GoldQuestion, RelevantPassage
from .parsing import extract_json

# Pull readable prose chunks; skip short fragments, huge dumps, and reference
# lists (DOIs / "et al." / numbered bibliographies are noise, not answerable).
_SAMPLE_SQL = """
    SELECT c.content, c.section, c.page_start, c.page_end, d.source_path, d.title
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE length(c.content) BETWEEN %s AND 3000
      AND c.content NOT ILIKE '%%doi.org%%'
      AND c.content NOT ILIKE '%%et al.%%'
      AND coalesce(c.section, '') NOT ILIKE '%%reference%%'
    ORDER BY random()
    LIMIT %s
"""

_GEN_SYSTEM = (
    "You write evaluation questions for a clinical retrieval system about type 2 "
    "diabetes. Given one source passage, produce exactly one question that:\n"
    "- is fully answerable from the passage alone,\n"
    "- is self-contained (do NOT say 'according to the passage/text'),\n"
    "- is specific and factual (a clinician could check it),\n"
    "and a concise reference answer (1-3 sentences) grounded only in the passage.\n"
    'Reply with JSON only: {"question": "...", "answer": "..."}.'
)


@dataclass
class _Sample:
    content: str
    section: str | None
    page_start: int | None
    page_end: int | None
    source_path: str
    title: str | None


def _digit_ratio(text: str) -> float:
    if not text:
        return 1.0
    return sum(ch.isdigit() for ch in text) / len(text)


def sample_chunks(n: int, min_chars: int = 400) -> list[_Sample]:
    """Pull up to `n` content-rich chunks at random, filtering numeric/boilerplate."""
    with connect() as conn:
        rows = conn.execute(_SAMPLE_SQL, (min_chars, n * 3)).fetchall()
    samples = [_Sample(*row) for row in rows]
    # Second-pass python filter: drop chunks that are mostly digits (tables,
    # dosing grids, citation runs) — they read poorly as standalone questions.
    samples = [s for s in samples if _digit_ratio(s.content) < 0.12]
    return samples[:n]


def _draft_one(llm: LLM, sample: _Sample, qid: str) -> GoldQuestion | None:
    """Ask the model for one Q&A grounded in `sample`; return a draft GoldQuestion."""
    user = (
        f"SOURCE PASSAGE (from {Path(sample.source_path).name}):\n\n"
        f"{sample.content.strip()}\n\n"
        "Write one question and its reference answer as JSON."
    )
    completion = llm.complete(
        [{"role": "system", "content": _GEN_SYSTEM}, {"role": "user", "content": user}]
    )
    data = extract_json(completion.text)
    if not data or not data.get("question") or not data.get("answer"):
        return None

    preview = " ".join(sample.content.split())[:200]
    return GoldQuestion(
        id=qid,
        question=str(data["question"]).strip(),
        reference_answer=str(data["answer"]).strip(),
        relevant=[
            RelevantPassage(
                source=Path(sample.source_path).name,
                page_start=sample.page_start,
                page_end=sample.page_end,
            )
        ],
        category=sample.section,
        notes=f"DRAFT — review against source. Passage preview: {preview}",
        verified=False,
    )


def generate_gold(
    cfg: GenerationConfig, n: int = 50, min_chars: int = 400
) -> list[GoldQuestion]:
    """Sample `n` chunks and draft one (unverified) gold question per chunk."""
    samples = sample_chunks(n, min_chars=min_chars)
    llm = LLM(cfg)
    drafts: list[GoldQuestion] = []
    for i, sample in enumerate(samples, 1):
        q = _draft_one(llm, sample, qid=f"g{i:04d}")
        if q is not None:
            drafts.append(q)
    return drafts
