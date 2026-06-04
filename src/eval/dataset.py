"""Gold evaluation set: the questions + the relevance labels we score against.

The single most important design choice here is how *relevance* is keyed. We do
**not** point at chunk ids — those are an artifact of one chunking/embedding run,
so they'd be invalidated the moment an ablation re-chunks the corpus. Instead a
relevant passage is `(source filename, page range)`. That survives re-chunking,
re-embedding, and even a switch of vector store, so the same gold set scores
every pipeline configuration fairly.

The gold set is stored as JSONL (one question per line) under ``data/gold/`` and
is committed to the repo — it's hand-verified, so it's an asset, not a cache.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RelevantPassage:
    """A span of a source document that answers a gold question.

    Matching is by source *filename* (basename), so absolute paths differing
    across machines don't matter, plus an (optional) inclusive page range. A
    retrieved chunk counts as hitting this passage when it comes from the same
    file and its page span overlaps ``[page_start, page_end]``. Omitting the
    pages means "anywhere in this document" (whole-doc relevance).
    """

    source: str  # filename basename, e.g. "ada_soc_2026_s09_pharmacology.pdf"
    page_start: int | None = None
    page_end: int | None = None

    def matches(self, hit) -> bool:
        """True if a retrieved hit (anything with source_path/page_start/page_end) overlaps."""
        if Path(hit.source_path).name != self.source:
            return False
        if self.page_start is None or hit.page_start is None:
            return True  # whole-doc relevance, or hit has no page info
        hit_end = hit.page_end or hit.page_start
        gold_end = self.page_end or self.page_start
        # inclusive integer-range overlap
        return hit.page_start <= gold_end and self.page_start <= hit_end


@dataclass
class GoldQuestion:
    """One evaluation item: a question, its reference answer, and what's relevant.

    ``verified`` flags whether a human has reviewed the (LLM-drafted) item. The
    eval harness can be told to score only verified questions so an unreviewed
    draft set doesn't silently pollute the metrics.
    """

    id: str
    question: str
    reference_answer: str
    relevant: list[RelevantPassage] = field(default_factory=list)
    category: str | None = None  # e.g. pharmacology | diagnosis | monitoring
    notes: str | None = None
    verified: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GoldQuestion":
        passages = [RelevantPassage(**p) for p in data.get("relevant", [])]
        return cls(
            id=data["id"],
            question=data["question"],
            reference_answer=data.get("reference_answer", ""),
            relevant=passages,
            category=data.get("category"),
            notes=data.get("notes"),
            verified=data.get("verified", False),
        )


def load_gold(path: str | Path, verified_only: bool = False) -> list[GoldQuestion]:
    """Load a JSONL gold file. Blank lines and ``#`` comment lines are ignored."""
    items: list[GoldQuestion] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        items.append(GoldQuestion.from_dict(json.loads(line)))
    if verified_only:
        items = [q for q in items if q.verified]
    return items


def save_gold(path: str | Path, questions: list[GoldQuestion]) -> None:
    """Write questions as JSONL (one compact object per line), creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(q.to_dict(), ensure_ascii=False) for q in questions]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
