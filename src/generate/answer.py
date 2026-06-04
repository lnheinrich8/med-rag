"""End-to-end RAG: retrieve -> build prompt -> generate -> attach citations.

This is the spine of `rag query` and the unit Step 4's eval will call per gold
question. It returns an `Answer` carrying the model's text, the `[n]` markers
resolved back to sources, and the contexts used (for faithfulness scoring).
"""

from __future__ import annotations

import re
import time

from ..config import ExperimentConfig
from ..models import Answer, Citation
from ..retrieve.dense import SearchHit
from ..retrieve.search import retrieve, select_contexts
from .llm import LLM
from .prompt import build_messages

_CITATION_RE = re.compile(r"\[(\d+)\]")

IDK = "I don't know based on the provided sources."
_IDK = IDK  # backwards-compatible alias


def is_abstention(text: str) -> bool:
    """True if the answer is the grounded refusal (no support in the sources)."""
    return text.strip().lower().startswith(IDK.lower())


def _extract_citations(text: str, contexts: list[SearchHit]) -> list[Citation]:
    """Resolve each distinct, in-range [n] marker in `text` to its context hit."""
    citations: list[Citation] = []
    seen: set[int] = set()
    for m in _CITATION_RE.finditer(text):
        n = int(m.group(1))
        if n in seen or not (1 <= n <= len(contexts)):
            continue
        seen.add(n)
        h = contexts[n - 1]
        citations.append(
            Citation(
                marker=n,
                chunk_id=h.chunk_id,
                source_path=h.source_path,
                title=h.title,
                page_start=h.page_start,
                page_end=h.page_end,
            )
        )
    return citations


def generate_answer(
    question: str,
    contexts: list[SearchHit],
    cfg: ExperimentConfig,
    retrieval_s: float = 0.0,
) -> Answer:
    """Generate + cite an answer from already-retrieved contexts.

    Split out of ``answer_question`` so callers that already have the ranked
    hits (notably the eval runner, which retrieves top_k for metrics) can reuse
    them instead of embedding the query and searching a second time.
    """
    if not contexts:
        return Answer(
            question=question,
            text=IDK,
            citations=[],
            contexts=[],
            retrieval_s=retrieval_s,
        )

    completion = LLM(cfg.generation).complete(build_messages(question, contexts))
    return Answer(
        question=question,
        text=completion.text,
        citations=_extract_citations(completion.text, contexts),
        contexts=contexts,
        retrieval_s=retrieval_s,
        generation_s=completion.latency_s,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
    )


def answer_question(question: str, cfg: ExperimentConfig) -> Answer:
    """Retrieve, generate a grounded answer, and resolve its citations."""
    t0 = time.perf_counter()
    hits = retrieve(question, cfg)
    contexts = select_contexts(hits, cfg)
    retrieval_s = time.perf_counter() - t0
    return generate_answer(question, contexts, cfg, retrieval_s)
