"""Prompt construction for grounded, cited generation.

The retrieved chunks are numbered ``[1..N]`` and the model is told to answer
*only* from them and to cite with ``[n]``. Keeping this separate from the LLM
wrapper lets the eval harness snapshot the exact prompt used for any query.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..retrieve.dense import SearchHit

SYSTEM_PROMPT = (
    "You are a clinical information assistant answering questions about type 2 "
    "diabetes using only the provided source excerpts. Rules:\n"
    "- Answer strictly from the excerpts. Do not rely on outside knowledge.\n"
    '- If the excerpts do not contain the answer, reply exactly: "I don\'t know '
    'based on the provided sources."\n'
    "- Cite every claim with bracketed source numbers like [1], or [2][3] when "
    "several support it.\n"
    "- Be concise and clinical. Do not give individualized medical advice."
)


def _source_label(hit: SearchHit) -> str:
    """'filename, p.X-Y' shown alongside each numbered excerpt."""
    name = Path(hit.source_path).name
    if not hit.page_start:
        return name
    span = f"p.{hit.page_start}"
    if hit.page_end and hit.page_end != hit.page_start:
        span += f"-{hit.page_end}"
    return f"{name}, {span}"


def format_context(hits: list[SearchHit]) -> str:
    """Render retrieved chunks as numbered, whitespace-collapsed source blocks."""
    blocks = []
    for i, h in enumerate(hits, 1):
        body = " ".join(h.content.split())
        blocks.append(f"[{i}] ({_source_label(h)})\n{body}")
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[SearchHit]) -> list[dict]:
    """Build the chat messages (system + user) for a RAG query."""
    user = (
        f"Source excerpts:\n\n{format_context(hits)}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the excerpts above, citing sources as [n]."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
