"""Reusable ingest pipeline: load -> chunk -> embed -> store.

Extracted from the `rag ingest` CLI command so the same logic backs both the
single-config `ingest` and the multi-config `ablate` (Step 6), which re-ingests
each config in turn (chunking lives in the DB, so an ablation that changes
chunk_size must re-chunk before evaluating).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from ..config import ExperimentConfig, settings
from ..db import connect
from ..embed.embedder import Embedder
from ..store.pgvector_store import is_unchanged, store_document
from .chunkers import chunk_document
from .loaders import load_corpus, load_pdf


def ingest_signature(cfg: ExperimentConfig) -> str:
    """The ingest-time identity: content-independent knobs that change what we store."""
    return (
        f"chunk={cfg.chunk.strategy}:{cfg.chunk.chunk_size}:"
        f"{cfg.chunk.chunk_overlap}:{cfg.chunk.clean_boilerplate}"
        f"|embed={cfg.embed.model}:{cfg.embed.dim}:{cfg.embed.normalize}"
    )


def ingest_corpus(
    cfg: ExperimentConfig,
    target: Path | None = None,
    force: bool = False,
    progress: Callable[[Iterable], Iterable] = lambda it: it,
) -> dict:
    """Ingest PDFs under `target` per `cfg`; return {stored, skipped, total_chunks, n_docs}.

    `progress` wraps the per-doc loop (e.g. rich's ``track``); defaults to a no-op.
    """
    target = target or settings.data_dir
    clean = cfg.chunk.clean_boilerplate
    docs = (
        [load_pdf(target, clean=clean)]
        if target.is_file()
        else list(load_corpus(target, clean=clean))
    )

    signature = ingest_signature(cfg)
    embedder = Embedder(cfg.embed)
    stored = skipped = total_chunks = 0
    with connect() as conn:
        for doc in progress(docs):
            # Skip (and don't waste embedding) docs already stored under this signature.
            if not force and is_unchanged(conn, doc, signature):
                skipped += 1
                continue
            chunks = chunk_document(doc, cfg.chunk)
            vectors = embedder.embed_passages([c.content for c in chunks])
            for c, v in zip(chunks, vectors):
                c.embedding = v
            store_document(conn, doc, chunks, signature=signature, force=True)
            stored += 1
            total_chunks += len(chunks)
        conn.commit()

    return {
        "stored": stored,
        "skipped": skipped,
        "total_chunks": total_chunks,
        "n_docs": len(docs),
    }
