"""Unit tests for ingest idempotency keying (src/store/pgvector_store.py).

The bug these guard against: re-ingest was keyed on document text alone, so
changing chunk_size/strategy/embed-model produced the same hash and the store
silently skipped the re-chunk. The signature must change the stored hash.
"""

from __future__ import annotations

from src.models import Document
from src.store.pgvector_store import ingest_hash


def _doc() -> Document:
    return Document(source_path="/x/doc.pdf", source_type="review", pages=["hello world"])


def test_empty_signature_is_plain_content_hash():
    doc = _doc()
    assert ingest_hash(doc, "") == doc.content_hash


def test_signature_changes_the_hash():
    doc = _doc()
    h512 = ingest_hash(doc, "chunk=recursive:512:64:True|embed=bge:768:True")
    h256 = ingest_hash(doc, "chunk=recursive:256:32:True|embed=bge:768:True")
    assert h512 != h256  # different chunk size -> different ingest -> re-chunk
    assert h512 != doc.content_hash  # signature actually participates


def test_same_signature_is_stable():
    doc, sig = _doc(), "chunk=section:512:64:True|embed=bge:768:True"
    assert ingest_hash(doc, sig) == ingest_hash(doc, sig)  # idempotent: skip re-ingest
