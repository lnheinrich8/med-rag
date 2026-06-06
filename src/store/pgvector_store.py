"""Persist Documents + Chunks into Postgres (idempotent).

Re-ingesting is cheap and safe: a document whose content_hash already matches the
stored row is skipped entirely. Otherwise the documents row is upserted (keyed on
source_path) and its chunks are fully replaced — the ON DELETE CASCADE on the FK
plus an explicit delete keep things consistent.

Connections are opened by the caller (so a whole ingest run shares one
transaction); these functions just issue SQL on the given connection. The
pgvector adapter is registered in db.connect(), so a Python list binds directly
to the VECTOR column.
"""

from __future__ import annotations

import hashlib

import psycopg

from ..models import Chunk, Document


def ingest_hash(doc: Document, signature: str = "") -> str:
    """The value stored in documents.content_hash to drive idempotent re-ingest.

    It must change whenever *anything that affects what we store* changes — not
    just the document text. ``signature`` carries the ingest-time pipeline knobs
    (chunk strategy/size/overlap, embed model/dim) so that re-chunking or swapping
    the embedder is correctly detected as "changed" instead of skipped. Without
    this, ``content_hash = sha256(full_text)`` alone made chunk/embed ablations
    silently no-op.
    """
    if not signature:
        return doc.content_hash
    return hashlib.sha256(f"{doc.content_hash}|{signature}".encode("utf-8")).hexdigest()


def is_unchanged(conn: psycopg.Connection, doc: Document, signature: str = "") -> bool:
    """True if this source_path is already stored with the same ingest signature."""
    row = conn.execute(
        "SELECT content_hash FROM documents WHERE source_path = %s",
        (str(doc.source_path),),
    ).fetchone()
    return row is not None and row[0] == ingest_hash(doc, signature)


def upsert_document(conn: psycopg.Connection, doc: Document, signature: str = "") -> int:
    """Insert or update the documents row (keyed on source_path); return its id."""
    row = conn.execute(
        """
        INSERT INTO documents (source_path, source_type, title, section_number,
                               n_pages, content_hash)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (source_path) DO UPDATE SET
            source_type    = EXCLUDED.source_type,
            title          = EXCLUDED.title,
            section_number = EXCLUDED.section_number,
            n_pages        = EXCLUDED.n_pages,
            content_hash   = EXCLUDED.content_hash
        RETURNING id
        """,
        (
            str(doc.source_path),
            doc.source_type,
            doc.title,
            doc.section_number,
            doc.n_pages,
            ingest_hash(doc, signature),
        ),
    ).fetchone()
    return row[0]


def insert_chunks(conn: psycopg.Connection, document_id: int, chunks: list[Chunk]) -> None:
    """Replace all chunks for a document: delete the old, batch-insert the new."""
    conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (document_id, chunk_index, content, token_count,
                                page_start, page_end, section, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    document_id,
                    c.chunk_index,
                    c.content,
                    c.token_count,
                    c.page_start,
                    c.page_end,
                    c.section,
                    c.embedding,
                )
                for c in chunks
            ],
        )
    conn.execute(
        "UPDATE documents SET n_chunks = %s WHERE id = %s",
        (len(chunks), document_id),
    )


def store_document(
    conn: psycopg.Connection,
    doc: Document,
    chunks: list[Chunk],
    signature: str = "",
    force: bool = False,
) -> bool:
    """Idempotently store one doc + its chunks. Returns True if written, False if skipped.

    A doc is skipped only when its source_path is already stored under the same
    ingest ``signature`` (content + chunk/embed params). ``force`` re-writes
    regardless — handy for re-embedding without any config change.
    """
    if not force and is_unchanged(conn, doc, signature):
        return False
    document_id = upsert_document(conn, doc, signature)
    insert_chunks(conn, document_id, chunks)
    return True
