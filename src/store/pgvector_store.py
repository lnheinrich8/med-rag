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

import psycopg

from ..models import Chunk, Document


def is_unchanged(conn: psycopg.Connection, doc: Document) -> bool:
    """True if this source_path is already stored with the same content_hash."""
    row = conn.execute(
        "SELECT content_hash FROM documents WHERE source_path = %s",
        (str(doc.source_path),),
    ).fetchone()
    return row is not None and row[0] == doc.content_hash


def upsert_document(conn: psycopg.Connection, doc: Document) -> int:
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
            doc.content_hash,
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


def store_document(conn: psycopg.Connection, doc: Document, chunks: list[Chunk]) -> bool:
    """Idempotently store one doc + its chunks. Returns True if written, False if skipped."""
    if is_unchanged(conn, doc):
        return False
    document_id = upsert_document(conn, doc)
    insert_chunks(conn, document_id, chunks)
    return True
