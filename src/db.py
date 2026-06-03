"""Postgres connection helpers and migration runner.

Connections register the pgvector adapter so Python lists/np arrays round-trip
to the ``vector`` type transparently.
"""

from __future__ import annotations

from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from .config import PROJECT_ROOT, settings

MIGRATIONS_DIR = PROJECT_ROOT / "migrations"


def connect(autocommit: bool = False) -> psycopg.Connection:
    """Open a connection with the pgvector type adapter registered."""
    conn = psycopg.connect(settings.database_url, autocommit=autocommit)
    register_vector(conn)
    return conn


def _ensure_migrations_table(conn: psycopg.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            filename   TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def apply_migrations() -> list[str]:
    """Apply any *.sql in migrations/ not yet recorded. Returns names applied."""
    applied: list[str] = []
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    # Migrations may create the vector extension, so register the adapter after.
    with psycopg.connect(settings.database_url, autocommit=False) as conn:
        _ensure_migrations_table(conn)
        done = {
            row[0]
            for row in conn.execute("SELECT filename FROM schema_migrations").fetchall()
        }
        for path in files:
            if path.name in done:
                continue
            conn.execute(path.read_text())
            conn.execute(
                "INSERT INTO schema_migrations (filename) VALUES (%s)", (path.name,)
            )
            applied.append(path.name)
        conn.commit()
    return applied


def health() -> dict:
    """Return a snapshot of DB connectivity, pgvector, and corpus counts."""
    info: dict = {"connected": False}
    try:
        with psycopg.connect(settings.database_url) as conn:
            info["connected"] = True
            info["server_version"] = conn.execute("SHOW server_version").fetchone()[0]
            row = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
            info["pgvector"] = row[0] if row else None
            info["tables"] = {
                t: _safe_count(conn, t) for t in ("documents", "chunks")
            }
    except Exception as exc:  # surfaced to the user by the CLI
        info["error"] = str(exc)
    return info


def _safe_count(conn: psycopg.Connection, table: str) -> int | None:
    try:
        return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    except Exception:
        return None  # table not created yet
