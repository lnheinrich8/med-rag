"""`rag` command-line entrypoint.

Step 1 ships working DB plumbing (`info`, `init-db`) and typed stubs for the
pipeline commands (`ingest`, `search`, `query`, `eval`) that later steps fill in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import ExperimentConfig, settings

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Local, evaluation-focused RAG over a Type 2 Diabetes corpus.",
)
console = Console()

_NOT_IMPLEMENTED = "[yellow]Not implemented yet[/] — arrives in a later build step."


@app.command()
def info() -> None:
    """Show config + database/pgvector health."""
    from .db import health

    table = Table(title=f"med-rag v{__version__}", show_header=False, title_justify="left")
    table.add_row("database_url", settings.database_url)
    table.add_row("data_dir", str(settings.data_dir))
    table.add_row("device", settings.device)

    h = health()
    if not h.get("connected"):
        table.add_row("db", f"[red]unreachable[/] ({h.get('error', 'unknown')})")
    else:
        table.add_row("db", "[green]connected[/]")
        table.add_row("server_version", h.get("server_version", "?"))
        pv = h.get("pgvector")
        table.add_row("pgvector", f"[green]{pv}[/]" if pv else "[red]not installed[/]")
        counts = h.get("tables", {})
        table.add_row("documents", str(counts.get("documents")))
        table.add_row("chunks", str(counts.get("chunks")))
    console.print(table)


@app.command("init-db")
def init_db() -> None:
    """Apply database migrations (idempotent)."""
    from .db import apply_migrations

    applied = apply_migrations()
    if applied:
        console.print(f"[green]Applied[/] {len(applied)} migration(s): {', '.join(applied)}")
    else:
        console.print("[green]Up to date[/] — no migrations to apply.")


@app.command()
def ingest(
    path: Optional[Path] = typer.Argument(None, help="File or dir of PDFs (default: data_dir)."),
    config: Path = typer.Option("configs/default.yaml", "--config", "-c"),
) -> None:
    """Parse → chunk → embed → upsert into Postgres. (Step 2)"""
    from rich.progress import track

    from .config import ExperimentConfig
    from .db import connect
    from .embed.embedder import Embedder
    from .ingest.chunkers import chunk_document
    from .ingest.loaders import load_corpus, load_pdf
    from .store.pgvector_store import store_document

    cfg = ExperimentConfig.from_yaml(config)
    target = path or settings.data_dir
    docs = [load_pdf(target)] if target.is_file() else list(load_corpus(target))
    if not docs:
        console.print(f"[yellow]No PDFs found under[/] {target}")
        raise typer.Exit()

    embedder = Embedder(cfg.embed)
    stored = skipped = total_chunks = 0
    with connect() as conn:
        for doc in track(docs, description="Ingesting"):
            chunks = chunk_document(doc, cfg.chunk)
            vectors = embedder.embed_passages([c.content for c in chunks])
            for c, v in zip(chunks, vectors):
                c.embedding = v
            if store_document(conn, doc, chunks):
                stored += 1
                total_chunks += len(chunks)
            else:
                skipped += 1
        conn.commit()

    console.print(
        f"[green]Ingested[/] {stored} document(s), {total_chunks} chunk(s); "
        f"{skipped} unchanged/skipped."
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="Retrieval query."),
    config: Path = typer.Option("configs/default.yaml", "--config", "-c"),
    k: Optional[int] = typer.Option(None, "--k", help="Override top_k from config."),
) -> None:
    """Retrieval only: show top chunks with scores. (Step 2)"""
    from .config import ExperimentConfig
    from .retrieve.dense import dense_search

    cfg = ExperimentConfig.from_yaml(config)
    if k is not None:
        cfg.retrieval.top_k = k

    hits = dense_search(query, cfg.embed, cfg.retrieval)
    if not hits:
        console.print("[yellow]No results.[/] Did you run `rag ingest` first?")
        raise typer.Exit()

    table = Table(title=f"Top {len(hits)} for: {query}", show_lines=True)
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("score", justify="right")
    table.add_column("source")
    table.add_column("pages", justify="right")
    table.add_column("chunk")
    for i, h in enumerate(hits, 1):
        src = Path(h.source_path).name
        pages = f"{h.page_start}-{h.page_end}" if h.page_start else "?"
        snippet = " ".join(h.content.split())[:160]
        table.add_row(str(i), f"{h.score:.3f}", src, pages, snippet)
    console.print(table)


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to answer with citations."),
    config: Path = typer.Option("configs/default.yaml", "--config", "-c"),
) -> None:
    """Full RAG: retrieve + generate a cited answer. (Step 3)"""
    console.print(_NOT_IMPLEMENTED)


@app.command("eval")
def eval_(
    config: Path = typer.Option("configs/default.yaml", "--config", "-c"),
) -> None:
    """Run the evaluation suite and emit a report. (Step 4)"""
    console.print(_NOT_IMPLEMENTED)


@app.command()
def config_show(
    config: Path = typer.Option("configs/default.yaml", "--config", "-c"),
) -> None:
    """Validate and print an experiment config."""
    cfg = ExperimentConfig.from_yaml(config)
    console.print_json(cfg.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
