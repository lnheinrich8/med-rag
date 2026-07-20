"""`rag` command-line entrypoint.

Step 1 ships working DB plumbing (`info`, `init-db`) and typed stubs for the
pipeline commands (`ingest`, `search`, `query`, `eval`) that later steps fill in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
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
    force: bool = typer.Option(
        False, "--force", help="Re-ingest even if content + pipeline config are unchanged."
    ),
) -> None:
    """Parse → chunk → embed → upsert into Postgres. (Step 2)"""
    from rich.progress import track

    from .config import ExperimentConfig
    from .ingest.pipeline import ingest_corpus

    cfg = ExperimentConfig.from_yaml(config)
    target = path or settings.data_dir
    if not (target.is_file() or any(target.rglob("*.pdf"))):
        console.print(f"[yellow]No PDFs found under[/] {target}")
        raise typer.Exit()

    r = ingest_corpus(
        cfg, target=target, force=force,
        progress=lambda docs: track(docs, description="Ingesting"),
    )
    console.print(
        f"[green]Ingested[/] {r['stored']} document(s), {r['total_chunks']} chunk(s); "
        f"{r['skipped']} unchanged/skipped."
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="Retrieval query."),
    config: Path = typer.Option("configs/default.yaml", "--config", "-c"),
    k: Optional[int] = typer.Option(None, "--k", help="Override top_k from config."),
) -> None:
    """Retrieval only: show top chunks with scores. (Step 2)"""
    from .config import ExperimentConfig
    from .retrieve.search import retrieve

    cfg = ExperimentConfig.from_yaml(config)
    if k is not None:
        cfg.retrieval.top_k = k

    hits = retrieve(query, cfg)
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


def _render_sources(citations) -> None:
    """Sources table for an answer; states plainly when nothing was cited."""
    if not citations:
        console.print("[yellow]No sources cited.[/]")
        return
    table = Table(title="Sources", show_header=True, title_justify="left")
    table.add_column("[n]", justify="right", style="cyan", no_wrap=True)
    table.add_column("source")
    for c in citations:
        table.add_row(f"[{c.marker}]", c.label())
    console.print(table)


def _render_answer(ans, show_context: bool, show_sources: bool = True) -> None:
    """Print an Answer (panel + sources + optional context + latency footer).

    Shared by `query` and `chat`. Chat suppresses the sources table — there
    it's on demand via /sources — while one-shot `query` keeps it inline.
    """
    console.print(Panel(ans.text, title="Answer", title_align="left", border_style="green"))
    if show_sources:
        _render_sources(ans.citations)

    if show_context:
        for i, h in enumerate(ans.contexts, 1):
            console.print(f"[dim][{i}][/] [cyan]{Path(h.source_path).name}[/] ({h.score:.3f})")
            console.print(" ".join(h.content.split())[:300])

    console.print(
        f"[dim]retrieval {ans.retrieval_s * 1000:.0f}ms · "
        f"generation {ans.generation_s:.1f}s · {ans.completion_tokens} tokens[/]"
    )


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to answer with citations."),
    config: Path = typer.Option("configs/default.yaml", "--config", "-c"),
    show_context: bool = typer.Option(
        False, "--show-context", help="Also print the retrieved chunks."
    ),
) -> None:
    """Full RAG: retrieve + generate a cited answer. (Step 3)"""
    from .config import ExperimentConfig
    from .generate.answer import answer_question

    cfg = ExperimentConfig.from_yaml(config)
    with console.status("Retrieving + generating…"):
        ans = answer_question(question, cfg)
    _render_answer(ans, show_context)


# Slash commands available inside `rag chat`. The registry is the single
# source of truth for both the completion menu and dispatch.
_CHAT_COMMANDS = {
    "/sources": "Show the sources cited by the last answer",
    "/clear": "Clear the screen and forget the last answer",
}


def _slash_matches(text: str) -> list[tuple[str, str]]:
    """Chat commands matching a partially typed command.

    Matches only while the line is nothing but a leading slash-token, so the
    menu appears for `/so` but never for `/sources x` or mid-sentence slashes.
    """
    if not text.startswith("/") or " " in text:
        return []
    return [(name, help_) for name, help_ in _CHAT_COMMANDS.items() if name.startswith(text)]


def _run_chat_command(raw: str, last_ans, on_clear=None):
    """Execute a /command typed in chat; returns the (possibly reset) last_ans.

    Unique prefixes dispatch too (`/so`, `/c`). `on_clear` is called after
    /clear wipes the screen, so the chat loop can repaint its banner.
    """
    token = raw.split()[0]
    matches = [token] if token in _CHAT_COMMANDS else [n for n, _ in _slash_matches(token)]
    if len(matches) != 1:
        console.print(f"[red]Unknown command:[/] {token}. Available:")
        for name, help_ in _CHAT_COMMANDS.items():
            console.print(f"  [cyan]{name}[/] — {help_}")
        return last_ans
    if matches[0] == "/sources":
        if last_ans is None:
            console.print("[yellow]No answer yet — ask a question first.[/]")
        else:
            _render_sources(last_ans.citations)
        return last_ans
    if matches[0] == "/clear":
        console.clear()
        if on_clear is not None:
            on_clear()
        return None
    return last_ans


@app.command()
def chat(
    config: Path = typer.Option("configs/default.yaml", "--config", "-c"),
    show_context: bool = typer.Option(
        False, "--show-context", help="Also print the retrieved chunks for each answer."
    ),
) -> None:
    """Interactive RAG chat: ask question after question, no quoting needed.

    Type a question and press Enter. A leading `/` runs a chat command
    (`/sources` re-shows the last answer's sources) with a completion menu as
    you type. Ctrl+C clears the current line; pressing Ctrl+C again on an
    empty line (or Ctrl+D) exits.
    """
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings

    from .config import ExperimentConfig
    from .generate.answer import answer_question

    cfg = ExperimentConfig.from_yaml(config)

    # Ctrl+C clears a non-empty line; on an empty line it quits (like the Claude
    # CLI). Overrides prompt_toolkit's default, which always aborts.
    bindings = KeyBindings()

    @bindings.add("c-c")
    def _(event) -> None:
        buf = event.app.current_buffer
        if buf.text:
            buf.reset()
        else:
            event.app.exit(exception=KeyboardInterrupt)

    class SlashCompleter(Completer):
        """Pop up matching /commands while the line is just a command token."""

        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            for name, help_ in _slash_matches(text):
                yield Completion(name, start_position=-len(text), display_meta=help_)

    session: PromptSession = PromptSession(
        key_bindings=bindings,
        history=InMemoryHistory(),
        completer=SlashCompleter(),
        complete_while_typing=True,
    )

    def banner() -> None:
        console.print(
            Panel(
                f"Ask anything about the corpus — just type and press Enter.\n"
                f"[dim]config: {cfg.name} · / for commands · "
                f"Ctrl+C clears the line, Ctrl+C again (empty) quits.\n"
                f"The first answer is slow while the models load.[/]",
                title="med-rag chat",
                title_align="left",
                border_style="cyan",
            )
        )

    banner()

    last_ans = None
    while True:
        try:
            question = session.prompt(HTML("\n<b><ansigreen>you</ansigreen></b> › ")).strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not question:
            continue
        if question.startswith("/"):
            last_ans = _run_chat_command(question, last_ans, on_clear=banner)
            continue
        try:
            with console.status("Thinking…"):
                ans = answer_question(question, cfg)
        except Exception as exc:  # noqa: BLE001 - keep the session alive on any failure
            console.print(f"[red]Error:[/] {exc}")
            continue
        last_ans = ans
        _render_answer(ans, show_context, show_sources=False)

    console.print("\n[dim]Bye.[/]")


_DEFAULT_GOLD = Path("data/gold/diabetes_qa.jsonl")


@app.command("eval-gen")
def eval_gen(
    config: Path = typer.Option("configs/default.yaml", "--config", "-c"),
    n: int = typer.Option(50, "--n", "-n", help="Number of draft questions to generate."),
    out: Path = typer.Option(
        Path("data/gold/diabetes_qa.draft.jsonl"), "--out", "-o", help="Draft JSONL path."
    ),
    min_chars: int = typer.Option(400, "--min-chars", help="Skip chunks shorter than this."),
) -> None:
    """Draft a gold Q&A set from the corpus for hand-verification. (Step 4)"""
    from .config import ExperimentConfig
    from .eval.dataset import save_gold
    from .eval.generate_gold import generate_gold

    cfg = ExperimentConfig.from_yaml(config)
    with console.status(f"Drafting up to {n} questions with {cfg.generation.model}…"):
        drafts = generate_gold(cfg.generation, n=n, min_chars=min_chars)

    if not drafts:
        console.print("[yellow]No drafts produced.[/] Did you run `rag ingest` first?")
        raise typer.Exit(1)

    save_gold(out, drafts)
    console.print(
        f"[green]Drafted[/] {len(drafts)} question(s) → {out}\n"
        "[dim]Next: review each line, fix the answer/relevant span, set "
        '"verified": true, and save as data/gold/diabetes_qa.jsonl.[/]'
    )


@app.command("eval")
def eval_(
    config: Path = typer.Option("configs/default.yaml", "--config", "-c"),
    gold: Path = typer.Option(_DEFAULT_GOLD, "--gold", "-g", help="Gold JSONL path."),
    no_generate: bool = typer.Option(
        False, "--no-generate", help="Retrieval metrics only (skip the LLM)."
    ),
    no_judge: bool = typer.Option(
        False, "--no-judge", help="Generate answers but skip LLM-as-judge scoring."
    ),
    verified_only: bool = typer.Option(
        False, "--verified-only", help="Score only questions marked verified=true."
    ),
) -> None:
    """Run the evaluation suite over the gold set and emit a report. (Step 4)"""
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from .config import ExperimentConfig
    from .eval.dataset import load_gold
    from .eval.report import build_table, save_report, summarize
    from .eval.runner import run_eval

    if not gold.exists():
        console.print(
            f"[red]Gold set not found:[/] {gold}\n"
            "Generate a draft with [cyan]rag eval-gen[/], verify it, then rerun."
        )
        raise typer.Exit(1)

    cfg = ExperimentConfig.from_yaml(config)
    questions = load_gold(gold, verified_only=verified_only)
    if not questions:
        scope = "verified " if verified_only else ""
        console.print(f"[yellow]No {scope}questions in[/] {gold}.")
        raise typer.Exit(1)

    generate = not no_generate
    judge = generate and not no_judge
    console.print(
        f"Evaluating [cyan]{cfg.name}[/] over {len(questions)} question(s) "
        f"(generate={generate}, judge={judge})…"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running", total=len(questions))
        run = run_eval(
            questions,
            cfg,
            generate=generate,
            judge=judge,
            on_result=lambda _r: progress.advance(task),
        )

    summary = summarize(run)
    console.print(build_table(summary))
    json_path, md_path = save_report(run, summary, settings.reports_dir)
    console.print(f"[green]Saved[/] {json_path}\n[green]Saved[/] {md_path}")


@app.command()
def ablate(
    configs: list[Path] = typer.Option(
        ..., "--config", "-c", help="Configs to ingest+evaluate in order (repeat -c)."
    ),
    gold: Path = typer.Option(_DEFAULT_GOLD, "--gold", "-g", help="Gold JSONL path."),
    no_generate: bool = typer.Option(False, "--no-generate", help="Retrieval metrics only."),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip LLM-as-judge scoring."),
    verified_only: bool = typer.Option(False, "--verified-only"),
) -> None:
    """Ingest + evaluate several configs in one run, then print a comparison. (Step 6)

    Each config is re-ingested (chunking lives in the DB) and fully evaluated; the
    DB is left in the state of the LAST config. A side-by-side comparison table and
    a markdown report are written under reports/.
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn, track

    from .config import ExperimentConfig
    from .eval.dataset import load_gold
    from .eval.report import build_table, compare_table, save_comparison, save_report, summarize
    from .eval.runner import run_eval
    from .ingest.pipeline import ingest_corpus

    if not gold.exists():
        console.print(f"[red]Gold set not found:[/] {gold}")
        raise typer.Exit(1)
    questions = load_gold(gold, verified_only=verified_only)
    if not questions:
        console.print(f"[yellow]No questions in[/] {gold}.")
        raise typer.Exit(1)

    generate = not no_generate
    judge = generate and not no_judge
    items: list[tuple[str, dict]] = []

    for config in configs:
        cfg = ExperimentConfig.from_yaml(config)
        console.rule(f"[bold]{cfg.name}[/]")

        console.print(f"Re-ingesting for [cyan]{cfg.name}[/] (chunk={cfg.chunk.chunk_size}…)")
        r = ingest_corpus(cfg, progress=lambda docs: track(docs, description="Ingesting"))
        console.print(f"  {r['stored']} stored, {r['skipped']} unchanged, {r['total_chunks']} chunks")

        console.print(f"Evaluating {len(questions)} question(s) (generate={generate}, judge={judge})…")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running", total=len(questions))
            run = run_eval(
                questions, cfg, generate=generate, judge=judge,
                on_result=lambda _r: progress.advance(task),
            )
        summary = summarize(run)
        console.print(build_table(summary))
        save_report(run, summary, settings.reports_dir)
        items.append((cfg.name, summary))

    if len(items) > 1:
        console.rule("[bold]comparison[/]")
        console.print(compare_table(items))
        path = save_comparison(items, settings.reports_dir)
        console.print(f"[green]Saved[/] {path}")


@app.command("report-compare")
def report_compare(
    reports: list[Path] = typer.Argument(..., help="Saved eval report JSON files to compare."),
) -> None:
    """Print a side-by-side comparison of already-saved eval reports."""
    import json

    from .eval.report import compare_table, save_comparison

    items: list[tuple[str, dict]] = []
    for p in reports:
        data = json.loads(Path(p).read_text())
        items.append((data["summary"]["config"], data["summary"]))
    console.print(compare_table(items))
    if len(items) > 1:
        path = save_comparison(items, settings.reports_dir)
        console.print(f"[green]Saved[/] {path}")


@app.command()
def config_show(
    config: Path = typer.Option("configs/default.yaml", "--config", "-c"),
) -> None:
    """Validate and print an experiment config."""
    cfg = ExperimentConfig.from_yaml(config)
    console.print_json(cfg.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
