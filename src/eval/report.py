"""Aggregate an EvalRun into summary metrics and persist a tagged report.

Produces three things from the raw per-question results:

* a ``summary`` dict (mean retrieval metrics, generation quality, latency
  percentiles) — the headline numbers,
* a Rich table for the terminal,
* a JSON + Markdown pair written under ``reports/`` and tagged with the config
  name and a timestamp, so ablation runs (Step 6) accumulate as comparable,
  reproducible artifacts rather than scrolling past in the terminal.

Latency is reported as p50/p95, not just a mean: tail latency is what a user
feels, and the mean hides it.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from rich.table import Table

from .metrics import percentile
from .runner import EvalRun


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(run: EvalRun) -> dict:
    """Reduce per-question results to headline retrieval/generation/latency stats."""
    results = run.results
    n = len(results)

    # Retrieval: average each metric key across questions.
    metric_keys = [f"recall@{k}" for k in run.ks]
    metric_keys += [f"precision@{k}" for k in run.ks]
    metric_keys += [f"ndcg@{k}" for k in run.ks]
    metric_keys += ["mrr"]
    retrieval = {key: _mean([r.metrics[key] for r in results]) for key in metric_keys}

    n_errors = sum(1 for r in results if r.error)
    summary: dict = {
        "config": run.config_name,
        "n_questions": n,
        "n_errors": n_errors,
        "ks": list(run.ks),
        "retrieval": retrieval,
        "latency": {
            "retrieval_ms_p50": percentile([r.retrieval_s * 1000 for r in results], 50),
            "retrieval_ms_p95": percentile([r.retrieval_s * 1000 for r in results], 95),
        },
    }

    if run.generate:
        gen = [r for r in results if r.generated]
        abstained = [r for r in gen if r.abstained]
        gen_s = [r.generation_s for r in gen]
        summary["generation"] = {
            "n_generated": len(gen),
            "abstention_rate": len(abstained) / len(gen) if gen else 0.0,
            "mean_citations": _mean([r.n_citations for r in gen]),
            "mean_completion_tokens": _mean([r.completion_tokens for r in gen]),
        }
        summary["latency"]["generation_s_p50"] = percentile(gen_s, 50)
        summary["latency"]["generation_s_p95"] = percentile(gen_s, 95)

    if run.judge:
        faith = [r.faithfulness for r in results if r.faithfulness is not None]
        corr = [r.correctness for r in results if r.correctness is not None]
        summary["judge"] = {
            "n_judged": len(faith),
            "mean_faithfulness": _mean(faith),  # 1-5
            "mean_correctness": _mean(corr),  # 1-5
            "note": "self-judged by the generator model; absolute values carry "
            "self-preference bias — use for relative comparison across configs.",
        }
    return summary


def build_table(summary: dict) -> Table:
    """Render the summary as a Rich table for the terminal."""
    table = Table(
        title=f"eval · {summary['config']} · n={summary['n_questions']}",
        show_header=True,
        title_justify="left",
    )
    table.add_column("metric")
    table.add_column("value", justify="right")

    for key, val in summary["retrieval"].items():
        table.add_row(key, f"{val:.3f}")

    if summary.get("n_errors"):
        table.add_section()
        table.add_row("[red]errored questions[/]", f"[red]{summary['n_errors']}[/]")

    lat = summary["latency"]
    table.add_section()
    table.add_row("retrieval p50 (ms)", f"{lat['retrieval_ms_p50']:.0f}")
    table.add_row("retrieval p95 (ms)", f"{lat['retrieval_ms_p95']:.0f}")

    if "generation" in summary:
        g = summary["generation"]
        table.add_section()
        table.add_row("abstention rate", f"{g['abstention_rate']:.1%}")
        table.add_row("mean citations", f"{g['mean_citations']:.2f}")
        table.add_row("generation p50 (s)", f"{lat['generation_s_p50']:.1f}")
        table.add_row("generation p95 (s)", f"{lat['generation_s_p95']:.1f}")

    if "judge" in summary:
        j = summary["judge"]
        table.add_section()
        table.add_row("mean faithfulness (1-5)", f"{j['mean_faithfulness']:.2f}")
        table.add_row("mean correctness (1-5)", f"{j['mean_correctness']:.2f}")
        table.add_row("[dim]judged[/]", f"[dim]{j['n_judged']}[/]")

    return table


def _to_markdown(summary: dict) -> str:
    lines = [
        f"# eval report — `{summary['config']}`",
        "",
        f"- questions: **{summary['n_questions']}**",
        f"- cutoffs (k): {summary['ks']}",
        "",
        "## Retrieval",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    lines += [f"| {k} | {v:.3f} |" for k, v in summary["retrieval"].items()]

    lat = summary["latency"]
    lines += [
        "",
        "## Latency",
        "",
        "| stage | p50 | p95 |",
        "| --- | ---: | ---: |",
        f"| retrieval (ms) | {lat['retrieval_ms_p50']:.0f} | {lat['retrieval_ms_p95']:.0f} |",
    ]
    if "generation_s_p50" in lat:
        lines.append(
            f"| generation (s) | {lat['generation_s_p50']:.1f} | {lat['generation_s_p95']:.1f} |"
        )

    if "generation" in summary:
        g = summary["generation"]
        lines += [
            "",
            "## Generation",
            "",
            f"- abstention rate: **{g['abstention_rate']:.1%}**",
            f"- mean citations / answer: {g['mean_citations']:.2f}",
            f"- mean completion tokens: {g['mean_completion_tokens']:.0f}",
        ]
    if "judge" in summary:
        j = summary["judge"]
        lines += [
            "",
            "## Judge (LLM-as-judge, 1-5)",
            "",
            f"- mean faithfulness: **{j['mean_faithfulness']:.2f}**",
            f"- mean correctness: **{j['mean_correctness']:.2f}**",
            f"- judged: {j['n_judged']}",
            "",
            f"> {j['note']}",
        ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Cross-config comparison (Step 6: baseline vs tuned, ablation matrices)
# --------------------------------------------------------------------------- #


def _f3(v: float) -> str:
    return f"{v:.3f}"


def _f2(v: float) -> str:
    return f"{v:.2f}"


def _f1(v: float) -> str:
    return f"{v:.1f}"


def _f0(v: float) -> str:
    return f"{v:.0f}"


def _pct(v: float) -> str:
    return f"{v:.1%}"


# (summary key path, display label, formatter, "up"=higher-is-better)
_COMPARE_ROWS = [
    ("retrieval.recall@1", "recall@1", _f3, "up"),
    ("retrieval.recall@5", "recall@5", _f3, "up"),
    ("retrieval.recall@20", "recall@20", _f3, "up"),
    ("retrieval.mrr", "MRR", _f3, "up"),
    ("retrieval.ndcg@5", "nDCG@5", _f3, "up"),
    ("generation.abstention_rate", "abstention", _pct, "down"),
    ("judge.mean_faithfulness", "faithfulness (1-5)", _f2, "up"),
    ("judge.mean_correctness", "correctness (1-5)", _f2, "up"),
    ("generation.mean_citations", "citations/answer", _f2, "up"),
    ("latency.retrieval_ms_p50", "retrieval p50 (ms)", _f0, "down"),
    ("latency.generation_s_p50", "generation p50 (s)", _f1, "down"),
]


def _dig(summary: dict, dotted: str):
    """Fetch a nested value by 'a.b' path; None if absent."""
    node = summary
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _delta_cell(fmt, first: float, last: float) -> str:
    """Signed delta string in the same units as the metric (pp for percentages)."""
    d = last - first
    if fmt is _pct:
        return f"{d * 100:+.1f}pp"
    digits = {_f3: 3, _f2: 2, _f1: 1, _f0: 0}.get(fmt, 3)
    return f"{d:+.{digits}f}"


def compare_table(items: list[tuple[str, dict]]) -> Table:
    """Side-by-side metric comparison across configs; +Δ column when exactly two."""
    table = Table(title="comparison", show_header=True, title_justify="left")
    table.add_column("metric")
    for name, _ in items:
        table.add_column(name, justify="right")
    show_delta = len(items) == 2
    if show_delta:
        table.add_column("Δ", justify="right")

    for path, label, fmt, direction in _COMPARE_ROWS:
        vals = [_dig(s, path) for _, s in items]
        if all(v is None for v in vals):
            continue
        cells = [fmt(v) if v is not None else "—" for v in vals]
        if show_delta and vals[0] is not None and vals[1] is not None:
            improved = (vals[1] > vals[0]) == (direction == "up")
            same = vals[1] == vals[0]
            color = "dim" if same else ("green" if improved else "red")
            cells.append(f"[{color}]{_delta_cell(fmt, vals[0], vals[1])}[/]")
        elif show_delta:
            cells.append("—")
        table.add_row(label, *cells)
    return table


def _comparison_markdown(items: list[tuple[str, dict]]) -> str:
    names = [name for name, _ in items]
    head = "| metric | " + " | ".join(names) + " |"
    sep = "| --- " + "| ---: " * len(names) + "|"
    lines = ["# eval comparison", "", head, sep]
    for path, label, fmt, _ in _COMPARE_ROWS:
        vals = [_dig(s, path) for _, s in items]
        if all(v is None for v in vals):
            continue
        cells = [fmt(v) if v is not None else "—" for v in vals]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def save_comparison(
    items: list[tuple[str, dict]], reports_dir: Path, stamp: str | None = None
) -> Path:
    """Write a Markdown side-by-side comparison of several config summaries."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    names = "_vs_".join(name for name, _ in items)
    path = reports_dir / f"compare_{names}_{stamp}.md"
    path.write_text(_comparison_markdown(items), encoding="utf-8")
    return path


def save_report(
    run: EvalRun, summary: dict, reports_dir: Path, stamp: str | None = None
) -> tuple[Path, Path]:
    """Write `<config>_<timestamp>.json` (full) and `.md` (summary); return paths."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    base = reports_dir / f"{run.config_name}_{stamp}"

    payload = {
        "summary": summary,
        "config": run.config_name,
        "generate": run.generate,
        "judge": run.judge,
        "results": [asdict(r) for r in run.results],
    }
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(summary), encoding="utf-8")
    return json_path, md_path
