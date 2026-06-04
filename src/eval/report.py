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

    summary: dict = {
        "config": run.config_name,
        "n_questions": n,
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
