"""Recompute aggregate metrics from per-instance JSONL.

Useful for re-aggregating without re-running the generator (e.g. after fixing
a metric definition, or after a generation run was interrupted and you want
to see partial results).

Usage:
    python scripts/05_compute_metrics.py results/C4_cascade.cce.jsonl
    python scripts/05_compute_metrics.py results/*.jsonl   # multi-file table
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import click  # noqa: E402

from adaptive_retrieval.eval.runner import aggregate_from_jsonl  # noqa: E402


@click.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
def main(paths: tuple[str, ...]) -> None:
    rows = []
    for p in paths:
        try:
            s = aggregate_from_jsonl(p)
        except ValueError as e:
            click.echo(f"  SKIP {p}: {e}", err=True)
            continue
        rows.append((Path(p).name, s))

    if not rows:
        click.echo("No usable JSONL.", err=True)
        sys.exit(1)

    print(
        f"{'file':<40} {'cfg':<22} {'n':>4} {'retr%':>6} "
        f"{'EM':>5} {'ES':>5} {'IdF1':>5} {'RSP':>5} {'hall%':>6} {'lat_ms':>8}"
    )
    print("-" * 110)
    for name, s in rows:
        m = s.metrics
        print(
            f"{name:<40} {s.config:<22} {s.n_instances:>4} {s.percent_retrieval:>5.1f} "
            f"{m['exact_match']:>5.2f} {m['edit_similarity']:>5.2f} "
            f"{m['identifier_f1']:>5.2f} {m['repo_symbol_precision']:>5.2f} "
            f"{m['hallucination_rate']:>5.2f} {s.mean_latency_ms:>8.1f}"
        )


if __name__ == "__main__":
    main()
