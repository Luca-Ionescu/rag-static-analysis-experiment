"""Phase 7 analysis runner.

Consumes per-instance JSONL written by ``scripts/04_run_experiment.py`` and
produces:
  - aggregate metrics per config (JSON to stdout + analysis/summary.json)
  - per-trigger-reason breakdown (cascade only)
  - CARD vs cascade disagreement analysis
  - McNemar's test on hallucinated flag
  - paired bootstrap CI for ES differences
  - T_RAG threshold sweep plot (PNG)

Usage (after running C1, C2, C3, C4 on the same dataset):
    python scripts/06_analysis.py \\
        --c1 results/C1_no_retrieve.cce.jsonl \\
        --c2 results/C2_always_retrieve.cce.jsonl \\
        --c3 results/C3_card.cce.jsonl \\
        --c4 results/C4_cascade.cce.jsonl \\
        --output-dir analysis/cce
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import click  # noqa: E402

from adaptive_retrieval.eval.analysis import (  # noqa: E402
    disagreement_analysis,
    es_paired_bootstrap,
    hallucination_mcnemar,
    load_records,
    threshold_sweep_paired,
    trigger_reason_breakdown,
)
from adaptive_retrieval.eval.runner import aggregate_from_jsonl  # noqa: E402


def _plot_threshold_sweep(rows: list[dict], output_path: Path) -> None:
    """Twin-axis plot: ES vs T_RAG (left), retrieval % vs T_RAG (right)."""
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    ts = [r["t_rag"] for r in rows]
    es = [r["mean_edit_similarity"] for r in rows]
    halls = [r["hallucination_rate"] * 100 for r in rows]
    retrs = [r["percent_retrieval"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
    ax1.set_xlabel("T_RAG")
    ax1.set_ylabel("Edit similarity (mean)", color="#1f3a64")
    ax1.plot(ts, es, marker="o", color="#1f3a64", label="ES")
    ax1.plot(ts, halls, marker="s", color="#c0392b", label="Hallucination % (10×)", linestyle="--")
    ax1.tick_params(axis="y", labelcolor="#1f3a64")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Retrieval %", color="#27ae60")
    ax2.plot(ts, retrs, marker="^", color="#27ae60", label="Retrieval %")
    ax2.tick_params(axis="y", labelcolor="#27ae60")

    fig.suptitle("T_RAG sweep (paired C1/C2 with CARD ŝ₀ values)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


@click.command()
@click.option("--c1", "c1_path", required=True, type=click.Path(exists=True))
@click.option("--c2", "c2_path", required=True, type=click.Path(exists=True))
@click.option("--c3", "c3_path", required=True, type=click.Path(exists=True))
@click.option("--c4", "c4_path", required=True, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path())
@click.option(
    "--t-rag-grid",
    default="0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95",
    help="Comma-separated T_RAG values to sweep.",
)
def main(
    c1_path: str,
    c2_path: str,
    c3_path: str,
    c4_path: str,
    output_dir: str,
    t_rag_grid: str,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading JSONL ...")
    c1 = load_records(c1_path)
    c2 = load_records(c2_path)
    c3 = load_records(c3_path)
    c4 = load_records(c4_path)
    print(f"  C1={len(c1)} C2={len(c2)} C3={len(c3)} C4={len(c4)}")

    print("\n[2/5] Aggregate metrics per config")
    summaries = {
        "C1_no_retrieve": aggregate_from_jsonl(c1_path),
        "C2_always_retrieve": aggregate_from_jsonl(c2_path),
        "C3_card": aggregate_from_jsonl(c3_path),
        "C4_cascade": aggregate_from_jsonl(c4_path),
    }
    summary_table = []
    for name, s in summaries.items():
        m = s.metrics
        summary_table.append(
            {
                "config": name,
                "n": s.n_instances,
                "percent_retrieval": s.percent_retrieval,
                "exact_match": m["exact_match"],
                "edit_similarity": m["edit_similarity"],
                "identifier_f1": m["identifier_f1"],
                "repo_symbol_precision": m["repo_symbol_precision"],
                "hallucination_rate": m["hallucination_rate"],
                "mean_latency_ms": s.mean_latency_ms,
            }
        )
        print(
            f"  {name:<22} n={s.n_instances:<5} retr={s.percent_retrieval:>5.1f}%  "
            f"EM={m['exact_match']:.3f} ES={m['edit_similarity']:.3f} "
            f"F1={m['identifier_f1']:.3f} hall={m['hallucination_rate']:.3f}"
        )

    print("\n[3/5] Per-trigger-reason breakdown (C4 cascade)")
    triggers = trigger_reason_breakdown(c4)
    for row in triggers:
        print(
            f"  {row['trigger_reason']:<22} n={row['n']:<4} "
            f"({row['fraction'] * 100:>5.1f}%) "
            f"ES={row['edit_similarity']:.3f} "
            f"hall={row['hallucination_rate']:.3f}"
        )

    print("\n[4/5] CARD vs cascade disagreement + statistical tests")
    disagreement = disagreement_analysis(c3, c4)
    print(f"  shared n: {disagreement['n_shared']}")
    for k in ("card_no_cascade_no", "card_no_cascade_yes", "card_yes_cascade_yes"):
        b = disagreement.get(k, {})
        if b.get("n", 0):
            print(
                f"  {k:<24} n={b['n']:<5} ({b['fraction'] * 100:>5.1f}%)  "
                f"CARD ES={b['card_mean_es']:.3f}  cascade ES={b['cascade_mean_es']:.3f}  "
                f"hall(c)={b['card_hallucination_rate']:.3f} "
                f"hall(x)={b['cascade_hallucination_rate']:.3f}"
            )

    mcnemar = hallucination_mcnemar(c3, c4)
    print(
        f"  McNemar (hallucinated, CARD vs cascade): p={mcnemar['p_value']:.4g}  "
        f"b={mcnemar['b']}  c={mcnemar['c']}"
    )
    bootstrap = es_paired_bootstrap(c3, c4)
    print(
        f"  Paired bootstrap ES(cascade - CARD): mean={bootstrap['mean_diff']:.4f}  "
        f"95% CI=[{bootstrap['ci_lower']:.4f}, {bootstrap['ci_upper']:.4f}]"
    )

    print("\n[5/5] T_RAG threshold sweep + plot")
    thresholds = [float(x) for x in t_rag_grid.split(",")]
    s_hats_by_id: dict[str, float] = {}
    for r in c3:
        if r.get("s_hat_0") is not None:
            s_hats_by_id[r["instance_id"]] = float(r["s_hat_0"])
    for r in c4:
        if r["instance_id"] not in s_hats_by_id and r.get("s_hat_0") is not None:
            s_hats_by_id[r["instance_id"]] = float(r["s_hat_0"])

    sweep_rows = threshold_sweep_paired(c1, c2, s_hats_by_id, thresholds)
    for row in sweep_rows:
        print(
            f"  t={row['t_rag']:<5.2f}  retr={row['percent_retrieval']:>5.1f}%  "
            f"ES={row['mean_edit_similarity']:.4f}  hall={row['hallucination_rate']:.4f}"
        )
    plot_path = out / "t_rag_sweep.png"
    _plot_threshold_sweep(sweep_rows, plot_path)
    print(f"  wrote {plot_path}")

    # Dump everything to JSON
    summary_path = out / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "aggregate": summary_table,
                "trigger_breakdown": triggers,
                "disagreement": disagreement,
                "mcnemar": mcnemar,
                "es_paired_bootstrap": bootstrap,
                "t_rag_sweep": sweep_rows,
            },
            indent=2,
        )
    )
    print(f"\n[done] wrote {summary_path}")


if __name__ == "__main__":
    main()
