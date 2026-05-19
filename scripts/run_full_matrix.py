"""Phase 6 orchestrator: run the full experiment matrix in one command.

Runs every (config, dataset) combination from IMPLEMENTATION_GUIDE §14 and
optionally the ablations. With the generation cache active, each config
beyond the first re-uses ZS/RAG generations rather than recomputing.

Usage:
    python scripts/run_full_matrix.py \\
        --backend mlx --model Qwen/Qwen2.5-Coder-0.5B \\
        --estimator-path models/estimator_qwen25_05b.lgb \\
        --datasets crosscodeeval_py \\
        --limit 100   # remove for the full ~2,665-instance run

Outputs:
    results/<dataset>/<config>.jsonl  for each combination
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import click

SCRIPT = Path(__file__).resolve().parent / "04_run_experiment.py"

ALL_CONFIGS = (
    "C1_no_retrieve",
    "C2_always_retrieve",
    "C3_card",
    "C4_cascade",
    "C5_static_only",
    "C6_oracle",
)
ALL_DATASETS = (
    "crosscodeeval_py",
    "repoeval_line",
    "repoeval_api",
)


def _run(config: str, dataset: str, args: dict) -> Path:
    out_path = Path(args["output_root"]) / dataset / f"{config}.jsonl"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--config", config,
        "--dataset", dataset,
        "--backend", args["backend"],
        "--model", args["model"],
        "--max-tokens", str(args["max_tokens"]),
        "--model-family", args["model_family"],
        "--t-rag", str(args["t_rag"]),
        "--top-k", str(args["top_k"]),
        "--cache-dir", args["cache_dir"],
        "--output", str(out_path),
    ]
    if args["limit"] is not None:
        cmd += ["--limit", str(args["limit"])]
    if config in {"C3_card", "C4_cascade"}:
        if not args["estimator_path"]:
            raise click.UsageError(
                f"{config} requires --estimator-path (none provided)"
            )
        cmd += ["--estimator-path", args["estimator_path"]]
    print(f"\n>>> {config} × {dataset}")
    print("    " + " ".join(cmd))
    t0 = time.time()
    rc = subprocess.call(cmd)
    dt = time.time() - t0
    print(f"    done in {dt:.1f}s (rc={rc})")
    if rc != 0:
        raise RuntimeError(f"Subprocess failed: {config} × {dataset} (rc={rc})")
    return out_path


@click.command()
@click.option(
    "--datasets",
    default=",".join(ALL_DATASETS),
    help="Comma-separated dataset list.",
)
@click.option(
    "--configs",
    default=",".join(ALL_CONFIGS),
    help="Comma-separated config list.",
)
@click.option("--backend", default="mlx", type=click.Choice(["mock", "hf", "vllm", "mlx"]))
@click.option("--model", required=True, help="Model name (HF identifier).")
@click.option("--estimator-path", default=None, type=click.Path(exists=True))
@click.option("--max-tokens", default=50, type=int)
@click.option("--model-family", default="qwen", type=click.Choice(["qwen", "codellama", "starcoder"]))
@click.option("--limit", default=None, type=int, help="Cap per dataset (omit for full run).")
@click.option("--t-rag", default=0.9, type=float)
@click.option("--top-k", default=10, type=int)
@click.option("--cache-dir", default="data/generation_cache", type=str)
@click.option("--output-root", default="results", type=click.Path())
def main(**kwargs):
    datasets = kwargs.pop("datasets").split(",")
    configs = kwargs.pop("configs").split(",")
    for c in configs:
        if c not in ALL_CONFIGS:
            raise click.UsageError(f"Unknown config: {c!r}")
    for d in datasets:
        if d not in ALL_DATASETS:
            raise click.UsageError(f"Unknown dataset: {d!r}")

    print("=" * 60)
    print(f"FULL MATRIX  {len(configs)} configs × {len(datasets)} datasets")
    print(f"  backend={kwargs['backend']}  model={kwargs['model']}")
    print(f"  cache_dir={kwargs['cache_dir']}  output_root={kwargs['output_root']}")
    if kwargs.get("limit") is not None:
        print(f"  limit={kwargs['limit']} per dataset (pilot mode)")
    print("=" * 60)

    overall_t0 = time.time()
    written: list[Path] = []
    for dataset in datasets:
        # Order configs so the cache warms cheap paths first (C1, C2).
        ordered = sorted(configs, key=lambda c: ALL_CONFIGS.index(c))
        for config in ordered:
            try:
                p = _run(config, dataset, kwargs)
                written.append(p)
            except RuntimeError as e:
                print(f"FAIL: {e}", file=sys.stderr)
                # Continue with the next (config, dataset) — the rest of the
                # matrix is still useful, partial results are still valuable.
                continue
    elapsed = time.time() - overall_t0
    print("\n" + "=" * 60)
    print(f"[done] {len(written)} runs complete in {elapsed / 60:.1f} min")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
