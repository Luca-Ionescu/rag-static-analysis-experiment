"""Re-score an existing RepoEval-function results JSONL with the current
metric logic (dedent body-extraction) — no GPU, no regeneration.

The raw ``prediction`` is stored in every record and the truncation is a
scoring-only change, so all metrics can be recomputed from disk. The static
metrics (repo_symbol_precision, hallucinated) need the analyzer, so we reload
the RepoEval-function instances to rebuild the per-repo symbol table (matching
run_experiment's use_repo_union=True).

Each record is rewritten with:
  - prediction_truncated : the body-extracted prediction (what MAIN scores)
  - metrics              : MAIN, scored on the body-extracted prediction
  - metrics_raw          : scored on the full untruncated prediction

Output goes to a SEPARATE file (``<name>.rescored.jsonl`` by default) so the
original run is preserved and the updated version is identifiable.

Usage:
    python scripts/11_rescore_function.py \
        results/qwen25_1.5b_repoeval_function/C1_no_retrieve.jsonl
    # -> writes results/qwen25_1.5b_repoeval_function/C1_no_retrieve.rescored.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import click  # noqa: E402

from adaptive_retrieval.eval.datasets import (  # noqa: E402
    build_repo_chunks_index,
    load_repoeval,
)
from adaptive_retrieval.metrics import (  # noqa: E402
    edit_similarity,
    exact_match,
    hallucination_flag,
    identifier_f1,
    repository_symbol_precision,
    truncate_to_function_body,
)
from adaptive_retrieval.static_analysis.analyzer import PredictionAnalyzer  # noqa: E402
from adaptive_retrieval.static_analysis.scope import InFileScopeAnalyzer  # noqa: E402
from adaptive_retrieval.static_analysis.symbol_table import RepositorySymbolTable  # noqa: E402


def _score(inst, prediction, analyzer) -> dict:
    return {
        "exact_match": exact_match(inst.ground_truth, prediction),
        "edit_similarity": edit_similarity(inst.ground_truth, prediction),
        "identifier_f1": identifier_f1(inst.ground_truth, prediction),
        "repo_symbol_precision": repository_symbol_precision(
            prediction, inst.x_left, inst.x_right, analyzer
        ),
        "hallucinated": hallucination_flag(
            prediction, inst.x_left, inst.x_right, analyzer
        ),
    }


def _mean(records, key, metric):
    vals = [float(r[key][metric]) for r in records if key in r]
    return sum(vals) / len(vals) if vals else 0.0


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--output", default=None, help="Output JSONL (default: <input>.rescored.jsonl).")
@click.option("--task", default="function", type=click.Choice(["line", "api", "function"]))
@click.option("--use-repo-union/--no-repo-union", default=True,
              help="Match run_experiment (default True).")
def main(input_path, output, task, use_repo_union):
    in_path = Path(input_path)
    out_path = Path(output) if output else in_path.with_suffix(".rescored.jsonl")

    # Load RepoEval instances in order. The runner writes one JSONL record per
    # instance in the SAME order load_repoeval yields, so we match positionally
    # (row i <-> instance i). This is robust to the old non-unique task_id-based
    # instance_id in already-committed results; we also sanity-check the match by
    # comparing ground_truth. (use_repo_union is matched to run_experiment.)
    print(f"[setup] loading repoeval task={task} ...")
    insts_list = list(load_repoeval(task=task))
    print(f"        {len(insts_list)} instances loaded")
    repo_index = build_repo_chunks_index(insts_list) if use_repo_union else {}

    scope = InFileScopeAnalyzer()
    sym_cache: dict[str, RepositorySymbolTable] = {}

    recs_in = [json.loads(l) for l in in_path.open() if l.strip()]
    if len(recs_in) > len(insts_list):
        raise click.ClickException(
            f"{len(recs_in)} records but only {len(insts_list)} instances — cannot align."
        )
    out_recs = []
    n_skip = 0
    n_gt_mismatch = 0
    for idx, rec in enumerate(recs_in):
        inst = insts_list[idx]
        # Sanity: positional match must agree on ground truth.
        if rec.get("ground_truth", "").strip() != inst.ground_truth.strip():
            n_gt_mismatch += 1
        repo = inst.repository or ""
        symtab = sym_cache.get(repo)
        if symtab is None:
            sym_files: dict[str, str] = {}
            if use_repo_union and repo:
                sym_files.update(repo_index.get(repo, {}))
            sym_files.update(inst.repo_files)
            symtab = RepositorySymbolTable.from_files(sym_files)
            if repo:
                sym_cache[repo] = symtab
        analyzer = PredictionAnalyzer(scope, symtab)

        raw = rec["prediction"]
        pred_trunc = truncate_to_function_body(inst.ground_truth, raw)
        rec["prediction_truncated"] = pred_trunc
        rec["metrics"] = _score(inst, pred_trunc, analyzer)        # MAIN
        rec["metrics_raw"] = _score(inst, raw, analyzer)           # secondary
        # Refresh to the (now unique) instance_id so the rescored file can be
        # joined per-instance downstream.
        rec["instance_id"] = inst.instance_id
        out_recs.append(rec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in out_recs:
            f.write(json.dumps(r) + "\n")

    scored = [r for r in out_recs if "metrics_raw" in r]
    n = len(scored)
    print(f"\n[done] {len(out_recs)} records ({n_skip} skipped — no instance match)")
    if n_gt_mismatch:
        print(f"  WARNING: {n_gt_mismatch} records' ground_truth did not match the "
              f"positionally-aligned instance — alignment may be off.")
    print(f"  output: {out_path}")
    if n:
        print(f"\n  {'metric':<22}{'MAIN (body-trunc)':>20}{'RAW':>12}")
        for m in ("exact_match", "edit_similarity", "identifier_f1",
                  "repo_symbol_precision", "hallucinated"):
            print(f"  {m:<22}{_mean(scored,'metrics',m):>20.4f}{_mean(scored,'metrics_raw',m):>12.4f}")


if __name__ == "__main__":
    main()
