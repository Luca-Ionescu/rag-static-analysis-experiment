"""Re-classify identifiers in existing JSONL records against the current
analyzer behaviour, without re-running the Generator.

When the analyzer's definition of ``fires`` / ``hallucinated`` /
``repo_symbol_precision`` changes — e.g., switching to structural-significance
filtering (see ``test_analyzer_strict.py``) — existing JSONL records still
have the old metric values frozen in. This script rebuilds the
``metrics.repo_symbol_precision`` and ``metrics.hallucinated`` fields and
the ``static_unresolved`` / ``static_crossfile`` diagnostic lists from the
predictions themselves, leaving all generator-side fields untouched.

Usage:
    python scripts/07_rescore.py \\
        results/cce_py_1.5b/C1_no_retrieve.jsonl \\
        --dataset crosscodeeval_py \\
        --output results/cce_py_1.5b_strict/C1_no_retrieve.jsonl

To rescore many files in one go:
    for f in results/cce_py_1.5b/*.jsonl; do
        python scripts/07_rescore.py "$f" --dataset crosscodeeval_py \\
            --output "results/cce_py_1.5b_strict/$(basename "$f")"
    done
"""
from __future__ import annotations

import sys
from itertools import islice
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import click  # noqa: E402
import jsonlines  # noqa: E402

from adaptive_retrieval.eval.datasets import (  # noqa: E402
    Instance,
    build_repo_chunks_index,
    load_crosscodeeval_python,
)
from adaptive_retrieval.metrics import (  # noqa: E402
    hallucination_flag,
    repository_symbol_precision,
)
from adaptive_retrieval.static_analysis.analyzer import PredictionAnalyzer  # noqa: E402
from adaptive_retrieval.static_analysis.scope import InFileScopeAnalyzer  # noqa: E402
from adaptive_retrieval.static_analysis.symbol_table import RepositorySymbolTable  # noqa: E402


_DATASET_LOADERS = {
    "crosscodeeval_py": load_crosscodeeval_python,
}


def _build_instance_index(dataset: str) -> dict[str, Instance]:
    """Map instance_id -> Instance so we can rebuild the analyzer per record."""
    loader = _DATASET_LOADERS[dataset]
    return {inst.instance_id: inst for inst in loader()}


def _build_symbol_table_files(
    inst: Instance,
    repo_index: dict[str, dict[str, str]],
    use_repo_union: bool,
) -> dict[str, str]:
    """Compose the file map fed to ``RepositorySymbolTable.from_files``."""
    files: dict[str, str] = {}
    if use_repo_union and inst.repository:
        files.update(repo_index.get(inst.repository, {}))
    files.update(inst.repo_files)
    return files


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--dataset",
    type=click.Choice(list(_DATASET_LOADERS.keys())),
    required=True,
    help="Dataset the JSONL was produced from. Needed to rebuild the per-instance "
    "repo_files / analyzer state.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(),
    help="Output JSONL path. Directory is created if needed.",
)
@click.option("--limit", default=None, type=int, help="Cap on records (for testing).")
@click.option(
    "--use-repo-union/--no-repo-union",
    default=True,
    help="Use the cross-instance per-repo chunk union for the symbol table.",
)
def main(
    input_path: str,
    dataset: str,
    output: str,
    limit: int | None,
    use_repo_union: bool,
) -> None:
    print(f"[setup] input={input_path}")
    print(f"        dataset={dataset}  use_repo_union={use_repo_union}")

    print("[1/3] Loading instance index ...")
    instance_index = _build_instance_index(dataset)
    print(f"  loaded {len(instance_index)} instance objects")
    repo_index: dict[str, dict[str, str]] = (
        build_repo_chunks_index(instance_index.values()) if use_repo_union else {}
    )
    if use_repo_union:
        print(f"  built per-repo chunk index for {len(repo_index)} repositories")

    print("[2/3] Reading + rescoring records ...")
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_skipped = 0
    n_hall_changed = 0
    flagged_before = 0
    flagged_after = 0

    with jsonlines.open(input_path) as reader, jsonlines.open(out_path, "w") as writer:
        records = islice(reader, limit) if limit else reader
        for rec in records:
            n_total += 1
            iid = rec.get("instance_id")
            inst = instance_index.get(iid)
            if inst is None:
                n_skipped += 1
                writer.write(rec)
                continue

            analyzer = PredictionAnalyzer(
                InFileScopeAnalyzer(),
                RepositorySymbolTable.from_files(
                    _build_symbol_table_files(inst, repo_index, use_repo_union)
                ),
            )

            prediction = rec["prediction"]
            old_hall = bool(rec.get("metrics", {}).get("hallucinated", False))
            if old_hall:
                flagged_before += 1

            new_rsp = repository_symbol_precision(
                prediction, inst.x_left, inst.x_right, analyzer
            )
            new_hall = hallucination_flag(
                prediction, inst.x_left, inst.x_right, analyzer
            )
            if new_hall:
                flagged_after += 1
            if old_hall != new_hall:
                n_hall_changed += 1

            sa = analyzer.analyze(prediction, inst.x_left, inst.x_right)

            rec.setdefault("metrics", {})
            rec["metrics"]["repo_symbol_precision"] = new_rsp
            rec["metrics"]["hallucinated"] = new_hall
            # Refresh the unified static signal + loose diagnostic lists.
            rec["static_out_of_scope"] = list(sa.significant_out_of_scope)
            rec["loose_unresolved"] = list(sa.unresolved_identifiers)
            rec["loose_crossfile"] = list(sa.cross_file_identifiers)
            # Strip old field names so rescored JSONL has a clean schema.
            rec.pop("static_unresolved", None)
            rec.pop("static_crossfile", None)

            writer.write(rec)

    print("[3/3] Done")
    print(f"  records read:           {n_total}")
    print(f"  records rescored:       {n_total - n_skipped}")
    if n_skipped:
        print(f"  records skipped (no matching instance): {n_skipped}")
    print(f"  hallucinated rate before: {flagged_before}/{n_total} = "
          f"{100.0 * flagged_before / n_total:.1f}%")
    print(f"  hallucinated rate after:  {flagged_after}/{n_total} = "
          f"{100.0 * flagged_after / n_total:.1f}%")
    print(f"  records where flag changed: {n_hall_changed}")
    print(f"\n  output: {out_path}")


if __name__ == "__main__":
    main()
