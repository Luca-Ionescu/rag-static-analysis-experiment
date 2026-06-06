"""Truncate predictions to the CCE line-completion boundary and recompute ALL
metrics on the truncated text, in place over existing JSONL.

Why this exists
---------------
CrossCodeEval *line* completion scores ONE line, but the generation harness
never stopped the model at the line boundary (vLLM SamplingParams had no
``stop``), so ~50 tokens of FIM over-generation (leaked file content + special
markers) were scored. That inflated edit-distance denominators and polluted the
static-analysis hallucination check, making the raw C1-C4 metrics meaningless
for a line task — e.g. C2 EM 0.46 -> 0.92, ES 0.53 -> 0.96 once truncated.

This is the score-time fix (Option B): nothing on the generator / CARD side is
touched. We truncate the prediction to its first line (cutting any FIM sentinel
that shares it) and recompute exact_match / edit_similarity / identifier_f1 /
repo_symbol_precision / hallucinated from the truncated text. CARD's calibration
and the stored ŝ₀/ŝ₁ are untouched, so every downstream T_RAG sweep stays valid.

The raw output is kept in ``prediction``; ``prediction_truncated`` records
exactly what was scored. The analyzer is rebuilt per instance with the same
per-repo symbol-table union Phase 6 used (run_experiment use_repo_union=True).

Usage:
    python scripts/09_truncate_rescore.py \\
        hf_artifacts/results/codellama_7b/C1_no_retrieve.jsonl \\
        --dataset crosscodeeval_py \\
        --output hf_artifacts/results/codellama_7b_line/C1_no_retrieve.jsonl
"""
from __future__ import annotations

import sys
from dataclasses import asdict, is_dataclass
from itertools import islice
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import click  # noqa: E402
import jsonlines  # noqa: E402

from adaptive_retrieval.eval.datasets import (  # noqa: E402
    DATASET_LOADERS,
    MULTILINE_DATASETS,
    build_repo_chunks_index,
)
from adaptive_retrieval.metrics import (  # noqa: E402
    edit_similarity,
    exact_match,
    identifier_f1,
)
from adaptive_retrieval.static_analysis.analyzer import PredictionAnalyzer  # noqa: E402
from adaptive_retrieval.static_analysis.scope import InFileScopeAnalyzer  # noqa: E402
from adaptive_retrieval.static_analysis.symbol_table import RepositorySymbolTable  # noqa: E402

# FIM / sentinel markers that leak into raw generations when the model is not
# stopped at the completion boundary. If one shares the kept span, cut at it.
_FIM_MARKERS = (
    "<|", "▁<", "<fim", "<PRE>", "<SUF>", "<MID>", "<EOT>", "<MID", "</s>", "<｜",
)


def truncate(text: str, n_lines: int) -> str:
    """Keep the first ``n_lines`` lines (``n_lines <= 0`` keeps all — for
    multi-line function completion), with any FIM sentinel removed."""
    out = text if n_lines <= 0 else "\n".join(text.split("\n")[:n_lines])
    for m in _FIM_MARKERS:
        i = out.find(m)
        if i != -1:
            out = out[:i]
    return out


def _serialise(items) -> list[dict]:
    out: list[dict] = []
    for it in items or ():
        if is_dataclass(it):
            out.append(asdict(it))
        elif isinstance(it, dict):
            out.append(it)
    return out


def _symbol_files(inst, repo_index, use_repo_union) -> dict[str, str]:
    files: dict[str, str] = {}
    if use_repo_union and inst.repository:
        files.update(repo_index.get(inst.repository, {}))
    files.update(inst.repo_files)
    return files


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--dataset", type=click.Choice(list(DATASET_LOADERS)), required=True)
@click.option("--output", required=True, type=click.Path())
@click.option("--limit", default=None, type=int, help="Cap on records (testing).")
@click.option(
    "--truncate-lines",
    default=None,
    type=int,
    help="Lines to keep at scoring time. Default: 1 for line tasks (CCE), "
    "0 (keep full body) for multi-line tasks (repoeval_function).",
)
@click.option(
    "--use-repo-union/--no-repo-union",
    default=True,
    help="Match Phase 6 (run_experiment default is True).",
)
def main(input_path, dataset, output, limit, truncate_lines, use_repo_union) -> None:
    if truncate_lines is None:
        truncate_lines = 0 if dataset in MULTILINE_DATASETS else 1
    print(f"[setup] input={input_path}")
    print(f"        dataset={dataset}  truncate_lines={truncate_lines}  use_repo_union={use_repo_union}")
    index = {inst.instance_id: inst for inst in DATASET_LOADERS[dataset]()}
    print(f"  loaded {len(index)} instances")
    repo_index = build_repo_chunks_index(index.values()) if use_repo_union else {}

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = n_skipped = 0
    es_b = es_a = 0.0
    em_b = em_a = 0
    hall_b = hall_a = 0

    # The per-repo symbol table is identical for every instance of a repo when
    # use_repo_union=True (each instance's chunks are a subset of the union), so
    # cache it: turns a per-record tree-sitter parse of the whole repo into a
    # dict lookup, taking the rescore from ~30 min to a few minutes.
    scope = InFileScopeAnalyzer()
    sym_cache: dict[str, RepositorySymbolTable] = {}

    with jsonlines.open(input_path) as reader, jsonlines.open(out_path, "w") as writer:
        records = islice(reader, limit) if limit else reader
        for rec in records:
            n += 1
            m0 = rec.get("metrics", {})
            es_b += float(m0.get("edit_similarity", 0.0))
            em_b += int(bool(m0.get("exact_match", False)))
            hall_b += int(bool(m0.get("hallucinated", False)))

            inst = index.get(rec.get("instance_id"))
            if inst is None:
                n_skipped += 1
                writer.write(rec)
                continue

            pred = truncate(rec["prediction"], truncate_lines)
            gt = inst.ground_truth

            repo = inst.repository or ""
            symtab = sym_cache.get(repo)
            if symtab is None:
                symtab = RepositorySymbolTable.from_files(
                    _symbol_files(inst, repo_index, use_repo_union)
                )
                if repo:
                    sym_cache[repo] = symtab
            analyzer = PredictionAnalyzer(scope, symtab)
            # Single analyze; derive repo_symbol_precision + hallucinated exactly
            # as metrics.py does, to avoid re-parsing the prediction three times.
            sa = analyzer.analyze(pred, inst.x_left, inst.x_right)
            n_used = sa.n_used_identifiers
            rsp = 1.0 if n_used == 0 else (n_used - len(sa.out_of_scope_identifiers)) / n_used
            hall = bool(sa.significant_out_of_scope) or bool(sa.signature_issues) or bool(sa.import_issues)

            rec["prediction_truncated"] = pred
            rec["metrics"] = {
                "exact_match": exact_match(gt, pred),
                "edit_similarity": edit_similarity(gt, pred),
                "identifier_f1": identifier_f1(gt, pred),
                "repo_symbol_precision": rsp,
                "hallucinated": hall,
            }
            rec["static_out_of_scope"] = list(sa.significant_out_of_scope)
            rec["signature_issues"] = _serialise(sa.signature_issues)
            rec["import_issues"] = _serialise(sa.import_issues)

            es_a += rec["metrics"]["edit_similarity"]
            em_a += int(rec["metrics"]["exact_match"])
            hall_a += int(rec["metrics"]["hallucinated"])
            writer.write(rec)

    print(f"[done] {n} records ({n_skipped} skipped — no instance match)")
    print(f"  EM    {em_b / n:.3f} -> {em_a / n:.3f}")
    print(f"  ES    {es_b / n:.3f} -> {es_a / n:.3f}")
    print(f"  hall  {hall_b / n:.3f} -> {hall_a / n:.3f}")
    print(f"  output: {out_path}")


if __name__ == "__main__":
    main()
