"""End-to-end smoke test for Phase 2.

Loads the first N CrossCodeEval-Python instances, runs no-retrieve and
always-retrieve baselines with MockGenerator (no real model required),
computes the full metric suite, and prints a summary. Satisfies the
IMPLEMENTATION_GUIDE §16 Phase 2 validation gate:
    "Both baselines complete 10 CrossCodeEval instances; metrics computed
    without errors."

Run:
    python scripts/03_smoke_pipeline.py --n 10

Pass ``--backend hf --model Qwen/Qwen2.5-Coder-0.5B`` to use a real
HuggingFace model instead of the mock.
"""
from __future__ import annotations

import statistics
import sys
import time
from itertools import islice
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import click  # noqa: E402

from adaptive_retrieval.baselines import (  # noqa: E402
    always_retrieve_baseline,
    no_retrieve_baseline,
)
from adaptive_retrieval.eval.datasets import load_crosscodeeval_python  # noqa: E402
from adaptive_retrieval.generator import MockGenerator, make_generator  # noqa: E402
from adaptive_retrieval.metrics import (  # noqa: E402
    edit_similarity,
    exact_match,
    hallucination_flag,
    identifier_f1,
    mean_latency_ms,
    repository_symbol_precision,
)
from adaptive_retrieval.retriever import BM25Retriever  # noqa: E402
from adaptive_retrieval.static_analysis.analyzer import PredictionAnalyzer  # noqa: E402
from adaptive_retrieval.static_analysis.scope import InFileScopeAnalyzer  # noqa: E402
from adaptive_retrieval.static_analysis.symbol_table import RepositorySymbolTable  # noqa: E402


def run_one(generator, instance, model_family: str):
    """Run both baselines on one instance, returning two records."""
    retriever = BM25Retriever(instance.repo_files)
    repo_syms = RepositorySymbolTable.from_files(instance.repo_files)
    analyzer = PredictionAnalyzer(InFileScopeAnalyzer(), repo_syms)

    records = []
    for name, run_fn in [
        ("C1_no_retrieve", lambda: no_retrieve_baseline(
            generator, instance.x_left, instance.x_right, model_family=model_family
        )),
        ("C2_always_retrieve", lambda: always_retrieve_baseline(
            generator, retriever, instance.x_left, instance.x_right,
            model_family=model_family,
        )),
    ]:
        out = run_fn()
        rec = {
            "instance_id": instance.instance_id,
            "config": name,
            "prediction": out.prediction,
            "retrieved": name == "C2_always_retrieve",
            "latency_ms": out.latency_ms,
            "metrics": {
                "exact_match": exact_match(instance.ground_truth, out.prediction),
                "edit_similarity": edit_similarity(instance.ground_truth, out.prediction),
                "identifier_f1": identifier_f1(instance.ground_truth, out.prediction),
                "repo_symbol_precision": repository_symbol_precision(
                    out.prediction, instance.x_left, instance.x_right, analyzer
                ),
                "hallucinated": hallucination_flag(
                    out.prediction, instance.x_left, instance.x_right, analyzer
                ),
            },
        }
        records.append(rec)
    return records


@click.command()
@click.option("--n", default=10, type=int, help="Number of CCE instances to run")
@click.option(
    "--backend",
    default="mock",
    type=click.Choice(["mock", "hf"]),
    help="mock = no model; hf = real HuggingFace model",
)
@click.option("--model", default="Qwen/Qwen2.5-Coder-0.5B")
@click.option("--max-tokens", default=20, type=int)
@click.option("--model-family", default="qwen")
def main(n: int, backend: str, model: str, max_tokens: int, model_family: str) -> None:
    print(f"[setup] backend={backend} model={model if backend == 'hf' else 'MockGenerator'}")
    if backend == "mock":
        generator = MockGenerator(default_prediction="dummy_prediction()")
    else:
        generator = make_generator(model, backend=backend, max_tokens=max_tokens)

    print(f"[load] streaming first {n} CrossCodeEval-Python instances ...")
    instances = list(islice(load_crosscodeeval_python(), n))
    print(f"  loaded {len(instances)} instances")
    print(f"  example: id={instances[0].instance_id}")
    print(f"           x_left tail: {instances[0].x_left[-60:]!r}")
    print(f"           ground_truth: {instances[0].ground_truth!r}")
    print(f"           repo_files: {len(instances[0].repo_files)} entries")

    print(f"\n[run] running both baselines on {len(instances)} instances ...")
    t0 = time.time()
    all_records = []
    for inst in instances:
        all_records.extend(run_one(generator, inst, model_family))
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s ({elapsed / (2 * n):.2f}s per generation)")

    print("\n[metrics] aggregate per-config:")
    print(f"  {'config':<22} {'EM':>6} {'ES':>6} {'IdF1':>6} {'RSP':>6} {'hall%':>6} {'lat_ms':>8}")
    for cfg in ("C1_no_retrieve", "C2_always_retrieve"):
        recs = [r for r in all_records if r["config"] == cfg]
        em = statistics.mean(int(r["metrics"]["exact_match"]) for r in recs)
        es = statistics.mean(r["metrics"]["edit_similarity"] for r in recs)
        f1 = statistics.mean(r["metrics"]["identifier_f1"] for r in recs)
        rsp = statistics.mean(r["metrics"]["repo_symbol_precision"] for r in recs)
        hall = statistics.mean(int(r["metrics"]["hallucinated"]) for r in recs)
        lat = mean_latency_ms(recs)
        print(f"  {cfg:<22} {em:>6.2f} {es:>6.2f} {f1:>6.2f} {rsp:>6.2f} {hall:>6.2f} {lat:>8.1f}")

    print("\nOK — pipeline wires together on real CrossCodeEval data.")


if __name__ == "__main__":
    main()
