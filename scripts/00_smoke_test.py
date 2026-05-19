"""IMPLEMENTATION_GUIDE Appendix F end-to-end smoke test.

Runs ONE hand-crafted synthetic instance through all six configurations,
asserts each returns a valid record, and prints a summary. Run after Phase 5
of the build order and before Phase 6 (the real experiment matrix) — it's
the last cheap chance to catch integration bugs before burning GPU/MLX time.

With MockGenerator (default) and a trained Estimator, all six configs should
run in under a second. Use ``--backend mlx --model Qwen/Qwen2.5-Coder-0.5B``
for a real-model run (~30 seconds on M4 Pro).

Usage:
    python scripts/00_smoke_test.py \\
        --estimator models/estimator_synthetic.lgb
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import click  # noqa: E402

from adaptive_retrieval.baselines import (  # noqa: E402
    always_retrieve_baseline,
    no_retrieve_baseline,
)
from adaptive_retrieval.card.estimator import Estimator  # noqa: E402
from adaptive_retrieval.card.pipeline import card_pipeline  # noqa: E402
from adaptive_retrieval.cascade import cascade_pipeline  # noqa: E402
from adaptive_retrieval.eval.datasets import Instance  # noqa: E402
from adaptive_retrieval.generator import MockGenerator, make_generator  # noqa: E402
from adaptive_retrieval.metrics import (  # noqa: E402
    edit_similarity,
    exact_match,
    hallucination_flag,
    identifier_f1,
    repository_symbol_precision,
)
from adaptive_retrieval.retriever import BM25Retriever  # noqa: E402
from adaptive_retrieval.static_analysis.analyzer import PredictionAnalyzer  # noqa: E402
from adaptive_retrieval.static_analysis.scope import InFileScopeAnalyzer  # noqa: E402
from adaptive_retrieval.static_analysis.symbol_table import RepositorySymbolTable  # noqa: E402


SYNTHETIC_REPO = {
    "server.py": (
        "class Server:\n"
        "    def __init__(self, port=8080):\n"
        "        self.port = port\n"
        "        self.running = False\n"
        "\n"
        "    async def start(self):\n"
        "        self.running = True\n"
        "\n"
        "    async def stop(self):\n"
        "        self.running = False\n"
    ),
}

INSTANCE = Instance(
    x_left=(
        "import asyncio\n"
        "from server import Server\n"
        "\n"
        "async def main():\n"
        "    s = Server()\n"
        "    "
    ),
    x_right=(
        "\n"
        "    await asyncio.sleep(1)\n"
        "    await s.stop()\n"
    ),
    ground_truth="await s.start()",
    repo_files=SYNTHETIC_REPO,
    instance_id="smoke/0",
    target_file="main.py",
    repository="synthetic",
)


def _compute_metrics(prediction, analyzer, inst):
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


def _assert_record_shape(record, name):
    for key in ("prediction", "retrieved", "metrics", "latency_ms"):
        assert key in record, f"[{name}] missing field: {key}"
    for metric in (
        "exact_match", "edit_similarity", "identifier_f1",
        "repo_symbol_precision", "hallucinated",
    ):
        assert metric in record["metrics"], f"[{name}] missing metric: {metric}"


@click.command()
@click.option("--backend", default="mock", type=click.Choice(["mock", "hf", "vllm", "mlx"]))
@click.option("--model", default="Qwen/Qwen2.5-Coder-0.5B")
@click.option("--max-tokens", default=20, type=int)
@click.option(
    "--estimator",
    "estimator_path",
    default=None,
    type=click.Path(exists=True),
    help="LightGBM .lgb path. Required for C3/C4.",
)
@click.option("--t-rag", default=0.7, type=float)
def main(backend, model, max_tokens, estimator_path, t_rag):
    print("=" * 60)
    print("END-TO-END SMOKE TEST: 1 synthetic instance × 6 configs")
    print("=" * 60)

    print(f"\n[1/3] Build components (backend={backend})")
    if backend == "mock":
        generator = MockGenerator(default_prediction="await s.start()")
    else:
        generator = make_generator(model, backend=backend, max_tokens=max_tokens)

    retriever = BM25Retriever(INSTANCE.repo_files)
    analyzer = PredictionAnalyzer(
        InFileScopeAnalyzer(),
        RepositorySymbolTable.from_files(INSTANCE.repo_files),
    )
    est: Estimator | None = Estimator.load(estimator_path) if estimator_path else None
    if est is None:
        print("  WARN: no --estimator provided; C3/C4 will be skipped.")

    print("\n[2/3] Run all 6 configs")
    results: dict[str, dict] = {}

    out_c1 = no_retrieve_baseline(generator, INSTANCE.x_left, INSTANCE.x_right)
    results["C1"] = {
        "prediction": out_c1.prediction, "retrieved": False, "trigger_reason": "none",
        "latency_ms": out_c1.latency_ms,
        "metrics": _compute_metrics(out_c1.prediction, analyzer, INSTANCE),
    }
    _assert_record_shape(results["C1"], "C1")

    out_c2 = always_retrieve_baseline(generator, retriever, INSTANCE.x_left, INSTANCE.x_right)
    results["C2"] = {
        "prediction": out_c2.prediction, "retrieved": True, "trigger_reason": "always",
        "latency_ms": out_c2.latency_ms,
        "metrics": _compute_metrics(out_c2.prediction, analyzer, INSTANCE),
    }
    _assert_record_shape(results["C2"], "C2")

    if est is not None:
        c = card_pipeline(
            generator, retriever, est,
            INSTANCE.x_left, INSTANCE.x_right,
            t_rag_schedule=[t_rag], t_acc_schedule=[0.8],
        )
        results["C3"] = {
            "prediction": c.prediction, "retrieved": bool(c.retrieved_at_iter),
            "trigger_reason": "card" if c.retrieved_at_iter else "none",
            "s_hat_0": c.s_hats[0] if c.s_hats else None,
            "latency_ms": c.latency_ms,
            "metrics": _compute_metrics(c.prediction, analyzer, INSTANCE),
        }
        _assert_record_shape(results["C3"], "C3")

        x = cascade_pipeline(
            generator, retriever, est, analyzer,
            INSTANCE.x_left, INSTANCE.x_right, t_rag=t_rag,
        )
        results["C4"] = {
            "prediction": x.prediction, "retrieved": x.retrieved,
            "trigger_reason": x.trigger_reason, "s_hat_0": x.s_hat_0,
            "latency_ms": x.latency_ms,
            "metrics": _compute_metrics(x.prediction, analyzer, INSTANCE),
        }
        assert x.trigger_reason in {"none", "card", "static_unresolved", "static_crossfile"}
        _assert_record_shape(results["C4"], "C4")

    # C5 static-only
    sa = analyzer.analyze(out_c1.prediction, INSTANCE.x_left, INSTANCE.x_right)
    if sa.fires:
        results["C5"] = {
            "prediction": out_c2.prediction, "retrieved": True, "trigger_reason": "static",
            "latency_ms": out_c1.latency_ms + out_c2.latency_ms,
            "metrics": _compute_metrics(out_c2.prediction, analyzer, INSTANCE),
        }
    else:
        results["C5"] = {
            "prediction": out_c1.prediction, "retrieved": False, "trigger_reason": "none",
            "latency_ms": out_c1.latency_ms,
            "metrics": _compute_metrics(out_c1.prediction, analyzer, INSTANCE),
        }
    _assert_record_shape(results["C5"], "C5")

    # C6 oracle
    es_no = edit_similarity(INSTANCE.ground_truth, out_c1.prediction)
    es_yes = edit_similarity(INSTANCE.ground_truth, out_c2.prediction)
    chose_rag = es_yes > es_no
    pick = out_c2 if chose_rag else out_c1
    results["C6"] = {
        "prediction": pick.prediction, "retrieved": chose_rag, "trigger_reason": "oracle",
        "latency_ms": out_c1.latency_ms + out_c2.latency_ms,
        "metrics": _compute_metrics(pick.prediction, analyzer, INSTANCE),
    }
    _assert_record_shape(results["C6"], "C6")

    print("\n[3/3] Cross-config sanity + summary")
    assert results["C1"]["retrieved"] is False
    assert results["C2"]["retrieved"] is True

    print(f"\n{'cfg':<6} {'retr':<5} {'trigger':<22} {'ES':>5} {'hall':<5} pred")
    print("-" * 78)
    for k, r in results.items():
        pred_preview = r["prediction"][:30].replace("\n", " ")
        print(
            f"{k:<6} {('yes' if r['retrieved'] else 'no'):<5} "
            f"{r.get('trigger_reason', '-'):<22} {r['metrics']['edit_similarity']:>5.2f} "
            f"{('yes' if r['metrics']['hallucinated'] else 'no'):<5} {pred_preview!r}"
        )
    print(f"\nGround truth: {INSTANCE.ground_truth!r}")
    print("\nOK — smoke passed. Pipeline is integration-correct.")


if __name__ == "__main__":
    main()
