"""Run one (config, dataset, model) combination, write per-instance JSONL.

Usage:
    python scripts/04_run_experiment.py \\
        --config C4_cascade \\
        --dataset crosscodeeval_py \\
        --backend mock \\
        --output results/C4_cascade.crosscodeeval.jsonl \\
        --limit 50

For a real model swap ``--backend mock`` for ``--backend hf`` (Mac/CUDA) or
``--backend vllm`` (GPU node) and pass ``--model Qwen/Qwen2.5-Coder-7B`` etc.

Configs that consult CARD (``C3_card``, ``C4_cascade``) require ``--estimator-path``.
"""
from __future__ import annotations

import json
import sys
from itertools import islice
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import click  # noqa: E402

from adaptive_retrieval.card.estimator import Estimator  # noqa: E402
from adaptive_retrieval.eval.datasets import (  # noqa: E402
    load_crosscodeeval_python,
    load_crosscodelongeval,
    load_repoeval,
)
from adaptive_retrieval.eval.runner import VALID_CONFIGS, run_experiment  # noqa: E402
from adaptive_retrieval.generator import (  # noqa: E402
    CachedGenerator,
    MockGenerator,
    make_generator,
)


_DATASET_LOADERS = {
    "crosscodeeval_py": lambda: load_crosscodeeval_python(),
    "repoeval_line": lambda: load_repoeval(task="line"),
    "repoeval_api": lambda: load_repoeval(task="api"),
    "repoeval_function": lambda: load_repoeval(task="function"),
    "crosscodelongeval_chunk": lambda: load_crosscodelongeval(task="chunk"),
    "crosscodelongeval_function": lambda: load_crosscodelongeval(task="function"),
}


def _resolve_generator(backend: str, model: str, max_tokens: int, cache_dir: str | None):
    if backend == "mock":
        gen = MockGenerator(default_prediction="mock_output()")
        if cache_dir:
            return CachedGenerator(gen, cache_dir, model_name="mock", max_tokens=max_tokens)
        return gen
    gen = make_generator(model, backend=backend, max_tokens=max_tokens)
    if cache_dir:
        return CachedGenerator(gen, cache_dir, model_name=model, max_tokens=max_tokens)
    return gen


@click.command()
@click.option(
    "--config",
    type=click.Choice(VALID_CONFIGS),
    required=True,
    help="Experiment config (C1..C6).",
)
@click.option(
    "--dataset",
    type=click.Choice(list(_DATASET_LOADERS.keys())),
    required=True,
    help="Dataset identifier.",
)
@click.option(
    "--backend",
    type=click.Choice(["mock", "hf", "vllm", "mlx"]),
    default="mock",
    help="Generator backend. mock = no model; mlx = Apple Silicon native.",
)
@click.option("--model", default="Qwen/Qwen2.5-Coder-7B", help="HF model name.")
@click.option("--max-tokens", default=50, type=int)
@click.option("--model-family", default="qwen", type=click.Choice(["qwen", "codellama", "starcoder"]))
@click.option("--estimator-path", default=None, help="LightGBM .lgb path (for C3, C4).")
@click.option("--output", required=True, type=click.Path(), help="JSONL output path.")
@click.option("--limit", default=None, type=int, help="Cap on instances.")
@click.option("--t-rag", default=0.9, type=float)
@click.option("--top-k", default=10, type=int)
@click.option(
    "--batch-size",
    default=1,
    type=int,
    help="Batch size for the C1/C2 generation fast-path (vLLM continuous "
         "batching). 1 = legacy per-instance path. Use ~64-512 on GPU.",
)
@click.option(
    "--cache-dir",
    default="data/generation_cache",
    type=str,
    help="Directory for the generation cache (set empty to disable).",
)
def main(
    config: str,
    dataset: str,
    backend: str,
    model: str,
    max_tokens: int,
    model_family: str,
    estimator_path: str | None,
    output: str,
    limit: int | None,
    t_rag: float,
    top_k: int,
    batch_size: int,
    cache_dir: str,
) -> None:
    print(f"[setup] config={config} dataset={dataset} backend={backend} model={model}")

    instances = _DATASET_LOADERS[dataset]()
    if limit is not None:
        instances = islice(instances, limit)

    generator = _resolve_generator(backend, model, max_tokens, cache_dir or None)
    estimator: Estimator | None = None
    if config in {"C3_card", "C4_cascade"}:
        if not estimator_path:
            raise click.UsageError(f"{config} requires --estimator-path")
        estimator = Estimator.load(estimator_path)

    summary = run_experiment(
        config=config,
        dataset_name=dataset,
        instances=instances,
        generator=generator,
        estimator=estimator,
        output_path=output,
        model_family=model_family,
        t_rag=t_rag,
        top_k=top_k,
        batch_size=batch_size,
    )

    print(f"\n[done] {summary.n_instances} instances written to {output}")
    print(f"  retrieved: {summary.n_retrieved}/{summary.n_instances} "
          f"({summary.percent_retrieval:.1f}%)")
    print(f"  metrics: {json.dumps(summary.metrics, indent=2)}")
    print(f"  mean_latency_ms: {summary.mean_latency_ms:.1f}")

    if isinstance(generator, CachedGenerator):
        total = generator.hits + generator.misses
        if total:
            print(f"  cache: {generator.hits} hits, {generator.misses} misses "
                  f"({100.0 * generator.hits / total:.1f}% hit rate)")


if __name__ == "__main__":
    main()
