"""Build (features, ES-score) pairs for the CARD Estimator.

This is the long-running GPU job referenced in IMPLEMENTATION_GUIDE §9.5 /
Appendix D.3: sample (X, y) pairs from The Stack-smol, K-means deduplicate
them, run the Generator over each X, and pair the resulting features with
ES(y, ŷ). On GPU this is ~24h for 250k pairs; locally with mlx-lm + a 0.5B
model it's roughly 5–10h for 50k pairs.

Output: an .npz file with arrays ``features`` (N, 13) and ``scores`` (N,).
Consume with ``scripts/02_train_estimator.py``.

Example:
    python scripts/01_construct_training_data.py \\
        --backend mlx --model Qwen/Qwen2.5-Coder-0.5B \\
        --n-pairs 50000 --output data/training_data/qwen25_05b.npz
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import click  # noqa: E402
import numpy as np  # noqa: E402

from adaptive_retrieval.card.train_data import construct_training_data  # noqa: E402
from adaptive_retrieval.generator import make_generator  # noqa: E402


def _load_python_files(source: str, limit: int | None) -> list[str]:
    if source == "the-stack-smol":
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise click.UsageError(
                "Loading the-stack-smol requires the `datasets` package. "
                "Install it via the project's requirements."
            ) from e
        ds = load_dataset(
            "bigcode/the-stack-smol", data_dir="data/python", split="train"
        )
        contents = []
        for ex in ds:
            contents.append(ex["content"])
            if limit and len(contents) >= limit:
                break
        return contents
    # Otherwise treat as a path: a directory of .py files, or a single .py file
    p = Path(source)
    if not p.exists():
        raise click.UsageError(f"Source not found: {source!r}")
    if p.is_file():
        return [p.read_text(encoding="utf-8")]
    out = []
    for f in p.rglob("*.py"):
        try:
            out.append(f.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
        if limit and len(out) >= limit:
            break
    return out


@click.command()
@click.option(
    "--source",
    default="the-stack-smol",
    help="Either 'the-stack-smol' (HuggingFace) or a path to a dir/file of .py files.",
)
@click.option(
    "--file-limit",
    default=None,
    type=int,
    help="Cap on Python files loaded from source.",
)
@click.option("--backend", default="mlx", type=click.Choice(["mlx", "hf", "vllm"]))
@click.option("--model", required=True, help="HF model identifier.")
@click.option("--max-tokens", default=20, type=int)
@click.option("--model-family", default="qwen", type=click.Choice(["qwen", "codellama", "starcoder"]))
@click.option("--n-pairs", default=50_000, type=int, help="Target pairs after dedup.")
@click.option("--per-file", default=25, type=int, help="Pairs sampled per valid file.")
@click.option("--batch-size", default=32, type=int, help="Generator batch size.")
@click.option("--output", required=True, type=click.Path(), help="Output .npz path.")
@click.option("--seed", default=42, type=int)
def main(
    source: str,
    file_limit: int | None,
    backend: str,
    model: str,
    max_tokens: int,
    model_family: str,
    n_pairs: int,
    per_file: int,
    batch_size: int,
    output: str,
    seed: int,
) -> None:
    print(f"[setup] source={source!r} backend={backend} model={model!r}")
    print(f"        n_pairs={n_pairs} per_file={per_file} batch={batch_size}")

    t0 = time.time()
    print("\n[1/3] Loading Python files ...")
    files = _load_python_files(source, file_limit)
    print(f"  loaded {len(files)} files in {time.time() - t0:.1f}s")

    print("\n[2/3] Loading generator ...")
    t1 = time.time()
    generator = make_generator(model, backend=backend, max_tokens=max_tokens)
    print(f"  ready in {time.time() - t1:.1f}s")

    print(
        f"\n[3/3] Sampling + dedup + generating "
        f"(target {n_pairs} pairs, may take hours) ..."
    )
    t2 = time.time()
    features, scores = construct_training_data(
        generator,
        files,
        n_target_pairs=n_pairs,
        per_file=per_file,
        batch_size=batch_size,
        model_family=model_family,
        seed=seed,
    )
    print(f"  generated {len(scores)} pairs in {(time.time() - t2) / 60:.1f} min")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, features=features, scores=scores)
    print(f"\n[done] wrote {len(scores)} (features, scores) pairs to {output}")
    print(f"  features shape: {features.shape}  scores shape: {scores.shape}")
    print(f"  mean ES: {float(np.mean(scores)):.4f}  std: {float(np.std(scores)):.4f}")


if __name__ == "__main__":
    main()
