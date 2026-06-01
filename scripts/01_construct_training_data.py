"""Build (features, ES-score) pairs for the CARD Estimator.

The long-running GPU job (IMPLEMENTATION_GUIDE §9.5 / Appendix D.3): sample
(X, y) holes from real-package Python files, K-means deduplicate them, run the
Generator over each X **without retrieval**, and pair the resulting 13-D
features with ES(y, ŷ).

Calibration-grade source is ``the-stack-dedup`` (gated; the corpus CARD used),
streamed and filtered to files with >=3 local imports and >20 non-empty lines.
``the-stack-smol`` is kept only for quick local smoke tests — it is a small
flat sample that the import filter rejects ~97% of, so it produces a
miscalibrated (strawman) Estimator and must not be used for real calibration.

Guardrails (this script ABORTS rather than silently producing a bad estimator):
  * ``--min-files`` — abort BEFORE loading the model if too few valid files
    stream out (gated license not accepted, wrong token, broken filter).
  * ``--min-pairs`` — refuse to write an .npz with too few deduplicated pairs.

Output: an .npz with arrays ``features`` (N, 13) and ``scores`` (N,).
Consume with ``scripts/02_train_estimator.py``.

Example (calibration, on a GPU node):
    HF_TOKEN=hf_xxx python scripts/01_construct_training_data.py \\
        --source the-stack-dedup --backend vllm \\
        --model codellama/CodeLlama-7b-hf --model-family codellama \\
        --max-tokens 50 --output data/training_data/codellama_7b.npz
"""
from __future__ import annotations

import os
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


def _stream_the_stack_dedup(target_valid_files: int, max_scan: int) -> list[str]:
    """Stream the gated ``bigcode/the-stack-dedup`` Python subset, keeping only
    files that pass the CARD filter (``is_valid_file``). Returns already-filtered
    contents so the caller can guardrail on the valid count before the GPU step.

    Gated: the HF account behind ``HF_TOKEN`` must have accepted the license at
    https://huggingface.co/datasets/bigcode/the-stack-dedup .
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise click.UsageError(
            "Streaming the-stack-dedup requires the `datasets` package."
        ) from e
    from adaptive_retrieval.card.train_data import is_valid_file

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise click.UsageError(
            "the-stack-dedup is gated: set HF_TOKEN to a token whose HF account "
            "has accepted the license at "
            "https://huggingface.co/datasets/bigcode/the-stack-dedup"
        )
    print(
        f"  [stream] bigcode/the-stack-dedup (data/python): target "
        f"{target_valid_files:,} valid files, max_scan {max_scan:,}"
    )
    try:
        ds = load_dataset(
            "bigcode/the-stack-dedup",
            data_dir="data/python",
            split="train",
            streaming=True,
            token=token,
        )
    except Exception as e:  # surface gated/access/license errors clearly
        raise click.UsageError(
            "Could not open the-stack-dedup. Check that the license is accepted "
            f"for this token and HF_TOKEN is valid. Underlying error: {e}"
        ) from e

    valid: list[str] = []
    scanned = 0
    t0 = time.time()
    for ex in ds:
        scanned += 1
        content = ex.get("content")
        if content and is_valid_file(content):
            valid.append(content)
            if len(valid) >= target_valid_files:
                print(
                    f"  [stream] target reached: {len(valid):,} valid files "
                    f"after {scanned:,} scanned ({time.time() - t0:.0f}s)"
                )
                break
        if scanned % 25000 == 0:
            frac = len(valid) / scanned
            print(
                f"  [stream] scanned {scanned:,}  valid {len(valid):,} "
                f"({frac:.1%} pass)  {time.time() - t0:.0f}s"
            )
        if scanned >= max_scan:
            print(
                f"  [stream] hit max_scan {max_scan:,}; stopping with "
                f"{len(valid):,} valid files"
            )
            break
    return valid


def _load_python_files(
    source: str, limit: int | None, max_scan: int
) -> list[str]:
    if source == "the-stack-dedup":
        return _stream_the_stack_dedup(limit or 15000, max_scan)
    if source == "the-stack-smol":
        print(
            "  [warn] the-stack-smol is a small flat sample — NOT calibration-"
            "grade. Use --source the-stack-dedup for real runs."
        )
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise click.UsageError(
                "Loading the-stack-smol requires the `datasets` package."
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
    default="the-stack-dedup",
    help="'the-stack-dedup' (gated, calibration-grade), 'the-stack-smol' "
    "(small smoke-test sample), or a path to a dir/file of .py files.",
)
@click.option(
    "--file-limit",
    default=None,
    type=int,
    help="Target number of VALID files to collect (streaming sources) / cap on "
    "files read (path sources). Default 15000 for the-stack-dedup.",
)
@click.option(
    "--max-scan",
    default=3_000_000,
    type=int,
    help="Safety bound on records scanned while streaming a gated dataset.",
)
@click.option("--backend", default="mlx", type=click.Choice(["mlx", "hf", "vllm"]))
@click.option("--model", required=True, help="HF model identifier.")
@click.option("--max-tokens", default=20, type=int)
@click.option("--model-family", default="qwen", type=click.Choice(["qwen", "codellama", "starcoder"]))
@click.option("--n-pairs", default=50_000, type=int, help="Upper cap on pairs after dedup.")
@click.option("--per-file", default=25, type=int, help="Pairs sampled per valid file.")
@click.option("--batch-size", default=32, type=int, help="Generator batch size.")
@click.option(
    "--min-files",
    default=8000,
    type=int,
    help="Abort BEFORE the GPU step if fewer valid files are collected "
    "(streaming sources only).",
)
@click.option(
    "--min-pairs",
    default=20000,
    type=int,
    help="Refuse to write the .npz if fewer deduplicated pairs result.",
)
@click.option("--output", required=True, type=click.Path(), help="Output .npz path.")
@click.option("--seed", default=42, type=int)
def main(
    source: str,
    file_limit: int | None,
    max_scan: int,
    backend: str,
    model: str,
    max_tokens: int,
    model_family: str,
    n_pairs: int,
    per_file: int,
    batch_size: int,
    min_files: int,
    min_pairs: int,
    output: str,
    seed: int,
) -> None:
    print(f"[setup] source={source!r} backend={backend} model={model!r}")
    print(f"        n_pairs(cap)={n_pairs} per_file={per_file} batch={batch_size}")
    print(f"        guardrails: min_files={min_files} min_pairs={min_pairs}")

    streaming = source == "the-stack-dedup"

    t0 = time.time()
    print("\n[1/3] Loading Python files ...")
    files = _load_python_files(source, file_limit, max_scan)
    print(f"  loaded {len(files)} valid files in {time.time() - t0:.1f}s")

    # Guardrail 1: fail before the (expensive) model load + GPU generation if
    # the source did not yield enough real-package files. This is what stops a
    # silent strawman-Estimator run.
    if streaming and len(files) < min_files:
        raise click.ClickException(
            f"Collected only {len(files)} valid files (--min-files={min_files}). "
            "Aborting BEFORE loading the model to avoid training a strawman "
            "Estimator.\n"
            "  Likely causes: the-stack-dedup license not accepted for this "
            "HF_TOKEN, token missing/wrong, or the >=3-local-imports filter "
            "rejecting everything. Fix the source — do NOT relax the filter."
        )

    print("\n[2/3] Loading generator ...")
    t1 = time.time()
    generator = make_generator(model, backend=backend, max_tokens=max_tokens)
    print(f"  ready in {time.time() - t1:.1f}s")

    print(
        f"\n[3/3] Sampling + dedup + generating "
        f"(cap {n_pairs} pairs, may take hours) ..."
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

    # Guardrail 2: refuse to persist a calibration set too small to train a
    # trustworthy Estimator. Better to fail the pipeline here than to upload a
    # miscalibrated CARD baseline.
    if len(scores) < min_pairs:
        raise click.ClickException(
            f"Only {len(scores)} deduplicated pairs (--min-pairs={min_pairs}). "
            "Refusing to write a calibration set this small — it would yield a "
            "miscalibrated CARD baseline and make the experiment unreliable. "
            f"Not writing {output!r}."
        )

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, features=features, scores=scores)
    print(f"\n[done] wrote {len(scores)} (features, scores) pairs to {output}")
    print(f"  features shape: {features.shape}  scores shape: {scores.shape}")
    print(f"  mean ES: {float(np.mean(scores)):.4f}  std: {float(np.std(scores)):.4f}")


if __name__ == "__main__":
    main()
