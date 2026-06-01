"""Train a LightGBM Estimator from a (features, scores) .npz.

Holds out a test split (separate from the internal early-stopping split) and
reports HONEST held-out MSE / Pearson, then gates on it:

  * FAIL (non-zero exit) if the Estimator is degenerate — i.e. no better than
    predicting the mean ES. A degenerate Estimator's retrieve/skip gate is
    meaningless, so the pipeline must not benchmark or upload it.
  * WARN if held-out MSE is above the paper-target band (<0.10; CARD ~0.07 with
    a 7B model) but still skillful.

Usage:
    python scripts/02_train_estimator.py \\
        --data data/training_data/codellama_7b.npz \\
        --output models/estimator_codellama_7b.lgb
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import click  # noqa: E402
import numpy as np  # noqa: E402

from adaptive_retrieval.card.estimator import Estimator  # noqa: E402


@click.command()
@click.option("--data", required=True, type=click.Path(exists=True), help="Path to .npz.")
@click.option("--output", required=True, type=click.Path(), help="Output .lgb path.")
@click.option(
    "--val-fraction",
    default=0.05,
    type=float,
    help="Internal early-stopping split carved out inside Estimator.train.",
)
@click.option(
    "--test-fraction",
    default=0.1,
    type=float,
    help="Held-out fraction used only for honest evaluation + gating.",
)
@click.option("--num-boost-round", default=500, type=int)
@click.option("--learning-rate", default=0.1, type=float)
@click.option(
    "--min-skill",
    default=0.02,
    type=float,
    help="Minimum held-out (1 - MSE/baseline_MSE). Below this the Estimator is "
    "degenerate (no better than predicting the mean) and the run FAILS.",
)
@click.option("--seed", default=42, type=int)
def main(
    data: str,
    output: str,
    val_fraction: float,
    test_fraction: float,
    num_boost_round: int,
    learning_rate: float,
    min_skill: float,
    seed: int,
) -> None:
    d = np.load(data)
    features = d["features"]
    scores = d["scores"]
    print(f"[setup] features {features.shape}  scores {scores.shape}")
    print(f"        ES mean={float(np.mean(scores)):.4f}  std={float(np.std(scores)):.4f}")

    n = len(scores)
    if n < 50:
        raise click.ClickException(
            f"Only {n} pairs in {data!r} — far too few to train/evaluate an "
            "Estimator. The calibration step (01) under-produced; do not proceed."
        )

    # Honest held-out test split, separate from the internal early-stopping
    # split that Estimator.train carves out of the *training* portion.
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(n * test_fraction))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    print(f"[split] train {len(train_idx)}  held-out test {len(test_idx)}")

    est = Estimator.train(
        features[train_idx],
        scores[train_idx],
        val_fraction=val_fraction,
        random_state=seed,
        num_boost_round=num_boost_round,
        learning_rate=learning_rate,
    )

    # Held-out metrics — the numbers that actually matter.
    test_true = scores[test_idx]
    test_pred = est.predict(features[test_idx])
    test_mse = float(np.mean((test_pred - test_true) ** 2))
    if np.std(test_pred) > 0 and np.std(test_true) > 0:
        test_pearson = float(np.corrcoef(test_pred, test_true)[0, 1])
    else:
        test_pearson = float("nan")

    # Degeneracy baseline: predict the *training* mean for every test point.
    baseline_pred = float(np.mean(scores[train_idx]))
    baseline_mse = float(np.mean((test_true - baseline_pred) ** 2))
    skill = 1.0 - (test_mse / baseline_mse) if baseline_mse > 0 else 0.0

    print(f"\n[eval] held-out MSE:        {test_mse:.4f}   (CARD ~0.07; gate <0.10)")
    print(f"[eval] held-out Pearson:    {test_pearson:.3f}")
    print(f"[eval] mean-predictor MSE:  {baseline_mse:.4f}")
    print(f"[eval] skill (1-MSE/base):  {skill:.3f}   (degenerate if < {min_skill})")

    # Hard gate: a degenerate Estimator (no better than predicting the mean) is
    # broken — fail the pipeline so it is never benchmarked or uploaded.
    if skill < min_skill:
        raise click.ClickException(
            f"Estimator is degenerate: held-out skill {skill:.3f} < {min_skill}. "
            "It predicts barely better than the training mean, so its retrieve/"
            "skip gate would be meaningless. NOT saving the model. Inspect the "
            "feature distribution and the calibration data (step 01)."
        )

    # Soft gate: above CARD's target band but not degenerate — warn, don't fail
    # (our benchmark/retriever differ from the paper, so some drift is expected).
    if test_mse >= 0.10:
        print(
            "  WARN — held-out MSE above the <0.10 paper-target band. Usable but "
            "weaker than CARD; check feature distribution + model size."
        )
    else:
        print("  PASS — held-out MSE within the paper-target band.")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    est.save(output)
    print(f"\n[done] saved Estimator to {output}")


if __name__ == "__main__":
    main()
